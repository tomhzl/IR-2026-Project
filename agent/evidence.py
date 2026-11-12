"""Select continuous source excerpts from the retrieved documents."""

import re
import time

from agent.retrieval import query_terms


def excerpt_from(body, passage_start, terms, source, width=1200):
    """Choose a dense window and copy it from the source file."""
    if len(body) <= width:
        offset = 0
    else:
        positions = range(0, len(body) - width + 1, 200)
        offset = max(positions, key=lambda pos: sum(
            1 for term in terms if term in body[pos:pos + width].casefold()
        ))
    start = passage_start + offset
    return start, source[start:start + width]


def anchored_windows(source, terms):
    """Find text around question terms, including occurrences missed by FTS ranking."""
    lowered = source.casefold()
    windows = []
    width = 1200
    for term in terms:
        positions = [match.start() for match in re.finditer(r'\b' + re.escape(term) + r'\b', lowered)]
        if not positions:
            continue
        # Repeated boilerplate can dominate a document; sample its full extent.
        if len(positions) > 60:
            positions = [positions[i * (len(positions) - 1) // 59] for i in range(60)]
        for position in positions:
            start = max(0, position - 900)
            windows.append((start, source[start:start + width]))
    return windows


def collect_snippets(db, doc_ids, best_passage, question, tools):
    """Read multiple candidate passages from the original top-ranked documents."""
    started = time.monotonic()
    all_terms = query_terms(question)
    focus_terms = query_terms(question[-500:])[-10:]
    expression = ' OR '.join(f'"{term}"' for term in focus_terms)
    snippets = []
    for doc_index, doc_id in enumerate(doc_ids[:10]):
        source = tools.corpus.text(doc_id)
        relevant = [best_passage[doc_id]]
        if expression:
            limit = 6 if doc_index == 0 else 10
            relevant.extend((int(start), body) for start, body in db.execute(
                'SELECT start, body FROM passages '
                'WHERE passages MATCH ? AND doc_id = ? ORDER BY rank LIMIT ?',
                (expression, doc_id, limit),
            ))
        if doc_index == 0:
            # Preserve BM25's top-document coverage; anchors otherwise displace
            # answer-bearing passages in long, highly relevant documents.
            candidates = [(0, source[:1000])] + relevant
            width = 1000
        else:
            # The requested value may appear only once, far from the document
            # opening. Search its original positions as well as FTS passages.
            anchors = anchored_windows(source, list(dict.fromkeys(focus_terms + all_terms))[-18:])
            def score(item):
                body = item[1].casefold()
                return (3 * sum(term in body for term in focus_terms)
                        + sum(term in body for term in all_terms))
            relevant.extend(anchors)
            relevant.sort(key=score, reverse=True)
            candidates = [(0, source[:1200])]
            if len(source) > 1200:
                candidates.append((1000, source[1000:2200]))
            candidates.extend(relevant)
            width = 1200
        seen_starts = set()
        kept = 0
        for passage_start, body in candidates:
            if passage_start == 0:
                start, text = 0, source[:width]
            else:
                start, text = excerpt_from(body, passage_start, focus_terms, source, width)
            if len(text) < 20 or any(abs(start - old) < 400 for old in seen_starts):
                continue
            seen_starts.add(start)
            snippets.append({'id': f'S{len(snippets) + 1}', 'doc_id': doc_id,
                             'start': start, 'text': text})
            kept += 1
            if kept >= (6 if doc_index == 0 else 3):
                break
    tools.record_retrieval(
        stage='read_evidence_candidates',
        arguments={'focus_query': expression, 'top_docs': min(len(doc_ids), 10)},
        output={'snippets': snippets},
        doc_ids=list(dict.fromkeys(item['doc_id'] for item in snippets)),
        elapsed_ms=(time.monotonic() - started) * 1000,
    )
    return snippets


def select_no_answer_evidence(snippets, doc_ids, question):
    """Keep diverse, question-related source excerpts when answering abstains."""
    terms = query_terms(question)
    focus = set(query_terms(question[-500:])[-10:])
    doc_rank = {doc_id: rank for rank, doc_id in enumerate(doc_ids)}

    def score(item):
        text = item['text'].casefold()
        return (2 * sum(term in text for term in focus)
                + sum(term in text for term in terms)
                - doc_rank[item['doc_id']] * 0.25)

    selected = []
    seen_docs = set()
    for item in sorted(snippets, key=score, reverse=True):
        if item['doc_id'] in seen_docs:
            continue
        selected.append(item)
        seen_docs.add(item['doc_id'])
        if len(selected) == 3:
            break
    return selected
