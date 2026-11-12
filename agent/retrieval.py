"""BM25 query construction and document ranking."""

import re
import time


STOP_WORDS = {
    'about', 'according', 'after', 'also', 'among', 'another', 'because',
    'before', 'between', 'could', 'during', 'first', 'following', 'from',
    'have', 'into', 'known', 'looking', 'more', 'name', 'named', 'other',
    'please', 'question', 'same', 'their', 'there', 'these', 'those', 'through',
    'what', 'when', 'where', 'which', 'while', 'with', 'would', 'year', 'years',
}


def query_terms(question):
    words = re.findall(r'[A-Za-z][A-Za-z0-9]*|\d{4}', question)
    return list(dict.fromkeys(word.lower() for word in words
                              if (len(word) >= 4 or word.isupper())
                              and word.lower() not in STOP_WORDS))


def queries_for(question):
    terms = query_terms(question)
    if not terms:
        return []
    groups = [terms[:8], terms[-8:], sorted(terms, key=len, reverse=True)[:8]]
    queries = []
    for group in groups:
        # Quoting makes each term literal FTS5 syntax, including numbers.
        expression = ' OR '.join(f'"{term}"' for term in group)
        if expression not in queries:
            queries.append(expression)
    return queries


def retrieve_bm25(db, question, tools):
    """Return the original BM25/RRF ranking, reusable for controlled comparisons."""
    scores = {}
    best_passage = {}
    for expression in queries_for(question):
        started = time.monotonic()
        rows = db.execute(
            'SELECT doc_id, start, body, rank FROM passages '
            'WHERE passages MATCH ? ORDER BY rank LIMIT 100', (expression,),
        ).fetchall()
        seen = set()
        hits = []
        for doc_id, start, body, rank in rows:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            position = len(hits) + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (30 + position)
            best_passage.setdefault(doc_id, (int(start), body))
            hits.append({'doc_id': doc_id, 'rank': position, 'bm25': rank})
        tools.record_retrieval(
            stage='fts5_bm25', arguments={'query': expression, 'limit': 100},
            output={'hits': hits}, doc_ids=[hit['doc_id'] for hit in hits],
            elapsed_ms=(time.monotonic() - started) * 1000,
        )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0])), best_passage
