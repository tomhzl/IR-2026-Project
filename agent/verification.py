"""Answer verification with source-grounded checks and conservative type guards."""

import re
import time
import math
import os
from collections import Counter

from agent.answer import ModelResponseError


def verification_schema(snippets):
    ids = [item['id'] for item in snippets]
    span = {'type': 'object', 'properties': {
        'snippet_id': {'type': 'string', 'enum': ids}, 'quote': {'type': 'string', 'maxLength': 240}},
        'required': ['snippet_id', 'quote'], 'additionalProperties': False}
    check = {'type': 'object', 'properties': {
        'claim': {'type': 'string', 'maxLength': 140},
        'status': {'type': 'string', 'enum': ['supported', 'missing', 'contradicted']},
        'snippet_id': {'type': 'string', 'enum': ids + ['']}, 'quote': {'type': 'string', 'maxLength': 240}},
        'required': ['claim', 'status', 'snippet_id', 'quote'], 'additionalProperties': False}
    properties = {'supported': {'type': 'boolean'}, 'answer_type': {'type': 'string', 'maxLength': 80},
                  'type_matches': {'type': 'boolean'},
                  'answer_support': {'type': 'array', 'items': span, 'maxItems': 3},
                  'checks': {'type': 'array', 'items': check, 'maxItems': 4},
                  'missing_clues': {'type': 'array', 'items': {'type': 'string', 'maxLength': 140}, 'maxItems': 3},
                  'unverified_background': {'type': 'array', 'items': {'type': 'string', 'maxLength': 140}, 'maxItems': 3}}
    return {'type': 'json_schema', 'json_schema': {'name': 'answer_verification', 'strict': True,
            'schema': {'type': 'object', 'properties': properties,
                       'required': list(properties), 'additionalProperties': False}}}


POLICY = 'source-grounded-v9'


def source_span(quote, text):
    """Resolve typography changes to a real contiguous span; preserve words/numbers."""
    pieces = [part.strip() for part in re.split(r'\.{3}|…', quote) if part.strip()]
    if len(pieces) >= 2:
        cursor = 0
        beginning = None
        for piece in pieces:
            span = source_span(piece, text[cursor:])
            if span is None:
                return None
            position = text.find(span, cursor)
            if beginning is None:
                beginning = position
            cursor = position + len(span)
        return text[beginning:cursor] if cursor - beginning <= 1200 else None
    token = re.compile(r'[+-]?\d+(?:[.,]\d+)*|[^\W\d_]+(?:[\'’][^\W\d_]+)*|[$€£%+=]')
    query = [m.group().casefold().replace('’', "'") for m in token.finditer(quote)]
    source = list(token.finditer(text))
    if not query:
        return None
    words = [m.group().casefold().replace('’', "'") for m in source]
    for position in range(len(words) - len(query) + 1):
        if words[position:position + len(query)] == query:
            return text[source[position].start():source[position + len(query) - 1].end()]
    return None


def materialize_audit(audit, snippets, proposed_answer=''):
    """Keep model decisions; resolve their proof spans to actual source substrings."""
    by_id = {s['id']: s for s in snippets}
    result = dict(audit)
    for key in ['answer_support', 'checks']:
        items = audit.get(key)
        if not isinstance(items, list):
            continue
        result[key] = []
        for item in items:
            if not isinstance(item, dict):
                result[key].append(item)
                continue
            item = dict(item)
            if (isinstance(item.get('snippet_id'), str) and item['snippet_id'] in by_id
                    and isinstance(item.get('quote'), str)):
                resolved = source_span(item['quote'], by_id[item['snippet_id']]['text'])
                if resolved is not None:
                    item['quote'] = resolved
            result[key].append(item)
    # A positive verdict may omit the name span while correctly citing its role.
    # Extract that literal name from the same source; never synthesize a value.
    if (audit.get('supported') is True and result.get('answer_support') == [] and proposed_answer):
        needed = {w for w in re.findall(r'[^\W\d_]+', proposed_answer.casefold()) if len(w) > 1}
        for check in result.get('checks', []):
            if not isinstance(check, dict) or check.get('status') != 'supported':
                continue
            words = set(re.findall(r'[^\W\d_]+', str(check.get('claim', '')).casefold()))
            snippet_id = check.get('snippet_id')
            if not needed or not needed <= words or snippet_id not in by_id:
                continue
            span = source_span(proposed_answer, by_id[snippet_id]['text'])
            if span is not None:
                result['answer_support'] = [{'snippet_id': snippet_id, 'quote': span}]
                break
    return result


def focus_snippets(question, answer, snippets):
    """Keep candidate-document context, including nearby excerpts using pronouns."""
    if len(snippets) <= 14:
        return list(snippets)
    stop = {'this', 'that', 'with', 'from', 'were', 'have', 'been', 'which', 'what',
            'there', 'their', 'between', 'before', 'after', 'would', 'about', 'name',
            'first', 'last', 'same', 'also', 'year', 'years', 'they', 'them', 'than'}
    terms = set(re.findall(r'[a-z]{4,}', question.casefold())) - stop
    answer_terms = set(re.findall(r'[a-z]{4,}', answer.casefold())) - stop
    documents = {}
    for item in snippets:
        documents.setdefault(item['doc_id'], []).append(item)
    doc_text = {doc_id: ' '.join(s['text'] for s in items).casefold()
                for doc_id, items in documents.items()}
    df = Counter(term for text in doc_text.values() for term in terms if term in text)
    weights = {term: math.log(1 + len(documents) / (1 + df[term])) for term in terms}

    def score(item):
        text = item['text'].casefold()
        return sum(weight for term, weight in weights.items() if term in text) + 4 * sum(
            term in text for term in answer_terms)

    exact_docs = [d for d, text in doc_text.items() if answer.casefold() in text]
    related = exact_docs or [d for d, text in doc_text.items() if any(t in text for t in answer_terms)]
    if not related:
        related = list(documents)
    related.sort(key=lambda d: max(score(s) for s in documents[d]), reverse=True)
    selected = []
    for doc_id in related[:4]:
        selected.extend(sorted(documents[doc_id], key=score, reverse=True)[:3])
    used_docs = {s['doc_id'] for s in selected}
    for item in sorted(snippets, key=score, reverse=True):
        if item['doc_id'] in used_docs:
            continue
        selected.append(item)
        used_docs.add(item['doc_id'])
        if len(used_docs) >= min(len(related), 4) + 2:
            break
    return selected[:14]


def answer_type_error(question, answer):
    asks = list(re.finditer(r'\b(?:who|what|which|when|where|how)\b', question, re.I))
    request = question[asks[-1].start():] if asks else question[-350:]
    person = bool(re.match(r'who\b', request, re.I) or re.search(
        r'first (?:and last name|name and surname)|first name and (?:last name|surname)', question, re.I))
    if person and not re.search(r'[A-Za-z\u00c0-\u024f]{2}', answer):
        return 'The question requests a person; the answer is a number or date.'
    if re.search(r'\b(?:name|title)\b', request, re.I) and not re.search(r'[A-Za-z\u00c0-\u024f]', answer):
        return 'The question requests a name or title; the answer is numeric.'
    return None


def validate_audit(audit, snippets, question, proposed_answer=''):
    """A model's supported=true is insufficient without valid supporting spans."""
    audit = materialize_audit(audit, snippets, proposed_answer)
    by_id = {item['id']: item for item in snippets}
    reasons = []
    evidence_ids = []
    if audit.get('supported') is not True:
        reasons.append('Verifier did not establish support.')
    if audit.get('type_matches') is not True:
        reasons.append('Requested answer type was not verified.')
    if not isinstance(audit.get('missing_clues'), list) or audit['missing_clues']:
        reasons.append('Essential clues are missing or the audit is incomplete.')

    def valid_span(item, minimum=12):
        if not isinstance(item, dict):
            return False
        snippet_id, quote = item.get('snippet_id'), item.get('quote')
        if not isinstance(snippet_id, str) or snippet_id not in by_id or not isinstance(quote, str):
            return False
        # Permit whitespace changes from PDF/table formatting, not new words.
        quote = ' '.join(quote.split())
        if not minimum <= len(quote) <= 1200 or source_span(quote, by_id[snippet_id]['text']) is None:
            return False
        if snippet_id not in evidence_ids:
            evidence_ids.append(snippet_id)
        return True

    direct = audit.get('answer_support')
    if not isinstance(direct, list) or not direct or not all(valid_span(item, 3) for item in direct):
        reasons.append('No valid source span directly supporting the requested answer.')
    elif proposed_answer:
        words = lambda text: set(re.findall(r'[^\W\d_]+', text.casefold()))
        needed = {word for word in words(proposed_answer) if len(word) > 1
                  and word not in {'the', 'and', 'or', 'of', 'an'}}
        quoted = words(' '.join(item['quote'] for item in direct))
        if not needed <= quoted:
            reasons.append('The proposed entity is absent from its direct supporting quotes.')
    checks = audit.get('checks')
    minimum = 2 if len(question) > 400 else 1
    if not isinstance(checks, list) or len(checks) < minimum:
        reasons.append('Too few distinguishing conditions were checked.')
    else:
        claims = {check.get('claim', '').strip().casefold() for check in checks
                  if isinstance(check, dict) and isinstance(check.get('claim'), str)}
        if len(claims) < minimum:
            reasons.append('Verification repeated the same condition instead of checking distinct conditions.')
        for check in checks:
            if (not isinstance(check, dict) or check.get('status') != 'supported'
                    or not isinstance(check.get('claim'), str) or len(check['claim'].strip()) < 8
                    or not valid_span(check)):
                reasons.append('A required condition is missing, contradicted, or lacks an exact quote.')
                break
    if len(evidence_ids) > 5:
        reasons.append('Supporting spans require more than five evidence excerpts.')
    return not reasons, reasons, evidence_ids if not reasons else []


def grounded_parts(audit, snippets):
    audit = materialize_audit(audit, snippets)
    by_id = {s['id']: s['text'] for s in snippets}

    def valid(item, minimum):
        if not isinstance(item, dict) or not isinstance(item.get('quote'), str):
            return False
        quote = ' '.join(item['quote'].split())
        return (isinstance(item.get('snippet_id'), str) and item['snippet_id'] in by_id
                and minimum <= len(quote) <= 1200 and source_span(quote, by_id[item['snippet_id']]) is not None)

    direct = audit.get('answer_support', [])
    checks = audit.get('checks', [])
    return ([s for s in direct if valid(s, 3)] if isinstance(direct, list) else [],
            [c for c in checks if isinstance(c, dict) and c.get('status') == 'supported' and valid(c, 12)]
            if isinstance(checks, list) else [])


def can_adjudicate(audit, snippets, answer, question):
    """Retry only grounded partial matches, never unsupported names or contradictions."""
    if audit.get('type_matches') is not True:
        return False
    checks = audit.get('checks')
    if not isinstance(checks, list) or any(isinstance(c, dict) and c.get('status') == 'contradicted' for c in checks):
        return False
    direct, supported = grounded_parts(audit, snippets)
    supported = adjudication_checks(direct, supported)
    claims = {c.get('claim', '').strip().casefold() for c in supported if isinstance(c.get('claim'), str)}
    if len(claims) < 2:
        return False
    probe = {**audit, 'supported': True, 'missing_clues': [], 'checks': supported, 'answer_support': direct}
    # This checks the spans, entity binding and distinctness, not the verdict.
    return validate_audit(probe, snippets, question, answer)[0]


def adjudication_checks(direct, supported):
    checks = list(supported)
    quotes = {' '.join(c['quote'].split()) for c in checks}
    for span in direct:
        quote = ' '.join(span['quote'].split())
        if len(quote) >= 12 and quote not in quotes:
            checks.append({**span, 'status': 'supported',
                           'claim': f'Direct source support in excerpt {span["snippet_id"]}'})
            quotes.add(quote)
    return checks


def audit_candidate(question, answer, snippets, model, tools, generator):
    """Log a real verification step; return measured usage and any API failure."""
    started = time.monotonic()
    type_error = answer_type_error(question, answer)
    input_tokens = output_tokens = 0
    error = None
    ids = []
    audit = None
    if type_error:
        output = {'supported': False, 'reasons': [type_error], 'model_called': False}
    else:
        selected = focus_snippets(question, answer, snippets)
        tools.record_retrieval(
            stage='verification_context_selection',
            arguments={'candidate': answer, 'input_snippets': len(snippets), 'max_snippets': 14},
            output={'snippet_ids': [item['id'] for item in selected]},
            doc_ids=list(dict.fromkeys(item['doc_id'] for item in selected)),
            elapsed_ms=(time.monotonic() - started) * 1000,
        )
        snippets = selected
        input_tokens = output_tokens = None
        try:
            audit, input_tokens, output_tokens = generator(question, snippets, model, proposed_answer=answer, answer_policy='verified')
            supported, reasons, ids = validate_audit(audit, snippets, question, answer)
            output = {'supported': supported, 'reasons': reasons, 'audit': materialize_audit(audit, snippets, answer),
                      'raw_audit': audit, 'model_called': True}
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            error = exc
            if isinstance(exc, ModelResponseError):
                input_tokens, output_tokens = exc.input_tokens, exc.output_tokens
            output = {'supported': False, 'error': f'{type(exc).__name__}: {exc}', 'model_called': True}
    tools.record_retrieval(
        stage='answer_verification',
        arguments={'model': model, 'proposed_answer': answer, 'policy': POLICY,
                   'thinking_enabled': (os.environ.get('IR2026_API_BASE', '').startswith('https://api.siliconflow.')
                                        and os.environ.get('IR2026_VERIFY_THINKING', '0') == '1'),
                   'snippet_ids': [item['id'] for item in snippets]},
        output=output, doc_ids=list(dict.fromkeys(item['doc_id'] for item in snippets if item['id'] in ids)),
        elapsed_ms=(time.monotonic() - started) * 1000,
    )
    if audit is not None and not output['supported'] and error is None and can_adjudicate(audit, snippets, answer, question):
        started = time.monotonic()
        try:
            direct, grounded = grounded_parts(audit, snippets)
            hint = {'grounded_checks': adjudication_checks(direct, grounded)}
            second, inputs, outputs = generator(question, snippets, model, proposed_answer=answer,
                                                 verification_hint=hint, answer_policy='verified')
            input_tokens = input_tokens + inputs if input_tokens is not None and inputs is not None else None
            output_tokens = output_tokens + outputs if output_tokens is not None and outputs is not None else None
            supported, reasons, ids = validate_audit(second, snippets, question, answer)
            output = {'supported': supported, 'reasons': reasons, 'audit': materialize_audit(second, snippets, answer),
                      'raw_audit': second, 'model_called': True}
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            error = exc
            if isinstance(exc, ModelResponseError):
                input_tokens = input_tokens + exc.input_tokens if input_tokens is not None and exc.input_tokens is not None else None
                output_tokens = output_tokens + exc.output_tokens if output_tokens is not None and exc.output_tokens is not None else None
            else:
                input_tokens = output_tokens = None
            ids = []
            output = {'supported': False, 'error': f'{type(exc).__name__}: {exc}', 'model_called': True}
        tools.record_retrieval(
            stage='answer_verification_adjudication',
            arguments={'model': model, 'proposed_answer': answer, 'policy': POLICY,
                       'snippet_ids': [item['id'] for item in snippets]},
            output=output, doc_ids=list(dict.fromkeys(s['doc_id'] for s in snippets if s['id'] in ids)),
            elapsed_ms=(time.monotonic() - started) * 1000,
        )
    return output['supported'], ids, input_tokens, output_tokens, error
