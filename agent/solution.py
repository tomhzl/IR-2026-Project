"""BM25 / hybrid retrieval followed by local evidence selection and answering."""

import hashlib
from pathlib import Path
import sqlite3
import time

from agent.answer import ModelAPIError, ModelResponseError, extract_percentage_answer, generate_answer
from agent.prompts import requires_v1_verification
# Keep older helper imports usable while their implementations live in focused modules.
from agent.retrieval import STOP_WORDS, query_terms, queries_for, retrieve_bm25
from agent.evidence import anchored_windows, collect_snippets, excerpt_from, select_no_answer_evidence


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / 'index' / 'passages.sqlite3'


def solve(question, tools, config):
    answer_policy = config.get('answer_policy', 'v1')
    if answer_policy not in {'v1', 'verified'}:
        raise ValueError('answer_policy must be v1 or verified')
    if not INDEX_PATH.is_file():
        raise RuntimeError('Missing FTS5 index; run python3 build_index.py first')
    expected = hashlib.sha256((ROOT / 'corpus' / 'documents.jsonl').read_bytes()).hexdigest()
    db = sqlite3.connect(f'file:{INDEX_PATH}?mode=ro', uri=True)
    try:
        actual = db.execute("SELECT value FROM metadata WHERE key='manifest_sha256'").fetchone()
        if actual is None or actual[0] != expected:
            raise RuntimeError('FTS5 index does not match corpus/documents.jsonl; rebuild it')

        ranked, best_passage = retrieve_bm25(db, question['question'], tools)
        started = time.monotonic()
        doc_ids = [doc_id for doc_id, _ in ranked[:10]]
        tools.record_retrieval(
            stage='rrf_fusion',
            arguments={'candidate_docs': len(ranked), 'top_k': 10, 'rrf_constant': 30},
            output={'hits': [{'doc_id': doc_id, 'score': score} for doc_id, score in ranked[:10]]},
            doc_ids=doc_ids,
            elapsed_ms=(time.monotonic() - started) * 1000,
        )
        is_hybrid = config.get('retriever', 'bm25') == 'hybrid'
        uses_lsa = False
        if is_hybrid:
            from agent.hybrid import INDEX_PATH as HYBRID_PATH, load_config, load_index, fuse_rankings
            started = time.monotonic()
            settings = config.get('hybrid') or load_config()
            uses_lsa = settings['weights']['lsa'] > 0
            index = load_index(str(HYBRID_PATH), expected, HYBRID_PATH.stat().st_mtime_ns, uses_lsa)
            lanes, features = index.search(question['question'], settings['candidate_limit'])
            tools.record_retrieval(
                stage='tfidf_lsa_search',
                arguments={'model': index.metadata['model'] if uses_lsa else None, 'query': question['question'],
                           'query_features': features, 'limit': settings['candidate_limit']},
                output={name: [{'doc_id': d, 'score': s} for d, s in hits]
                        for name, hits in lanes.items()},
                doc_ids=list(dict.fromkeys(d for hits in lanes.values() for d, _ in hits)),
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
            started = time.monotonic()
            lanes['bm25'] = ranked[:settings['candidate_limit']]
            ranked = fuse_rankings(lanes, settings['weights'], settings['rrf_constant'])
            doc_ids = [doc_id for doc_id, _ in ranked]
            tools.record_retrieval(
                stage='hybrid_rrf', arguments=settings,
                output={'hits': [{'doc_id': d, 'score': s} for d, s in ranked]},
                doc_ids=doc_ids, elapsed_ms=(time.monotonic() - started) * 1000,
            )
            for doc_id in doc_ids:
                if doc_id not in best_passage:
                    best_passage[doc_id] = (0, tools.corpus.text(doc_id)[:1800])
        if config.get('retrieval_only'):
            started = time.monotonic()
            evidence = []
            if doc_ids:
                doc_id = doc_ids[0]
                start, body = best_passage[doc_id]
                source = tools.corpus.text(doc_id)
                _, text = excerpt_from(body, start, query_terms(question['question']), source)
                if len(text) >= 20:
                    evidence = [{'doc_id': doc_id, 'text': text}]
            tools.record_retrieval(
                stage='read_retrieved_excerpt', arguments={'max_chars': 1200},
                output={'evidence': evidence}, doc_ids=[e['doc_id'] for e in evidence],
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
            return {'answer': '', 'doc_ids': doc_ids, 'evidence': evidence, 'status': 'no_answer',
                    'usage': {'model_input_tokens': None if uses_lsa else 0,
                              'model_output_tokens': None if uses_lsa else 0,
                              'token_notes': ('Local TF-IDF/LSA uses vocabulary features; language-model token counts '
                                              'are not applicable. No answer-generation API called.' if uses_lsa
                                              else 'Sparse retrieval only; no model used.')}}
        snippets = collect_snippets(db, doc_ids, best_passage, question['question'], tools)
        by_id = {item['id']: item for item in snippets}
        evidence = ([{'doc_id': snippets[0]['doc_id'], 'text': snippets[0]['text']}]
                    if snippets else [])
        answer = ''
        api_error = None
        candidate_answer = ''
        verification_info = None
        model_answer = False
        input_tokens = output_tokens = 0
        token_notes = 'No model used.'
        if snippets:
            started = time.monotonic()
            answer, chosen_ids = extract_percentage_answer(question['question'], snippets)
            if answer:
                evidence = [
                    {'doc_id': by_id[item]['doc_id'], 'text': by_id[item]['text']}
                    for item in chosen_ids
                ]
                token_notes = 'Deterministic extraction; no model call for this question.'
            tools.record_retrieval(
                stage='extractive_answer',
                arguments={'rule': 'explicit_percentage'},
                output={'answer': answer, 'evidence_ids': chosen_ids},
                doc_ids=list(dict.fromkeys(item['doc_id'] for item in evidence)),
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        if config.get('model') and snippets and not answer:
            started = time.monotonic()
            input_tokens = output_tokens = None
            token_notes = 'Model call attempted; usage was unavailable.'
            primary_completed = False
            try:
                response, input_tokens, output_tokens = generate_answer(
                    question['question'], snippets, config['model'], answer_policy=answer_policy)
                primary_completed = True
                candidate = response.get('answer')
                chosen = response.get('evidence_ids')
                if (isinstance(candidate, str) and 0 < len(candidate.strip()) <= 300
                        and isinstance(chosen, list) and chosen
                        and all(isinstance(item, str) and item in by_id for item in chosen)):
                    answer = candidate.strip()
                    model_answer = True
                    evidence = [
                        {'doc_id': by_id[item]['doc_id'], 'text': by_id[item]['text']}
                        for item in dict.fromkeys(chosen)
                    ][:5]
                token_notes = 'API-reported prompt and completion tokens.' if (
                    input_tokens is not None and output_tokens is not None
                ) else 'API did not provide complete token usage.'
                output = {'answer': answer, 'evidence_ids': chosen if isinstance(chosen, list) else []}
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                if isinstance(exc, ModelAPIError):
                    api_error = exc
                if isinstance(exc, ModelResponseError):
                    input_tokens, output_tokens = exc.input_tokens, exc.output_tokens
                    if input_tokens is not None and output_tokens is not None:
                        token_notes = 'API-reported tokens for incomplete or invalid model response.'
                output = {'error': f'{type(exc).__name__}: {exc}'}
            tools.record_retrieval(
                stage='answer_generation',
                arguments={'model': config['model'], 'snippet_ids': list(by_id)},
                output=output,
                doc_ids=list(dict.fromkeys(item['doc_id'] for item in evidence)),
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
            if primary_completed and not answer:
                started = time.monotonic()
                focused = []
                counts = {}
                for item in snippets:
                    count = counts.get(item['doc_id'], 0)
                    if count < 2:
                        focused.append(item)
                        counts[item['doc_id']] = count + 1
                focused_ids = {item['id'] for item in focused}
                try:
                    fallback, extra_input, extra_output = generate_answer(
                        question['question'], focused, config['model'], relaxed=True, answer_policy=answer_policy)
                    input_tokens = (input_tokens + extra_input
                                    if input_tokens is not None and extra_input is not None else None)
                    output_tokens = (output_tokens + extra_output
                                     if output_tokens is not None and extra_output is not None else None)
                    candidate = fallback.get('answer')
                    chosen = fallback.get('evidence_ids')
                    if (isinstance(candidate, str) and 0 < len(candidate.strip()) <= 300
                            and isinstance(chosen, list) and chosen
                            and all(isinstance(item, str) and item in focused_ids for item in chosen)
                            and len({by_id[item]['doc_id'] for item in chosen}) >= 2
                            and any(candidate.strip().casefold() in by_id[item]['text'].casefold()
                                    for item in chosen)):
                        answer = candidate.strip()
                        model_answer = True
                        evidence = [
                            {'doc_id': by_id[item]['doc_id'], 'text': by_id[item]['text']}
                            for item in dict.fromkeys(chosen)
                        ][:5]
                    fallback_output = {'answer': answer,
                                       'evidence_ids': chosen if isinstance(chosen, list) else []}
                except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                    if isinstance(exc, ModelAPIError):
                        api_error = exc
                    if isinstance(exc, ModelResponseError):
                        input_tokens = (input_tokens + exc.input_tokens
                                        if input_tokens is not None and exc.input_tokens is not None else None)
                        output_tokens = (output_tokens + exc.output_tokens
                                         if output_tokens is not None and exc.output_tokens is not None else None)
                    else:
                        input_tokens = output_tokens = None
                    fallback_output = {'error': f'{type(exc).__name__}: {exc}'}
                token_notes = ('API-reported tokens for generation and focused fallback.'
                               if input_tokens is not None and output_tokens is not None
                               else 'Some model token usage was unavailable.')
                tools.record_retrieval(
                    stage='answer_candidate_fallback',
                    arguments={'model': config['model'],
                               'snippet_ids': [item['id'] for item in focused]},
                    output=fallback_output,
                    doc_ids=list(dict.fromkeys(item['doc_id'] for item in evidence)),
                    elapsed_ms=(time.monotonic() - started) * 1000,
                )
        if (config.get('model') and answer and not api_error and answer_policy == 'v1'
                and model_answer and requires_v1_verification(question['question'])):
            from agent.answer_policy import verify_v1_candidate
            supported, inputs, outputs, error = verify_v1_candidate(
                question['question'], answer, snippets, config['model'], tools, generate_answer)
            input_tokens = input_tokens + inputs if input_tokens is not None and inputs is not None else None
            output_tokens = output_tokens + outputs if output_tokens is not None and outputs is not None else None
            if isinstance(error, ModelAPIError):
                api_error = error
            if not supported:
                answer = ''
            token_notes = ('API-reported tokens for generation and v1 numbered-clue verification.'
                           if input_tokens is not None and output_tokens is not None
                           else 'Some model token usage was unavailable.')
        if config.get('model') and answer and not api_error and answer_policy == 'verified':
            from agent.verification import POLICY, audit_candidate
            candidate_answer = answer
            supported, verified_ids, audit_input, audit_output, audit_error = audit_candidate(
                question['question'], answer, snippets, config['model'], tools, generate_answer)
            input_tokens = (input_tokens + audit_input
                            if input_tokens is not None and audit_input is not None else None)
            output_tokens = (output_tokens + audit_output
                             if output_tokens is not None and audit_output is not None else None)
            if isinstance(audit_error, ModelAPIError):
                api_error = audit_error
            verdict = tools.trace[-1]['output']
            verification_info = {'policy': POLICY,
                                 'status': 'error' if audit_error else ('verified' if supported else 'unsupported'),
                                 'reasons': verdict.get('reasons', [verdict['error']] if 'error' in verdict else [])}
            if supported:
                evidence = [{'doc_id': by_id[item]['doc_id'], 'text': by_id[item]['text']}
                            for item in verified_ids]
            else:
                answer = ''
            token_notes = ('API-reported tokens for generation and source-grounded verification.'
                           if input_tokens is not None and output_tokens is not None
                           else 'Some model token usage was unavailable.')

        if api_error:
            answer = ''
        if not answer and snippets:
            started = time.monotonic()
            selected = select_no_answer_evidence(snippets, doc_ids, question['question'])
            evidence = [{'doc_id': item['doc_id'], 'text': item['text']}
                        for item in selected]
            tools.record_retrieval(
                stage='select_no_answer_evidence',
                arguments={'max_quotes': 3},
                output={'snippet_ids': [item['id'] for item in selected]},
                doc_ids=[item['doc_id'] for item in selected],
                elapsed_ms=(time.monotonic() - started) * 1000,
            )

        if uses_lsa:
            if not config.get('model'):
                input_tokens = output_tokens = None
            token_notes += (' Local corpus-trained LSA also used; its vocabulary features do not have '
                            'language-model token counts. Any reported tokens cover only answer generation.')
        result = {
            'answer': answer,
            'doc_ids': doc_ids if evidence else [],
            'evidence': evidence,
            'status': 'error' if api_error or (verification_info and verification_info['status']=='error') else ('ok' if answer else 'no_answer'),
            'usage': {'model_input_tokens': input_tokens,
                      'model_output_tokens': output_tokens,
                      'token_notes': token_notes},
        }
        if api_error:
            result.update(error=str(api_error), api_status_code=api_error.status_code)
        if candidate_answer:
            result.update(candidate_answer=candidate_answer, verification=verification_info)
        return result
    finally:
        db.close()
