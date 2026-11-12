"""The original v1 post-answer check, separate from experimental strict checks."""

import time

from agent.answer import ModelResponseError
from agent.prompts import V1_VERIFICATION_POLICY


def verify_v1_candidate(question, answer, snippets, model, tools, generator):
    started = time.monotonic()
    inputs = outputs = None
    error = None
    try:
        audit, inputs, outputs = generator(question, snippets, model,
                                            proposed_answer=answer, answer_policy='v1')
        supported = audit.get('supported') is True
        result = {'supported': supported, 'missing_clues': audit.get('missing_clues', [])}
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        error = exc
        if isinstance(exc, ModelResponseError):
            inputs, outputs = exc.input_tokens, exc.output_tokens
        supported = False
        result = {'supported': False, 'error': f'{type(exc).__name__}: {exc}'}
    tools.record_retrieval(
        stage='answer_verification',
        arguments={'model': model, 'policy': V1_VERIFICATION_POLICY, 'proposed_answer': answer,
                   'snippet_ids': [s['id'] for s in snippets]},
        output=result, doc_ids=list(dict.fromkeys(s['doc_id'] for s in snippets)),
        elapsed_ms=(time.monotonic() - started) * 1000,
    )
    return supported, inputs, outputs, error
