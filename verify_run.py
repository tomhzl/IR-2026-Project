#!/usr/bin/env python3
"""Audit an existing run's fixed answers without changing its retrieval rankings.

Source traces and their measured costs are retained; actual new audit events and
costs are appended. The derived run explicitly records its source and scope.
No labels are read by this program.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import time

from agent.answer import ModelAPIError, SILICONFLOW_API_BASE, generate_answer, load_api_environment
from agent.evidence import select_no_answer_evidence
from agent.verification import POLICY, audit_candidate, validate_audit, answer_type_error, source_span, materialize_audit
from ir2026 import Corpus, CorpusTools, load_policy, read_jsonl, validate_results, write_jsonl


ROOT = Path(__file__).resolve().parent


def align_existing(args, parser):
    """Recheck already measured model verdicts; add no model calls or token costs."""
    if args.candidate_run is None:
        parser.error('--align-source-spans requires --candidate-run')
    info = json.loads((args.run_dir / 'run_info.json').read_text())
    if info.get('execution_status') != 'complete':
        parser.error('The audit run must be complete before span alignment')
    questions = read_jsonl(args.run_dir / 'questions.jsonl')
    if questions != read_jsonl(args.candidate_run / 'questions.jsonl'):
        parser.error('Candidate and audit questions differ')
    source = read_jsonl(args.run_dir / 'results.jsonl')
    candidates = {r['id']: r for r in read_jsonl(args.candidate_run / 'results.jsonl')}
    corpus = Corpus(ROOT / 'corpus')
    policy = load_policy(ROOT / 'configs/default.json')
    errors = validate_results(questions, source, corpus, policy['limits'],
                              trace_dir=args.run_dir / 'traces', method_type='retrieval')
    if errors:
        parser.error('Invalid audit run: ' + '; '.join(errors[:3]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'traces').mkdir()
    write_jsonl(args.output / 'questions.jsonl', questions)
    qmap = {q['id']: q for q in questions}
    info.update(method=info['method'] + '-span-aligned', derived_from=str(args.run_dir.resolve()),
                candidate_run=str(args.candidate_run.resolve()), verification_policy=POLICY,
                verification_scope='fixed candidates and fixed model audits; deterministic source-span alignment',
                execution_status='running')
    (args.output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
    results = []
    for previous in source:
        started = time.monotonic()
        row = copy.deepcopy(previous)
        candidate = candidates[row['id']]
        if candidate['doc_ids'] != row['doc_ids']:
            parser.error('Retrieval rankings changed between candidate and audit runs')
        tools = CorpusTools(corpus, row['id'], policy['limits'], method_type='retrieval')
        tools.started = started - previous['usage']['elapsed_seconds']
        tools.trace = read_jsonl(args.run_dir / 'traces' / f'{row["id"]}.jsonl')
        if candidate['answer']:
            snippets = next(e['output']['snippets'] for e in tools.trace if e['tool']=='read_evidence_candidates')
            audits = [e for e in tools.trace if e['tool'] in {'answer_verification', 'answer_verification_adjudication'}]
            audit = audits[-1]['output'].get('audit') if audits else None
            type_error = answer_type_error(qmap[row['id']]['question'], candidate['answer'])
            supported, reasons, ids = (False, [type_error or 'No complete model audit.'], [])
            if not type_error and isinstance(audit, dict):
                audit = materialize_audit(audit, snippets, candidate['answer'])
                supported, reasons, ids = validate_audit(audit, snippets, qmap[row['id']]['question'], candidate['answer'])
            grounded = []
            if supported:
                by_id = {s['id']: s for s in snippets}
                row.update(answer=candidate['answer'], status='ok',
                           evidence=[{'doc_id': by_id[i]['doc_id'], 'text': by_id[i]['text']} for i in ids])
                for item in audit['answer_support'] + audit['checks']:
                    grounded.append({'snippet_id': item['snippet_id'],
                                     'quote': source_span(item['quote'], by_id[item['snippet_id']]['text'])})
            else:
                row['answer'] = ''
                if row['status'] != 'error':
                    row['status'] = 'no_answer'
            if audit is None and audits and 'error' in audits[-1]['output']:
                row.update(status='error', error=audits[-1]['output']['error'])
            row.update(candidate_answer=candidate['answer'], verification={
                'policy': POLICY, 'status': 'error' if row['status']=='error' else ('verified' if supported else 'unsupported'),
                'reasons': reasons})
            tools.record_retrieval(
                stage='verification_source_span_alignment',
                arguments={'policy': POLICY, 'proposed_answer': candidate['answer'], 'model_called': False},
                output={'supported': supported, 'reasons': reasons, 'source_spans': grounded},
                doc_ids=list(dict.fromkeys(s['doc_id'] for s in snippets if s['id'] in ids)),
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
            row['usage'].update(trace_events=len(tools.trace),
                                elapsed_seconds=round(previous['usage']['elapsed_seconds'] + time.monotonic() - started, 4),
                                token_notes='Cumulative source generation and audit usage; final source-span alignment called no model.')
        write_jsonl(args.output / 'traces' / f'{row["id"]}.jsonl', tools.trace)
        results.append(row)
    write_jsonl(args.output / 'results.jsonl', results)
    info['execution_status'] = 'complete'
    (args.output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'Aligned answers: {sum(bool(r["answer"]) for r in results)}/{len(results)}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-interval', type=float, default=8, help='Seconds between audit requests')
    parser.add_argument('--align-source-spans', action='store_true', help='Recheck existing model audit verdicts without API calls')
    parser.add_argument('--candidate-run', type=Path, help='Original candidate run for source-span alignment')
    args = parser.parse_args()
    if args.min_interval < 0 or args.min_interval > 60:
        parser.error('min-interval must be between 0 and 60')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory')
    if args.align_source_spans:
        return align_existing(args, parser)
    load_api_environment()
    info = json.loads((args.run_dir / 'run_info.json').read_text())
    questions = read_jsonl(args.run_dir / 'questions.jsonl')
    source = read_jsonl(args.run_dir / 'results.jsonl')
    model = info['model'].split(';')[-1]
    if info.get('api_provider') == 'siliconflow':
        os.environ['IR2026_API_BASE'] = SILICONFLOW_API_BASE
        if not (os.environ.get('SILICONFLOW_API_KEY') or os.environ.get('IR2026_API_KEY')):
            parser.error('Set SILICONFLOW_API_KEY')
    corpus = Corpus(ROOT / 'corpus')
    config = load_policy(ROOT / 'configs/default.json')
    errors = validate_results(questions, source, corpus, config['limits'],
                              trace_dir=args.run_dir / 'traces', method_type='retrieval')
    if errors:
        parser.error('Invalid source run: ' + '; '.join(errors[:3]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'traces').mkdir()
    write_jsonl(args.output / 'questions.jsonl', questions)
    info.update(method=info['method'] + '-verified-' + POLICY, derived_from=str(args.run_dir.resolve()),
                verification_policy=POLICY, verification_scope='fixed source-run candidates',
                verification_thinking=(info.get('api_provider') == 'siliconflow'
                                       and os.environ.get('IR2026_VERIFY_THINKING', '0') == '1'),
                execution_status='running')
    (args.output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
    qmap = {q['id']: q for q in questions}
    results = []
    next_request = 0
    for position, previous in enumerate(source, 1):
        started = time.monotonic()
        row = copy.deepcopy(previous)
        qid = row['id']
        tools = CorpusTools(corpus, qid, config['limits'], trace_path=args.output / 'traces' / f'{qid}.jsonl',
                            method_type='retrieval')
        tools.trace = read_jsonl(args.run_dir / 'traces' / f'{qid}.jsonl')
        tools.started = started - previous['usage']['elapsed_seconds']
        if previous['answer'] and previous['status'] == 'ok':
            snippets = next(e['output']['snippets'] for e in tools.trace
                            if e['tool'] == 'read_evidence_candidates')
            delay = next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_request = time.monotonic() + args.min_interval
            supported, ids, inputs, outputs, error = audit_candidate(
                qmap[qid]['question'], previous['answer'], snippets, model, tools, generate_answer)
            verdict = tools.trace[-1]['output']
            if supported:
                by_id = {s['id']: s for s in snippets}
                row['evidence'] = [{'doc_id': by_id[i]['doc_id'], 'text': by_id[i]['text']} for i in ids]
            else:
                row.update(answer='', status='error' if error is not None else 'no_answer')
                selected = select_no_answer_evidence(snippets, row['doc_ids'], qmap[qid]['question'])
                row['evidence'] = [{'doc_id': s['doc_id'], 'text': s['text']} for s in selected]
                tools.record_retrieval('select_rejected_answer_evidence', {'max_quotes': 3},
                                       {'snippet_ids': [s['id'] for s in selected]},
                                       [s['doc_id'] for s in selected], 0)
            if isinstance(error, ModelAPIError):
                row.update(error=str(error), api_status_code=error.status_code)
            elif error is not None:
                row['error'] = str(error)
            row.update(candidate_answer=previous['answer'], verification={
                'policy': POLICY, 'status': 'error' if error else ('verified' if supported else 'unsupported'),
                'reasons': verdict.get('reasons', [str(error)] if error else [])})
            for key, extra in [('model_input_tokens', inputs), ('model_output_tokens', outputs)]:
                old = row['usage'].get(key)
                row['usage'][key] = old + extra if old is not None and extra is not None else None
            row['usage'].update(trace_events=len(tools.trace),
                                elapsed_seconds=round(previous['usage']['elapsed_seconds'] + time.monotonic() - started, 4),
                                token_notes='Cumulative measured usage: source-run generation plus source-grounded answer audit.')
        write_jsonl(args.output / 'traces' / f'{qid}.jsonl', tools.trace)
        results.append(row)
        write_jsonl(args.output / 'results.jsonl', results)
        print(f'{position}/{len(source)} id={qid} status={row["status"]}', flush=True)
        if row.get('api_status_code') in {401, 402, 403}:
            info.update(execution_status='api_error', stop_reason=row['error'])
            (args.output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
            return 2
    info['execution_status'] = 'complete'
    (args.output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
