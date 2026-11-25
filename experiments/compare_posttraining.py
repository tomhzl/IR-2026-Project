#!/usr/bin/env python3
"""Compare fixed-pipeline dev runs before and after post-training."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from ir2026 import normalize_answer, read_jsonl


def load_run(path):
    with (path / 'per_query.csv').open(newline='') as handle:
        per_query = {
            row['id']: row for row in csv.DictReader(handle)
        }
    results = {
        row['id']: row for row in read_jsonl(path / 'results.jsonl')
    }
    events = Counter()
    generation_errors = Counter()
    for trace in (path / 'traces').glob('*.jsonl'):
        for event in read_jsonl(trace):
            tool = event['tool']
            events[tool] += 1
            output = event.get('output') or {}
            if (tool in {'answer_generation', 'answer_candidate_fallback'}
                    and 'error' in output):
                generation_errors[f'{tool}: {output["error"]}'] += 1
    return {
        'metrics': json.loads((path / 'metrics.json').read_text()),
        'summary': json.loads((path / 'run_summary.json').read_text()),
        'per_query': per_query,
        'results': results,
        'events': events,
        'generation_errors': generation_errors,
    }


def public_run(data):
    return {
        'metrics': data['metrics'],
        'summary': data['summary'],
        'event_counts': dict(sorted(data['events'].items())),
        'generation_errors': dict(sorted(
            data['generation_errors'].items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--proposed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    proposed = load_run(args.proposed)
    if set(baseline['results']) != set(proposed['results']):
        raise ValueError('Runs have different query IDs')
    if baseline['metrics']['split'] != 'dev':
        raise ValueError('Comparison is reserved for dev runs')
    if proposed['metrics']['split'] != 'dev':
        raise ValueError('Comparison is reserved for dev runs')

    ids = set(baseline['results'])
    baseline_correct = {
        qid for qid in ids
        if int(baseline['per_query'][qid]['answer_em']) == 1
    }
    proposed_correct = {
        qid for qid in ids
        if int(proposed['per_query'][qid]['answer_em']) == 1
    }
    transitions = Counter(
        f"{baseline['results'][qid]['status']}"
        f"->{proposed['results'][qid]['status']}"
        for qid in ids
    )
    changed_answers = sum(
        normalize_answer(baseline['results'][qid].get('answer', ''))
        != normalize_answer(proposed['results'][qid].get('answer', ''))
        for qid in ids
    )

    metric_keys = [
        'gold_hit_at_1', 'gold_hit_at_5', 'gold_hit_at_10',
        'evidence_recall_at_10', 'answer_em', 'completion_rate',
        'mean_tool_calls', 'mean_trace_events', 'mean_elapsed_seconds',
    ]
    summary_keys = [
        'model_input_tokens', 'model_output_tokens',
        'elapsed_seconds_total', 'elapsed_seconds_mean',
        'elapsed_seconds_max',
    ]
    comparison = {
        'split': 'dev',
        'baseline_dir': str(args.baseline),
        'proposed_dir': str(args.proposed),
        'baseline': public_run(baseline),
        'proposed': public_run(proposed),
        'metric_deltas': {
            key: proposed['metrics'][key] - baseline['metrics'][key]
            for key in metric_keys
        },
        'summary_deltas': {
            key: proposed['summary'][key] - baseline['summary'][key]
            for key in summary_keys
        },
        'answer_em_gained_ids': sorted(proposed_correct - baseline_correct),
        'answer_em_lost_ids': sorted(baseline_correct - proposed_correct),
        'answer_em_retained_ids': sorted(
            baseline_correct & proposed_correct),
        'status_transitions': dict(sorted(transitions.items())),
        'normalized_answer_changed_queries': changed_answers,
        'answer_api_calls': 0,
        'note': ('Fifty-query dev comparison; no significance claim. '
                 'No dev/test data was used for LoRA training.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
