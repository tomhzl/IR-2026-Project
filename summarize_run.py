#!/usr/bin/env python3
"""Aggregate a completed run without changing the official submission files."""

import argparse
import json
import math
from pathlib import Path

from ir2026 import STATUSES, read_jsonl


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def token_total(values):
    if not values or any(type(value) is not int or value < 0 for value in values):
        return None
    return sum(values)


def summarize(results, info):
    usage = [row.get('usage') if isinstance(row.get('usage'), dict) else {} for row in results]
    status_counts = {status: 0 for status in sorted(STATUSES)}
    for row in results:
        status = row.get('status')
        if status in status_counts:
            status_counts[status] += 1

    elapsed = [item.get('elapsed_seconds') for item in usage]
    elapsed_complete = bool(results) and all(number(value) and value >= 0 for value in elapsed)
    total_elapsed = round(sum(elapsed), 4) if elapsed_complete else None
    input_values = [item.get('model_input_tokens') for item in usage]
    output_values = [item.get('model_output_tokens') for item in usage]
    complete_tokens = sum(
        type(item.get('model_input_tokens')) is int and item.get('model_input_tokens') >= 0
        and type(item.get('model_output_tokens')) is int and item.get('model_output_tokens') >= 0
        for item in usage
    )
    incomplete_tokens = len(results) - complete_tokens
    return {
        'release': info.get('release'),
        'protocol_version': info.get('protocol_version'),
        'method': info.get('method'),
        'split': info.get('split'),
        'execution_status': info.get('execution_status'),
        'queries': len(results),
        'expected_queries': info.get('queries'),
        'status_counts': status_counts,
        'completion_rate': status_counts['ok'] / len(results) if results else 0,
        'elapsed_seconds_total': total_elapsed,
        'elapsed_seconds_mean': round(total_elapsed / len(results), 6) if elapsed_complete else None,
        'elapsed_seconds_max': max(elapsed) if elapsed_complete else None,
        'model_input_tokens': token_total(input_values),
        'model_output_tokens': token_total(output_values),
        'token_complete_queries': complete_tokens,
        'token_incomplete_queries': incomplete_tokens,
        'incomplete_tool_usage_queries': sum(item.get('tool_calls_complete') is False for item in usage),
        'token_note': ('Token totals are null because at least one query has incomplete usage.'
                       if incomplete_tokens else 'Token totals include all queries.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    info = json.loads((args.run_dir / 'run_info.json').read_text(encoding='utf-8'))
    results = read_jsonl(args.run_dir / 'results.jsonl')
    summary = summarize(results, info)
    output = args.run_dir / 'run_summary.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                      encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
