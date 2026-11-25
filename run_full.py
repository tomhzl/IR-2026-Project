#!/usr/bin/env python3
"""Run the selected full workflow, summary, submission checks, and evaluation."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_step(command):
    printable = ' '.join(str(item) for item in command)
    print(f'\n+ {printable}', flush=True)
    return subprocess.run([str(item) for item in command], cwd=ROOT).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    output = args.output or ROOT / 'output' / f'qwen_hybrid_v1_{args.split}'
    python = sys.executable
    retrieve = [python, ROOT / 'run_qwen_hybrid.py', '--split', args.split, '--output', output]
    if args.resume:
        retrieve.append('--resume')
    steps = [
        retrieve,
        [python, ROOT / 'summarize_run.py', '--run-dir', output],
        [python, ROOT / 'check_results.py', '--split', args.split, '--run-dir', output, '--submission'],
    ]
    if args.split != 'test':
        steps.append([python, ROOT / 'evaluate.py', '--split', args.split, '--run-dir', output])
    for step in steps:
        code = run_step(step)
        if code:
            print(f'Step failed with exit code {code}: {step[1]}', file=sys.stderr)
            return code
    if args.split == 'test':
        print('Test labels are teacher-only; submission checking completed without local evaluation.')
    print(f'Run artifacts: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
