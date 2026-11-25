#!/usr/bin/env python3
"""Run the selected Qwen + hybrid retrieval v1 preset."""

import argparse
import json
from pathlib import Path

from run import main as run_main


ROOT = Path(__file__).resolve().parent
PRESET_PATH = ROOT / 'configs' / 'qwen_hybrid_v1.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
    parser.add_argument('--preset', type=Path, default=PRESET_PATH,
                        help='Qwen hybrid preset; defaults to the frozen SiliconFlow v1 configuration')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--show-config', action='store_true', help='Show the frozen preset without calling the API')
    args = parser.parse_args(argv)
    preset_path = args.preset
    preset = json.loads(preset_path.read_text(encoding='utf-8'))
    if args.show_config:
        print(json.dumps(preset, ensure_ascii=False, indent=2))
        return 0
    output = args.output or ROOT / 'output' / f'{preset_path.stem}_{args.split}'
    command = ['--split', args.split, '--output', str(output), '--hybrid-config', str(preset_path)]
    for name in ['agent', 'method_type', 'method_label', 'model_label', 'api_provider', 'retriever', 'answer_policy']:
        command.extend(['--' + name.replace('_', '-'), preset[name]])
    if args.limit is not None:
        command.extend(['--limit', str(args.limit)])
    if args.resume:
        command.append('--resume')
    return run_main(command)


if __name__ == '__main__':
    raise SystemExit(main())
