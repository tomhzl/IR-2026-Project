#!/usr/bin/env python3
"""Build answer-action SFT samples exclusively from train teacher traces."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from agent.prompts import build_messages
from ir2026 import normalize_answer, read_jsonl


def load_events(path):
    return read_jsonl(path)


def find_event(events, tool):
    for event in events:
        if event.get('tool') == tool:
            return event
    return None


def focused_snippets(snippets):
    selected = []
    counts = {}
    for item in snippets:
        doc_id = item['doc_id']
        if counts.get(doc_id, 0) < 2:
            selected.append(item)
            counts[doc_id] = counts.get(doc_id, 0) + 1
    return selected


def make_sample(question_id, question, snippets, answer, evidence_ids,
                relaxed, sample_type, source_tool):
    available = {item['id'] for item in snippets}
    evidence_ids = list(dict.fromkeys(evidence_ids))
    if not answer:
        evidence_ids = []
    if any(item not in available for item in evidence_ids):
        raise ValueError(f'{question_id}: supervision cites an unavailable snippet')

    messages = build_messages(
        question,
        snippets,
        relaxed=relaxed,
        answer_policy='v1',
    )
    messages.append({
        'role': 'assistant',
        'content': json.dumps(
            {'answer': answer, 'evidence_ids': evidence_ids},
            ensure_ascii=False,
            separators=(',', ':'),
        ),
    })
    return {
        'id': question_id,
        'split': 'train',
        'sample_type': sample_type,
        'source_tool': source_tool,
        'messages': messages,
    }


def extract_supervision(result, question, events):
    snippets_event = find_event(events, 'read_evidence_candidates')
    if snippets_event is None:
        return None, 'missing_read_evidence_candidates'
    snippets = snippets_event.get('output', {}).get('snippets')
    if not isinstance(snippets, list) or not snippets:
        return None, 'missing_snippets'

    question_id = str(result['id'])
    answer = result.get('answer', '')
    primary = find_event(events, 'answer_generation')
    fallback = find_event(events, 'answer_candidate_fallback')

    if result.get('status') == 'ok':
        choices = [
            (primary, snippets, False),
            (fallback, focused_snippets(snippets), True),
        ]
        for event, selected, relaxed in choices:
            if event is None:
                continue
            output = event.get('output') or {}
            evidence_ids = output.get('evidence_ids')
            if ('error' not in output
                    and output.get('answer') == answer
                    and answer
                    and isinstance(evidence_ids, list)
                    and all(isinstance(item, str) for item in evidence_ids)):
                try:
                    sample = make_sample(
                        question_id, question, selected, answer, evidence_ids,
                        relaxed, 'correct_answer', event['tool'],
                    )
                except ValueError:
                    continue
                return sample, 'exported'

        # A deterministic extractive answer has no teacher generation target.
        extractive = find_event(events, 'extractive_answer')
        if (extractive is not None
                and (extractive.get('output') or {}).get('answer') == answer):
            return None, 'discard_deterministic_extraction'
        return None, 'discard_no_matching_teacher_action'

    if result.get('status') == 'no_answer':
        if primary is None:
            return None, 'discard_missing_primary_action'
        output = primary.get('output') or {}
        if ('error' in output
                or output.get('answer', '') != ''
                or not isinstance(output.get('evidence_ids'), list)):
            return None, 'discard_unreliable_abstention_action'
        sample = make_sample(
            question_id, question, snippets, '', [],
            False, 'justified_abstention', 'answer_generation',
        )
        return sample, 'exported'

    return None, 'discard_non_answer_status'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--questions', type=Path,
                        default=Path('datas/train.jsonl'))
    parser.add_argument('--labels', type=Path,
                        default=Path('datas/train_labels.jsonl'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    run_info = json.loads(
        (args.run_dir / 'run_info.json').read_text(encoding='utf-8'))
    if run_info.get('split') not in (None, 'train'):
        raise ValueError('SFT teacher run must use the train split')

    results = read_jsonl(args.run_dir / 'results.jsonl')
    labels = {
        str(row['id']): row
        for row in read_jsonl(args.labels)
    }
    questions = {
        str(row['id']): row['question']
        for row in read_jsonl(args.questions)
        if str(row['id']) in labels
    }
    if len(results) != 100 or len(labels) != 100:
        raise ValueError('Expected the complete 100-question train run')
    if {str(row['id']) for row in results} != set(labels):
        raise ValueError('Teacher results and train labels have different IDs')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    audit = []
    counts = Counter()

    for result in results:
        question_id = str(result['id'])
        gold = labels[question_id]
        gold_hit = bool(
            set(gold['gold_doc_ids'])
            & set(result.get('doc_ids', [])[:10])
        )
        status = result.get('status')
        answer = result.get('answer', '')
        correct = (
            status == 'ok'
            and bool(answer.strip())
            and normalize_answer(answer)
                == normalize_answer(gold['answer'])
        )

        if correct:
            category = 'candidate_correct_answer'
        elif status == 'ok':
            category = 'discard_wrong_answer'
        elif status == 'no_answer' and not gold_hit:
            category = 'candidate_justified_abstention'
        elif status == 'no_answer':
            category = 'discard_abstention_with_gold_retrieved'
        else:
            category = 'discard_other_status'

        sample = None
        outcome = category
        if category.startswith('candidate_'):
            events = load_events(
                args.run_dir / 'traces' / f'{question_id}.jsonl')
            sample, outcome = extract_supervision(
                result, questions[question_id], events)
            if sample is not None:
                samples.append(sample)

        counts[outcome] += 1
        audit.append({
            'id': question_id,
            'category': category,
            'outcome': outcome,
            'gold_hit_at_10': gold_hit,
            'source_tool': sample['source_tool'] if sample else None,
        })

    samples.sort(key=lambda row: row['id'])
    sft_path = args.output_dir / 'sft_train.jsonl'
    with sft_path.open('w', encoding='utf-8') as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with (args.output_dir / 'selection_audit.jsonl').open(
            'w', encoding='utf-8') as handle:
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    stats = {
        'teacher_run': str(args.run_dir),
        'training_split': 'train',
        'selection_split': 'train',
        'dev_or_test_used': False,
        'input_results': len(results),
        'exported_samples': len(samples),
        'sample_types': dict(Counter(
            row['sample_type'] for row in samples)),
        'source_tools': dict(Counter(
            row['source_tool'] for row in samples)),
        'outcomes': dict(sorted(counts.items())),
        'sft_sha256': hashlib.sha256(sft_path.read_bytes()).hexdigest(),
        'target_format': 'JSON answer and evidence_ids; tool outputs are inputs only',
    }
    (args.output_dir / 'stats.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
