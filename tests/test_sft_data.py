import json
import unittest

from experiments.build_sft_data import extract_supervision


SNIPPETS = [
    {'id': 'S1', 'doc_id': 'd1', 'start': 0,
     'text': 'Alice wrote the first example in 1990.'},
    {'id': 'S2', 'doc_id': 'd1', 'start': 100,
     'text': 'The first document continues with more detail.'},
    {'id': 'S3', 'doc_id': 'd1', 'start': 200,
     'text': 'This third passage must not enter focused input.'},
    {'id': 'S4', 'doc_id': 'd2', 'start': 0,
     'text': 'Bob received the award in 2004.'},
]


def events(primary=None, fallback=None, extractive=None):
    rows = [{
        'tool': 'read_evidence_candidates',
        'output': {'snippets': SNIPPETS},
    }]
    if primary is not None:
        rows.append({'tool': 'answer_generation', 'output': primary})
    if fallback is not None:
        rows.append({
            'tool': 'answer_candidate_fallback',
            'output': fallback,
        })
    if extractive is not None:
        rows.append({'tool': 'extractive_answer', 'output': extractive})
    return rows


class SFTDataTests(unittest.TestCase):
    def test_primary_answer_becomes_assistant_json(self):
        result = {'id': '1', 'status': 'ok', 'answer': 'Alice'}
        sample, outcome = extract_supervision(
            result,
            'Who wrote the first example?',
            events(primary={
                'answer': 'Alice',
                'evidence_ids': ['S1'],
            }),
        )
        self.assertEqual(outcome, 'exported')
        self.assertEqual(sample['source_tool'], 'answer_generation')
        self.assertEqual(
            [row['role'] for row in sample['messages']],
            ['system', 'user', 'assistant'],
        )
        self.assertIn('Alice wrote the first example',
                      sample['messages'][1]['content'])
        self.assertEqual(
            json.loads(sample['messages'][2]['content']),
            {'answer': 'Alice', 'evidence_ids': ['S1']},
        )

    def test_fallback_uses_focused_snippets(self):
        result = {'id': '2', 'status': 'ok', 'answer': 'Bob'}
        sample, outcome = extract_supervision(
            result,
            'Who received the award?',
            events(
                primary={'answer': '', 'evidence_ids': []},
                fallback={'answer': 'Bob', 'evidence_ids': ['S4']},
            ),
        )
        self.assertEqual(outcome, 'exported')
        self.assertEqual(
            sample['source_tool'], 'answer_candidate_fallback')
        user = sample['messages'][1]['content']
        self.assertIn('[S1]', user)
        self.assertIn('[S2]', user)
        self.assertIn('[S4]', user)
        self.assertNotIn('[S3]', user)

    def test_abstention_has_empty_json_target(self):
        result = {'id': '3', 'status': 'no_answer', 'answer': ''}
        sample, outcome = extract_supervision(
            result,
            'An unsupported question',
            events(primary={'answer': '', 'evidence_ids': []}),
        )
        self.assertEqual(outcome, 'exported')
        self.assertEqual(sample['sample_type'], 'justified_abstention')
        self.assertEqual(
            json.loads(sample['messages'][2]['content']),
            {'answer': '', 'evidence_ids': []},
        )

    def test_deterministic_extraction_is_not_model_supervision(self):
        result = {'id': '4', 'status': 'ok', 'answer': '5%'}
        sample, outcome = extract_supervision(
            result,
            'What was the decrease?',
            events(extractive={'answer': '5%', 'evidence_ids': ['S1']}),
        )
        self.assertIsNone(sample)
        self.assertEqual(outcome, 'discard_deterministic_extraction')

    def test_unavailable_evidence_id_is_rejected(self):
        result = {'id': '5', 'status': 'ok', 'answer': 'Alice'}
        sample, outcome = extract_supervision(
            result,
            'Who wrote the first example?',
            events(primary={
                'answer': 'Alice',
                'evidence_ids': ['S999'],
            }),
        )
        self.assertIsNone(sample)
        self.assertEqual(outcome, 'discard_no_matching_teacher_action')


if __name__ == '__main__':
    unittest.main()
