import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.answer import generate_answer
from agent.local_model import parse_local_spec
from run_qwen_hybrid import main as preset_main


class LocalModelTests(unittest.TestCase):
    def test_model_spec_without_adapter(self):
        self.assertEqual(
            parse_local_spec('local:models/base'),
            ('models/base', None),
        )

    def test_model_spec_with_adapter(self):
        self.assertEqual(
            parse_local_spec('local:models/base@output/lora'),
            ('models/base', 'output/lora'),
        )

    def test_empty_base_or_adapter_is_rejected(self):
        for value in ['local:', 'local:models/base@']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_local_spec(value)

    @patch('agent.local_model.generate_local')
    def test_answer_generation_dispatches_to_local_model(self, local):
        local.return_value = {
            'choices': [{
                'finish_reason': 'stop',
                'message': {'content': '```json\n{"answer":"Alice","evidence_ids":[1]}\n```'},
            }],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 8},
        }
        snippets = [{
            'id': 'S1', 'doc_id': 'd1', 'start': 0,
            'text': 'Alice wrote this example.',
        }]
        response, inputs, outputs = generate_answer(
            'Who wrote the example?', snippets, 'local:models/base')
        self.assertEqual(response['answer'], 'Alice')
        self.assertEqual(response['evidence_ids'], ['S1'])
        self.assertEqual((inputs, outputs), (20, 8))
        messages, label = local.call_args.args[:2]
        self.assertEqual(label, 'local:models/base')
        self.assertIn('Alice wrote this example.', messages[1]['content'])
        self.assertEqual(local.call_args.kwargs['max_new_tokens'], 768)

    @patch('run_qwen_hybrid.run_main', return_value=0)
    def test_local_preset_does_not_select_an_api_provider(self, run_main):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preset = {
                'agent': 'agent.solution',
                'method_type': 'retrieval',
                'method_label': 'local-test',
                'model_label': 'local:models/base',
                'retriever': 'hybrid',
                'answer_policy': 'v1',
                'hybrid': {},
            }
            path = root / 'preset.json'
            path.write_text(json.dumps(preset))
            output = root / 'output'
            code = preset_main([
                '--preset', str(path), '--split', 'dev',
                '--output', str(output),
            ])
        self.assertEqual(code, 0)
        command = run_main.call_args.args[0]
        self.assertIn('--model-label', command)
        self.assertNotIn('--api-provider', command)


if __name__ == '__main__':
    unittest.main()
