import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.answer import generate_answer
import run_qwen_hybrid


ROOT = Path(__file__).resolve().parents[1]


class DashScopeProviderTests(unittest.TestCase):
    def test_dashscope_uses_generic_key_and_disables_thinking(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"answer":"blue","evidence_ids":["S1"]}'}}],
            'usage': {'prompt_tokens': 23, 'completion_tokens': 9}}
        for base in ['https://dashscope.aliyuncs.com/compatible-mode/v1',
                     'https://dashscope-intl.aliyuncs.com/compatible-mode/v1']:
            with self.subTest(base=base):
                environment = {'IR2026_API_BASE': base, 'IR2026_API_KEY': 'dashscope-test-key'}
                with patch.dict('os.environ', environment, clear=True):
                    with patch('agent.answer.request.urlopen',
                               return_value=io.BytesIO(json.dumps(response).encode())) as opened:
                        answer, input_tokens, output_tokens = generate_answer(
                            'Colour?', [{'id': 'S1', 'doc_id': 'd1', 'text': 'A blue marker.'}],
                            'qwen3.5-flash')
                req = opened.call_args.args[0]
                payload = json.loads(req.data)
                self.assertEqual(req.full_url, base + '/chat/completions')
                self.assertEqual(req.get_header('Authorization'), 'Bearer dashscope-test-key')
                self.assertIs(payload['enable_thinking'], False)
                self.assertNotIn('thinking', payload)
                self.assertNotIn('thinking_budget', payload)
                self.assertEqual(payload['response_format']['type'], 'json_object')
                self.assertEqual(answer['answer'], 'blue')
                self.assertEqual((input_tokens, output_tokens), (23, 9))

    @patch('run_qwen_hybrid.run_main', return_value=0)
    def test_bailian_preset_is_forwarded_without_an_api_call(self, run_main):
        preset = ROOT / 'configs' / 'qwen_bailian_v1.json'
        self.assertEqual(run_qwen_hybrid.main([
            '--preset', str(preset), '--split', 'dev', '--output', 'output/test-bailian',
            '--limit', '3']), 0)
        command = run_main.call_args.args[0]

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(value('--model-label'), 'qwen3.5-flash')
        self.assertEqual(value('--api-provider'), 'dashscope')
        self.assertEqual(value('--hybrid-config'), str(preset))
        self.assertEqual(value('--output'), 'output/test-bailian')
        self.assertEqual(value('--limit'), '3')

    @patch('run_qwen_hybrid.run_main', return_value=0)
    def test_default_siliconflow_preset_is_unchanged(self, run_main):
        self.assertEqual(run_qwen_hybrid.main(['--split', 'dev']), 0)
        command = run_main.call_args.args[0]

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(value('--model-label'), 'Qwen/Qwen3.5-4B')
        self.assertEqual(value('--api-provider'), 'siliconflow')
        self.assertTrue(value('--output').endswith('qwen_hybrid_v1_dev'))


if __name__ == '__main__':
    unittest.main()
