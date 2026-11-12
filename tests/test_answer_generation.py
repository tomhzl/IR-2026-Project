import io
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.answer import ModelAPIError, extract_percentage_answer, generate_answer, load_api_environment
from agent import solution
from agent.prompts import build_messages, requires_v1_verification
from ir2026 import Corpus, CorpusTools, write_jsonl


class AnswerGenerationTests(unittest.TestCase):
    def test_evidence_windows_cover_opening_and_distant_answer_within_quote_limit(self):
        class FakeCorpus:
            def text(self, doc_id):
                return source

        class FakeTools:
            corpus = FakeCorpus()

            def record_retrieval(self, **kwargs):
                pass

        class EmptyDB:
            def execute(self, *args):
                return []

        source = ('x' * 1600 + ' First answer. ' + 'x' * 8950
                  + ' California Pizza Kitchen. ' + 'x' * 500
                  + ' restaurant ' + 'x' * 500)
        snippets = solution.collect_snippets(
            EmptyDB(), ['top', 'd1'],
            {'top': (0, source[:1800]), 'd1': (0, source[:1800])},
            'Find the restaurant in the acknowledgments.', FakeTools())
        self.assertTrue(all(20 <= len(item['text']) <= 1200 for item in snippets))
        self.assertTrue(any('First answer' in item['text'] for item in snippets))
        self.assertTrue(any('California Pizza Kitchen' in item['text'] for item in snippets))

    def test_explicit_percentage_uses_matching_subject(self):
        question = 'What was the percentage decrease in non-GAAP operating expenses compared to 2020?'
        snippets = [{'id': 'S1', 'doc_id': 'd1', 'text': (
            'Fiscal 2021 compared to Fiscal 2020: total operating expenses decreased 10% '
            'and total non-GAAP operating expenses decreased 7%.'
        )}]
        self.assertEqual(extract_percentage_answer(question, snippets), ('7%', ['S1']))

    def test_ambiguous_percentage_abstains(self):
        question = 'What was the percentage decrease in non-GAAP operating expenses?'
        snippets = [
            {'id': 'S1', 'doc_id': 'd1', 'text': 'Non-GAAP operating expenses decreased 7%.'},
            {'id': 'S2', 'doc_id': 'd2', 'text': 'Non-GAAP operating expenses decreased 9%.'},
        ]
        self.assertEqual(extract_percentage_answer(question, snippets), ('', []))

    def test_model_response_parsing_and_usage(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"answer":"7%","evidence_ids":["S1"]}'}}],
            'usage': {'prompt_tokens': 123, 'completion_tokens': 15}}
        with patch.dict('os.environ', {'IR2026_API_KEY': 'test-key'}, clear=True):
            with patch('agent.answer.request.urlopen', return_value=io.BytesIO(
                    json.dumps(response).encode('utf-8'))) as opened:
                parsed, input_tokens, output_tokens = generate_answer(
                    'What was the decrease?', [{'id': 'S1', 'doc_id': 'd1', 'text': 'Decreased 7%.'}],
                    'test-model')
        self.assertEqual(parsed['answer'], '7%')
        self.assertEqual((input_tokens, output_tokens), (123, 15))
        self.assertEqual(opened.call_args.args[0].full_url,
                         'https://api.deepseek.com/chat/completions')
        sent = json.loads(opened.call_args.args[0].data)
        self.assertEqual(sent['thinking'], {'type': 'disabled'})

    def test_payment_error_keeps_status_code(self):
        with patch.dict('os.environ', {'IR2026_API_KEY': 'test-key'}, clear=True):
            with patch('agent.answer.request.urlopen', side_effect=HTTPError(
                    'https://api.deepseek.com/chat/completions', 402, 'Payment required', {}, None)):
                with self.assertRaises(ModelAPIError) as raised:
                    generate_answer('Question', [], 'test-model')
        self.assertEqual(raised.exception.status_code, 402)

    def test_siliconflow_uses_its_key_and_disables_thinking(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"answer":"blue","evidence_ids":["S1"]}'}}],
            'usage': {'prompt_tokens': 23, 'completion_tokens': 9}}
        environment = {'IR2026_API_BASE': 'https://api.siliconflow.cn/v1',
                       'SILICONFLOW_API_KEY': 'siliconflow-test-key', 'DEEPSEEK_API_KEY': 'other-test-key'}
        with patch.dict('os.environ', environment, clear=True):
            with patch('agent.answer.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as opened:
                answer, _, _ = generate_answer('Colour?', [{'id':'S1','doc_id':'d1','text':'A blue marker.'}],
                                                'Qwen/Qwen3.5-4B')
        req = opened.call_args.args[0]
        payload = json.loads(req.data)
        self.assertEqual(req.full_url, 'https://api.siliconflow.cn/v1/chat/completions')
        self.assertEqual(req.get_header('Authorization'), 'Bearer siliconflow-test-key')
        self.assertIs(payload['enable_thinking'], False)
        self.assertNotIn('thinking', payload)
        self.assertEqual(payload['response_format']['type'], 'json_object')
        self.assertEqual(answer['answer'], 'blue')

    def test_siliconflow_verification_has_strict_schema_and_optional_thinking(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"supported":false}'}}]}
        for flag in ['0', '1']:
            with patch.dict('os.environ', {'IR2026_API_BASE': 'https://api.siliconflow.cn/v1',
                                          'SILICONFLOW_API_KEY': 'test-key', 'IR2026_VERIFY_THINKING': flag}, clear=True):
                with patch('agent.answer.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as opened:
                    generate_answer('Colour?', [{'id':'S1','doc_id':'d1','text':'A blue marker.'}],
                                    'Qwen/Qwen3.5-4B', proposed_answer='blue', answer_policy='verified')
            payload = json.loads(opened.call_args.args[0].data)
            self.assertEqual(payload['enable_thinking'], flag == '1')
            if flag == '1':
                self.assertEqual(payload['thinking_budget'], 1024)
            schema = payload['response_format']['json_schema']
            self.assertTrue(schema['strict'])
            props = schema['schema']['properties']
            self.assertEqual(props['supported']['type'], 'boolean')
            self.assertEqual(props['type_matches']['type'], 'boolean')
            self.assertEqual(props['checks']['maxItems'], 4)
            self.assertEqual(props['answer_support']['items']['properties']['snippet_id']['enum'], ['S1'])

    def test_v1_keeps_legacy_prompt_and_non_thinking_audit(self):
        snippets = [{'id':'S1','doc_id':'d1','text':'A blue marker.'}]
        primary = build_messages('Colour?', snippets)[0]['content']
        fallback = build_messages('Colour?', snippets, relaxed=True)[0]['content']
        self.assertIn('background clues need not all be repeated', primary)
        self.assertIn('You may answer when background clues are absent', fallback)
        self.assertTrue(requires_v1_verification('1. One clue. 2. Another clue. 3. Third clue.'))
        self.assertFalse(requires_v1_verification('Who founded Orion?'))
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"supported":true,"missing_clues":[]}'}}]}
        with patch.dict('os.environ', {'IR2026_API_BASE':'https://api.siliconflow.cn/v1',
                                      'SILICONFLOW_API_KEY':'test-key', 'IR2026_VERIFY_THINKING':'1'}, clear=True):
            with patch('agent.answer.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as opened:
                generate_answer('Colour?', snippets, 'Qwen/Qwen3.5-4B', proposed_answer='blue')
        payload = json.loads(opened.call_args.args[0].data)
        self.assertIs(payload['enable_thinking'], False)
        self.assertEqual(payload['max_tokens'], 1200)
        self.assertEqual(payload['response_format']['type'], 'json_object')

    def test_dotenv_assignments_do_not_execute_shell_or_override_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('SILICONFLOW_API_KEY="file-test-key"\nDEEPSEEK_API_KEY=ignored\nUNRELATED=unused\n')
            with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'environment-test-key'}, clear=True):
                load_api_environment(path)
                import os
                self.assertEqual(os.environ['SILICONFLOW_API_KEY'], 'file-test-key')
                self.assertEqual(os.environ['DEEPSEEK_API_KEY'], 'environment-test-key')
                self.assertNotIn('UNRELATED', os.environ)

    def test_removed_local_model_is_rejected_before_api_request(self):
        with patch('agent.answer.request.urlopen') as opened:
            with self.assertRaisesRegex(ValueError, 'Local model inference has been removed'):
                generate_answer('Colour?', [{'id':'S1','doc_id':'d1','text':'A blue marker.'}],
                                'local:removed-model')
        opened.assert_not_called()

    def test_solution_uses_model_selected_local_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_root = root / 'corpus'
            (corpus_root / 'docs').mkdir(parents=True)
            source = ('The Orion research project was founded by Alpha University in 1984. '
                      'This sentence provides the source evidence for the answer.')
            (corpus_root / 'docs' / 'd1.txt').write_text(source)
            write_jsonl(corpus_root / 'documents.jsonl', [
                {'doc_id': 'd1', 'path': 'docs/d1.txt',
                 'sha256': hashlib.sha256(source.encode()).hexdigest()}
            ])
            index = root / 'index.sqlite3'
            db = sqlite3.connect(index)
            db.execute("CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED, start UNINDEXED, body, tokenize='porter unicode61')")
            db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('INSERT INTO passages(doc_id, start, body) VALUES (?, ?, ?)', ('d1', 0, source))
            digest = hashlib.sha256((corpus_root / 'documents.jsonl').read_bytes()).hexdigest()
            db.execute('INSERT INTO metadata VALUES (?, ?)', ('manifest_sha256', digest))
            db.commit()
            db.close()
            tools = CorpusTools(Corpus(corpus_root), 'q1', method_type='retrieval')
            quote = source[:source.index('.') + 1]
            with patch.object(solution, 'ROOT', root), patch.object(solution, 'INDEX_PATH', index):
                with patch.object(solution, 'generate_answer', side_effect=[
                        ({'answer': 'Alpha University', 'evidence_ids': ['S1']}, 20, 5),
                        ({'supported': True, 'type_matches': True, 'missing_clues': [],
                          'answer_support': [{'snippet_id': 'S1', 'quote': quote}],
                          'checks': [{'claim': 'Alpha University founded Orion', 'status': 'supported',
                                      'snippet_id': 'S1', 'quote': quote}]}, 10, 5),
                ]) as generated:
                    result = solution.solve({'id': 'q1', 'question': 'Who founded Orion?'},
                                            tools, {'model': 'test-model', 'answer_policy': 'verified'})
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['answer'], 'Alpha University')
            self.assertIn(result['evidence'][0]['text'], source)
            self.assertEqual(result['usage']['model_input_tokens'], 30)
            self.assertEqual(generated.call_count, 2)
            self.assertEqual(generated.call_args.kwargs['proposed_answer'], 'Alpha University')
            self.assertEqual(result['candidate_answer'], 'Alpha University')
            self.assertEqual(result['verification']['status'], 'verified')
            v1_tools = CorpusTools(Corpus(corpus_root), 'q1', method_type='retrieval')
            with patch.object(solution, 'ROOT', root), patch.object(solution, 'INDEX_PATH', index):
                with patch.object(solution, 'generate_answer', return_value=(
                        {'answer':'Alpha University','evidence_ids':['S1']}, 20, 5)) as generated:
                    v1 = solution.solve({'id':'q1','question':'Who founded Orion?'}, v1_tools, {'model':'test-model'})
            self.assertEqual(v1['answer'], 'Alpha University')
            self.assertEqual(generated.call_count, 1)
            self.assertNotIn('verification', v1)
            self.assertFalse(any(e['tool']=='answer_verification' for e in v1_tools.trace))
            for supported in [True, False]:
                numbered_tools = CorpusTools(Corpus(corpus_root), 'q1', method_type='retrieval')
                with patch.object(solution, 'ROOT', root), patch.object(solution, 'INDEX_PATH', index):
                    with patch.object(solution, 'generate_answer', side_effect=[
                            ({'answer':'Alpha University','evidence_ids':['S1']},20,5),
                            ({'supported':supported,'missing_clues':[]},10,5)]) as generated:
                        numbered = solution.solve({'id':'q1','question':'1. Orion research. 2. Founded in 1984. 3. University. Who founded Orion?'},
                                                  numbered_tools, {'model':'test-model'})
                self.assertEqual(numbered['answer'], 'Alpha University' if supported else '')
                self.assertEqual(generated.call_count, 2)
                self.assertEqual(generated.call_args.kwargs['answer_policy'], 'v1')
            failed_tools = CorpusTools(Corpus(corpus_root), 'q1', method_type='retrieval')
            with patch.object(solution, 'ROOT', root), patch.object(solution, 'INDEX_PATH', index):
                with patch.object(solution, 'generate_answer', side_effect=ModelAPIError(402)) as failed:
                    rejected = solution.solve({'id': 'q1', 'question': 'Who founded Orion?'},
                                              failed_tools, {'model': 'test-model'})
            self.assertEqual(rejected['status'], 'error')
            self.assertEqual(rejected['api_status_code'], 402)
            self.assertEqual(rejected['answer'], '')
            self.assertTrue(rejected['doc_ids'])
            self.assertEqual(failed.call_count, 1)

if __name__ == '__main__':
    unittest.main()
