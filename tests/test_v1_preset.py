import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.hybrid import load_config
from ir2026 import DataError, write_jsonl
from run import load_resume


ROOT = Path(__file__).resolve().parents[1]


class V1PresetTests(unittest.TestCase):
    def test_selected_entry_shows_frozen_v1_config_without_api(self):
        result = subprocess.run([sys.executable, 'run_qwen_hybrid.py', '--show-config'],
                                cwd=ROOT, text=True, capture_output=True, timeout=15, check=True)
        preset = json.loads(result.stdout)
        self.assertEqual(preset['answer_policy'], 'v1')
        self.assertEqual(preset['api_provider'], 'siliconflow')
        self.assertEqual(preset['model_label'], 'Qwen/Qwen3.5-4B')
        self.assertEqual(preset['hybrid']['weights'], {'bm25':0.5,'tfidf':1.0,'lsa':0.0})
        self.assertEqual(load_config(ROOT/'configs/qwen_hybrid_v1.json'), preset['hybrid'])

    def test_original_v1_run_can_resume_but_strict_policy_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'traces').mkdir()
            original = {'agent_module':'agent.solution','model':'Qwen/Qwen3.5-4B',
                        'api_provider':'siliconflow','method':'hybrid-qwen35-4b-v1'}
            (output/'run_info.json').write_text(json.dumps(original))
            questions = [{'id':'q1','question':'A question'}]
            results = [{'id':'q1','status':'ok','answer':'A'}]
            write_jsonl(output/'questions.jsonl', questions)
            write_jsonl(output/'results.jsonl', results)
            write_jsonl(output/'traces/q1.jsonl', [])
            current = {**original, 'answer_policy':'v1','verification_policy':'v1-numbered-clues',
                       'verification_thinking':False}
            before = (output/'run_info.json').read_bytes()
            self.assertEqual(load_resume(output,questions,current), results)
            self.assertEqual((output/'run_info.json').read_bytes(), before)
            with self.assertRaises(DataError):
                load_resume(output,questions,{**current,'answer_policy':'verified'})


if __name__ == '__main__':
    unittest.main()
