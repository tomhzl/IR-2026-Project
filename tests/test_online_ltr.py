import json
from pathlib import Path
import tempfile
import unittest

from agent.hybrid import load_config
from agent.rerank import load_model, rank_row, rerank


ROOT = Path(__file__).resolve().parents[1]


class FakeCorpus:
    def text(self, doc_id):
        return {
            'a': 'Alpha',
            'b': 'Alpha beta',
            'c': 'beta',
        }[doc_id]


class OnlineLTRTests(unittest.TestCase):
    def test_online_rerank_matches_offline_score_and_rank(self):
        question = {'id': 'q1', 'question': 'Alpha beta'}
        lanes = {
            'bm25': [('a', 0.03), ('b', 0.02)],
            'tfidf': [('b', 0.8), ('c', 0.6)],
        }
        settings = {
            'weights': {
                'bm25': 0.5,
                'tfidf': 1.0,
                'lsa': 0.0,
            },
            'rrf_constant': 30,
        }
        model = {
            'training_split': 'train',
            'selection_split': 'train',
            'feature_names': ['query_coverage'],
            'scaler_mean': [0.0],
            'scaler_scale': [1.0],
            'coefficients': [2.0],
        }

        ranked = rerank(
            question, lanes, FakeCorpus(), settings, model
        )
        row = {
            'candidates': [
                {
                    'doc_id': doc_id,
                    'features': {'query_coverage': score / 2},
                }
                for doc_id, score in ranked
            ]
        }

        self.assertEqual(
            [doc_id for doc_id, _ in ranked],
            ['b', 'a', 'c'],
        )
        self.assertEqual(
            rank_row(row, model), ['b', 'a', 'c']
        )

    def test_model_metadata_and_default_presets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps({
                'training_split': 'dev',
                'selection_split': 'train',
                'feature_names': ['x'],
                'scaler_mean': [0.0],
                'scaler_scale': [1.0],
                'coefficients': [1.0],
            }))
            with self.assertRaises(ValueError):
                load_model(path)

        original = load_config(
            ROOT / 'configs' / 'qwen_bailian_v1.json'
        )
        selected = load_config(
            ROOT / 'configs' / 'qwen_bailian_ltr_v1.json'
        )
        self.assertNotIn('reranker', original)
        self.assertEqual(
            selected['reranker']['type'], 'pairwise_ltr'
        )
        self.assertEqual(
            selected['weights'], original['weights']
        )
        self.assertEqual(
            selected['candidate_limit'],
            original['candidate_limit'],
        )


if __name__ == '__main__':
    unittest.main()
