import math
import unittest

from experiments.build_rerank_features import (
    FEATURE_NAMES, build_query_row, text_features,
)


class FakeCorpus:
    def __init__(self, texts):
        self.texts = texts

    def text(self, doc_id):
        return self.texts[doc_id]


class RerankFeatureTests(unittest.TestCase):
    def test_text_features_use_token_coverage_and_finite_length(self):
        features = text_features(
            'Alpha beta 2026',
            'Alpha only; 2026 appears twice 2026.'
        )
        self.assertAlmostEqual(features['query_coverage'], 2 / 3)
        self.assertAlmostEqual(features['opening_coverage'], 2 / 3)
        self.assertTrue(math.isfinite(features['log_doc_length']))

    def test_candidate_order_missing_lanes_and_feature_schema(self):
        question = {'id': 'q1', 'question': 'Alpha beta'}
        ranking = {
            'id': 'q1',
            'rankings': {
                'bm25': [['a', 0.03], ['b', 0.02]],
                'tfidf': [['b', 0.8], ['c', 0.6]],
                'lsa': [],
            },
        }
        corpus = FakeCorpus({
            'a': 'Alpha',
            'b': 'Alpha beta',
            'c': 'beta',
        })
        settings = {
            'weights': {
                'bm25': 0.5,
                'tfidf': 1.0,
                'lsa': 0.0,
            },
            'rrf_constant': 30,
        }
        row = build_query_row(
            question, ranking, corpus, settings
        )

        self.assertEqual(
            row['baseline_doc_ids'], ['b', 'c', 'a']
        )
        self.assertEqual(len(row['candidates']), 3)

        by_id = {
            item['doc_id']: item['features']
            for item in row['candidates']
        }
        self.assertEqual(by_id['b']['both_lanes'], 1.0)
        self.assertEqual(by_id['c']['bm25_present'], 0.0)
        self.assertEqual(by_id['c']['bm25_rrf'], 0.0)

        for features in by_id.values():
            self.assertEqual(
                set(features), set(FEATURE_NAMES)
            )
            self.assertTrue(all(
                math.isfinite(value)
                for value in features.values()
            ))


if __name__ == '__main__':
    unittest.main()
