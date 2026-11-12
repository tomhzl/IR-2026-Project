import contextlib
import hashlib
import io
import sqlite3
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.hybrid import HybridIndex, fuse_rankings
from agent import hybrid, solution
from build_hybrid_index import build
from ir2026 import Corpus, CorpusTools, write_jsonl


class HybridTests(unittest.TestCase):
    def test_rrf_deduplicates_within_lane_and_combines_independent_lanes(self):
        lanes = {'bm25': [('a', 8), ('a', 7), ('b', 6)], 'tfidf': [('b', 0.9), ('c', 0.8)]}
        hits = fuse_rankings(lanes, {'bm25': 1, 'tfidf': 1}, constant=30)
        self.assertEqual([d for d, _ in hits], ['b', 'a', 'c'])
        self.assertAlmostEqual(hits[0][1], 1 / 32 + 1 / 31)
        self.assertEqual([d for d, _ in fuse_rankings(lanes, {'bm25': 1})], ['a', 'b'])

    def test_invalid_weights_fail(self):
        for weight in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                fuse_rankings({'bm25': [('d1', 1)]}, {'bm25': weight})

    def test_index_roundtrip_oov_and_corpus_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'corpus'
            corpus.mkdir()
            texts = {
                'a': 'Astronomy telescope galaxy observation. Telescope astronomy stars research.',
                'b': 'Astronomy galaxy satellite observation. Satellite stars research.',
                'c': 'Restaurant pizza kitchen cooking. Restaurant cooking chef food.',
                'd': 'Kitchen pizza restaurant cooking. Chef food recipe restaurant.',
            }
            rows = []
            for doc_id, text in texts.items():
                (corpus / f'{doc_id}.txt').write_text(text)
                rows.append({'doc_id': doc_id, 'path': f'{doc_id}.txt',
                             'sha256': hashlib.sha256(text.encode()).hexdigest()})
            write_jsonl(corpus / 'documents.jsonl', rows)
            path = root / 'hybrid.npz'
            with contextlib.redirect_stdout(io.StringIO()):
                build(path, corpus, max_features=100, dimensions=2)
            digest = hashlib.sha256((corpus / 'documents.jsonl').read_bytes()).hexdigest()
            index = HybridIndex(path, digest)
            lanes, features = index.search('astronomy galaxy observation')
            self.assertGreater(features, 0)
            self.assertIn(lanes['tfidf'][0][0], ['a', 'b'])
            self.assertIn(lanes['lsa'][0][0], ['a', 'b'])
            self.assertEqual(index.search('zzzzzzzzzz'), ({'tfidf': [], 'lsa': []}, 0))
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                HybridIndex(path, 'different corpus')
            # A document surfaced only by the new lanes still needs a valid
            # local quote and provenance in the final result.
            bm25_path = root / 'passages.sqlite3'
            with sqlite3.connect(bm25_path) as db:
                db.execute('CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED, start UNINDEXED, body)')
                db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)')
                db.execute('INSERT INTO metadata VALUES (?, ?)', ('manifest_sha256', digest))
            tools = CorpusTools(Corpus(corpus), 'q1', method_type='retrieval')
            config = {'model': None, 'retriever': 'hybrid', 'retrieval_only': True,
                      'hybrid': {'weights': {'bm25': 1, 'tfidf': 1, 'lsa': 0.25},
                                 'rrf_constant': 30, 'candidate_limit': 100}}
            with patch.object(solution, 'ROOT', root), patch.object(solution, 'INDEX_PATH', bm25_path), \
                    patch.object(hybrid, 'INDEX_PATH', path):
                result = solution.solve({'id': 'q1', 'question': 'astronomy galaxy observation'}, tools, config)
            self.assertIn(result['doc_ids'][0], ['a', 'b'])
            self.assertTrue(result['evidence'])
            for quote in result['evidence']:
                self.assertIn(quote['text'], texts[quote['doc_id']])
            self.assertTrue(any(event['tool'] == 'hybrid_rrf' for event in tools.trace))
            self.assertIsNone(result['usage']['model_input_tokens'])


if __name__ == '__main__':
    unittest.main()
