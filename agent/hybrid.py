"""Local full-document TF-IDF and corpus-trained LSA retrieval.

Only the corpus is used to fit this index. Questions and labels are never read
here; evaluation and weight selection belong to the offline experiment script.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / 'index' / 'hybrid.npz'
CONFIG_PATH = ROOT / 'configs' / 'hybrid.json'
FORMAT_VERSION = 1


def vectorizer(**kwargs):
    return TfidfVectorizer(
        lowercase=True, strip_accents='unicode', stop_words='english',
        token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1, 2),
        sublinear_tf=True, norm='l2', dtype=np.float32, **kwargs,
    )


def fuse_rankings(rankings, weights, constant=30, limit=10):
    """Weighted RRF with one contribution per document per retrieval lane."""
    if constant <= 0 or limit < 1:
        raise ValueError('RRF constant and result limit must be positive')
    scores = {}
    for name, hits in rankings.items():
        weight = float(weights.get(name, 0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Retrieval weights must be finite and nonnegative')
        if not weight:
            continue
        seen = set()
        for doc_id, _ in hits:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (constant + len(seen))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def load_config(path=None):
    config = json.loads(Path(path or CONFIG_PATH).read_text())
    config = config.get('hybrid', config)
    if set(config['weights']) != {'bm25', 'tfidf', 'lsa'}:
        raise ValueError('Hybrid config must specify bm25, tfidf and lsa weights')
    if not any(config['weights'].values()):
        raise ValueError('At least one retrieval weight must be positive')
    return config


class HybridIndex:
    def __init__(self, path, manifest_sha256, include_lsa=True):
        if not Path(path).is_file():
            raise RuntimeError('Missing hybrid index; run python3 build_hybrid_index.py')
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata']))
            if self.metadata['format_version'] != FORMAT_VERSION:
                raise RuntimeError('Unsupported hybrid index version; rebuild it')
            if self.metadata['manifest_sha256'] != manifest_sha256:
                raise RuntimeError('Hybrid index does not match the corpus; rebuild it')
            self.doc_ids = data['doc_ids'].tolist()
            self.tfidf = csr_matrix((data['data'], data['indices'], data['indptr']),
                                    shape=tuple(data['shape']))
            self.components = data['components'] if include_lsa else None
            self.dense = data['dense'] if include_lsa else None
            self.vectorizer = vectorizer(vocabulary={term: i for i, term in enumerate(data['terms'])})
            self.vectorizer.idf_ = data['idf']
        if self.tfidf.shape[0] != len(self.doc_ids) or (include_lsa and self.dense.shape[0] != len(self.doc_ids)):
            raise RuntimeError('Inconsistent hybrid index dimensions; rebuild it')

    def search(self, question, limit=100):
        query = self.vectorizer.transform([question])
        if not query.nnz:
            return {'tfidf': [], 'lsa': []}, 0
        lexical = (self.tfidf @ query.T).toarray().ravel()
        semantic = None
        if self.components is not None:
            with threadpool_limits(limits=2):
                latent = np.asarray(query @ self.components.T).ravel()
                norm = np.linalg.norm(latent)
                semantic = self.dense @ (latent / norm) if norm > 0 else np.zeros(len(self.doc_ids))

        def ranked(scores):
            # Stable ties preserve corpus row order and make runs reproducible.
            order = np.argsort(-scores, kind='stable')[:limit]
            return [(self.doc_ids[i], float(scores[i])) for i in order if scores[i] > 0]

        return {'tfidf': ranked(lexical), 'lsa': ranked(semantic) if semantic is not None else []}, int(query.nnz)


@lru_cache(maxsize=1)
def load_index(path, manifest_sha256, modified_ns, include_lsa=True):
    """Cache within a persistent worker; mtime invalidates rebuilt artifacts."""
    return HybridIndex(path, manifest_sha256, include_lsa)
