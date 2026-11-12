#!/usr/bin/env python3
"""Build full-document TF-IDF and LSA indexes from the local corpus only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from agent.hybrid import FORMAT_VERSION, INDEX_PATH, ROOT, vectorizer
from ir2026 import Corpus


def build(path, corpus_root=ROOT / 'corpus', max_features=120000, dimensions=192):
    started = time.monotonic()
    corpus = Corpus(corpus_root)
    doc_ids = sorted(corpus.docs)
    encoder = vectorizer(max_features=max_features, min_df=2, max_df=0.95)
    print(f'Fitting TF-IDF on {len(doc_ids)} complete documents...', flush=True)
    matrix = encoder.fit_transform(corpus.text(doc_id) for doc_id in doc_ids)
    dimensions = min(dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if dimensions < 1:
        raise ValueError('LSA needs at least two documents and two vocabulary features')
    print(f'Fitting {dimensions}-dimensional LSA on {matrix.shape}, nnz={matrix.nnz}...', flush=True)
    model = TruncatedSVD(n_components=dimensions, n_iter=5, random_state=2026)
    with threadpool_limits(limits=4):
        dense = normalize(model.fit_transform(matrix)).astype(np.float32)
    metadata = {
        'format_version': FORMAT_VERSION,
        'manifest_sha256': hashlib.sha256((Path(corpus_root) / 'documents.jsonl').read_bytes()).hexdigest(),
        'documents': len(doc_ids), 'vocabulary': matrix.shape[1], 'dimensions': dimensions,
        'model': f'corpus-lsa-{dimensions}-v1', 'sklearn_version': sklearn.__version__,
        'fit_source': 'local corpus only; no questions or labels',
        'max_features': max_features, 'min_df': 2, 'max_df': 0.95,
        'ngram_range': [1, 2], 'sublinear_tf': True, 'random_state': 2026,
        'explained_variance': float(model.explained_variance_ratio_.sum()),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.hybrid-', suffix='.npz', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(
                stream, metadata=json.dumps(metadata), doc_ids=np.asarray(doc_ids),
                terms=encoder.get_feature_names_out().astype(str), idf=encoder.idf_,
                data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                shape=np.asarray(matrix.shape), components=model.components_, dense=dense,
            )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    report = {**metadata, 'index': str(path), 'build_seconds': round(time.monotonic() - started, 3),
              'bytes': path.stat().st_size}
    path.with_suffix('.build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=INDEX_PATH)
    parser.add_argument('--max-features', type=int, default=120000)
    parser.add_argument('--dimensions', type=int, default=192)
    args = parser.parse_args()
    if args.max_features < 2 or args.dimensions < 1:
        parser.error('max-features must be >= 2 and dimensions must be positive')
    build(args.output, max_features=args.max_features, dimensions=args.dimensions)
