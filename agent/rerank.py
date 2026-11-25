"""Shared pairwise-LTR inference for offline experiments and online retrieval."""

import json
import math
from pathlib import Path

import numpy as np

from experiments.build_rerank_features import build_query_row


def matrix(row, feature_names):
    return np.asarray([
        [candidate['features'][name] for name in feature_names]
        for candidate in row['candidates']
    ], dtype=np.float64)


def score_row(row, model):
    if not row['candidates']:
        return []

    features = matrix(row, model['feature_names'])
    mean = np.asarray(model['scaler_mean'])
    scale = np.asarray(model['scaler_scale'])
    coefficients = np.asarray(model['coefficients'])
    scores = ((features - mean) / scale) @ coefficients

    return sorted(
        zip(
            (item['doc_id'] for item in row['candidates']),
            (float(value) for value in scores),
        ),
        key=lambda item: (-item[1], item[0]),
    )


def rank_row(row, model):
    return [doc_id for doc_id, _ in score_row(row, model)]


def load_model(path):
    model = json.loads(Path(path).read_text(encoding='utf-8'))

    if (model.get('training_split') != 'train'
            or model.get('selection_split') != 'train'):
        raise ValueError(
            'LTR model must be trained and selected on train only'
        )

    names = model.get('feature_names')
    if (not isinstance(names, list) or not names
            or len(names) != len(set(names))):
        raise ValueError('Invalid LTR feature names')

    vectors = [
        model.get(key)
        for key in ('scaler_mean', 'scaler_scale', 'coefficients')
    ]
    if any(
        not isinstance(vector, list) or len(vector) != len(names)
        for vector in vectors
    ):
        raise ValueError('LTR model vector lengths differ')

    if not all(
        math.isfinite(value)
        for vector in vectors for value in vector
    ):
        raise ValueError('LTR model contains non-finite values')

    if any(value <= 0 for value in model['scaler_scale']):
        raise ValueError('LTR scaler scales must be positive')

    return model


def rerank(question, lanes, corpus, settings, model, limit=10):
    row = build_query_row(
        question,
        {'id': question['id'], 'rankings': lanes},
        corpus,
        settings,
    )
    return score_row(row, model)[:limit]
