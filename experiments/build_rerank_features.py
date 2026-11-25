#!/usr/bin/env python3
"""Build label-free BM25/TF-IDF reranking features from frozen candidate lists."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.hybrid import fuse_rankings, load_config
from agent.retrieval import query_terms
from ir2026 import Corpus, keyed, read_jsonl, write_jsonl

TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9]*|\d{4}')
FEATURE_NAMES = (
    'bm25_present', 'tfidf_present', 'both_lanes',
    'bm25_rrf', 'tfidf_rrf', 'weighted_rrf',
    'bm25_score_ratio', 'tfidf_score_ratio',
    'query_coverage', 'opening_coverage', 'log_doc_length',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lane_map(hits):
    result = {}
    for rank, (doc_id, score) in enumerate(hits, 1):
        if doc_id in result:
            raise ValueError(f'Duplicate document in retrieval lane: {doc_id}')
        result[doc_id] = (rank, float(score))
    return result


def text_features(question, text):
    terms = query_terms(question)
    tokens = set(TOKEN_RE.findall(text.casefold()))
    opening = set(TOKEN_RE.findall(text[:1800].casefold()))
    denominator = max(len(terms), 1)
    return {
        'query_coverage': sum(term in tokens for term in terms) / denominator,
        'opening_coverage': sum(term in opening for term in terms) / denominator,
        'log_doc_length': math.log1p(len(text)),
    }


def build_query_row(question, ranking_row, corpus, settings):
    if ranking_row['id'] != question['id']:
        raise ValueError('Question and ranking IDs differ')
    lanes = {
        name: ranking_row['rankings'].get(name, [])
        for name in ('bm25', 'tfidf')
    }
    maps = {name: lane_map(hits) for name, hits in lanes.items()}
    pool = set().union(*(set(items) for items in maps.values()))
    if not pool:
        return {'id': question['id'], 'baseline_doc_ids': [], 'candidates': []}

    weights = settings['weights']
    constant = settings['rrf_constant']
    baseline = fuse_rankings(lanes, weights, constant, limit=len(pool))
    top_scores = {
        name: max((score for _, score in items.values()), default=0.0)
        for name, items in maps.items()
    }

    candidates = []
    for doc_id, _ in baseline:
        features = {}
        for name in ('bm25', 'tfidf'):
            rank, score = maps[name].get(doc_id, (0, 0.0))
            features[f'{name}_present'] = float(bool(rank))
            features[f'{name}_rrf'] = (
                1.0 / (constant + rank) if rank else 0.0
            )
            features[f'{name}_score_ratio'] = (
                score / top_scores[name]
                if rank and top_scores[name] > 0 else 0.0
            )

        features['both_lanes'] = float(
            all(doc_id in items for items in maps.values())
        )
        features['weighted_rrf'] = sum(
            weights[name] * features[f'{name}_rrf']
            for name in ('bm25', 'tfidf')
        )
        features.update(text_features(
            question['question'], corpus.text(doc_id)
        ))

        if (set(features) != set(FEATURE_NAMES)
                or not all(math.isfinite(v) for v in features.values())):
            raise ValueError(
                f'Invalid features for {question["id"]}/{doc_id}'
            )
        candidates.append({'doc_id': doc_id, 'features': features})

    return {
        'id': question['id'],
        'baseline_doc_ids': [
            doc_id for doc_id, _ in baseline[:10]
        ],
        'candidates': candidates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--split', choices=['train', 'dev'], required=True
    )
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--config', type=Path,
        default=ROOT / 'configs' / 'hybrid.json'
    )
    args = parser.parse_args(argv)

    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory')

    settings = load_config(args.config)
    weights = settings['weights']
    if (weights['lsa'] != 0
            or weights['bm25'] <= 0 or weights['tfidf'] <= 0):
        parser.error(
            'This v1 feature builder requires BM25+TF-IDF '
            'with LSA weight zero'
        )

    questions = read_jsonl(
        ROOT / 'datas' / f'{args.split}.jsonl'
    )
    rankings = keyed(read_jsonl(args.rankings), 'rankings')
    if set(rankings) != {row['id'] for row in questions}:
        parser.error('Question IDs and ranking IDs differ')

    corpus = Corpus(ROOT / 'corpus')
    rows = []
    for position, question in enumerate(questions, 1):
        rows.append(build_query_row(
            question, rankings[question['id']], corpus, settings
        ))
        if position % 10 == 0 or position == len(questions):
            print(
                f'{args.split}: {position}/{len(questions)}',
                flush=True
            )

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / 'features.jsonl', rows)

    metadata = {
        'version': 1,
        'split': args.split,
        'queries': len(rows),
        'candidates': sum(len(row['candidates']) for row in rows),
        'feature_names': list(FEATURE_NAMES),
        'labels_used': False,
        'rankings_sha256': sha256(args.rankings),
        'config_sha256': sha256(args.config),
    }
    (args.output / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
