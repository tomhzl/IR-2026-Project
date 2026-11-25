#!/usr/bin/env python3
"""Train query-grouped pairwise logistic rerankers using train labels only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ir2026 import read_jsonl, write_jsonl


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(row, feature_names):
    return np.asarray([
        [candidate['features'][name] for name in feature_names]
        for candidate in row['candidates']
    ], dtype=np.float64)


def pairwise_data(rows, labels, feature_names):
    blocks = []
    used_queries = 0
    for row in rows:
        candidates = row['candidates']
        gold = set(labels[row['id']]['gold_doc_ids'])
        positive = [
            i for i, item in enumerate(candidates)
            if item['doc_id'] in gold
        ]
        negative = [
            i for i, item in enumerate(candidates)
            if item['doc_id'] not in gold
        ]
        if not positive or not negative:
            continue

        features = matrix(row, feature_names)
        differences = (
            features[positive][:, None, :]
            - features[negative][None, :, :]
        ).reshape(len(positive) * len(negative), -1)
        blocks.append(differences)
        used_queries += 1

    if not blocks:
        raise ValueError(
            'No train query has both positive and negative candidates'
        )

    differences = np.concatenate(blocks, axis=0)
    # 对称样本平衡类别，并消除对截距项的需要。
    x = np.concatenate([differences, -differences], axis=0)
    y = np.concatenate([
        np.ones(len(differences), dtype=np.int64),
        np.zeros(len(differences), dtype=np.int64),
    ])
    return x, y, used_queries, len(differences)


def fit_model(
    rows, labels, feature_names, c,
    max_iter=2000, random_state=2026
):
    x, y, used_queries, pair_count = pairwise_data(
        rows, labels, feature_names
    )
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    classifier = LogisticRegression(
        C=c,
        fit_intercept=False,
        solver='lbfgs',
        max_iter=max_iter,
        random_state=random_state,
    )
    classifier.fit(x, y)

    return {
        'feature_names': list(feature_names),
        'c': c,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'coefficients': classifier.coef_[0].tolist(),
        'training_queries': len(rows),
        'pairwise_queries': used_queries,
        'positive_negative_pairs': pair_count,
    }


def rank_row(row, model):
    if not row['candidates']:
        return []

    features = matrix(row, model['feature_names'])
    mean = np.asarray(model['scaler_mean'])
    scale = np.asarray(model['scaler_scale'])
    coefficients = np.asarray(model['coefficients'])
    scores = ((features - mean) / scale) @ coefficients

    order = sorted(
        zip(
            (item['doc_id'] for item in row['candidates']),
            scores,
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return [doc_id for doc_id, _ in order]


def evaluate(rows, labels, rankings):
    metrics = {'queries': len(rows)}
    for cutoff in (1, 5, 10):
        metrics[f'gold_hit_at_{cutoff}'] = sum(
            bool(
                set(rankings[row['id']][:cutoff])
                & set(labels[row['id']]['gold_doc_ids'])
            )
            for row in rows
        ) / len(rows)

    metrics['evidence_recall_at_10'] = sum(
        len(
            set(rankings[row['id']][:10])
            & set(labels[row['id']]['evidence_doc_ids'])
        ) / len(labels[row['id']]['evidence_doc_ids'])
        for row in rows
    ) / len(rows)
    return metrics


def rankings_for(rows, model=None):
    return {
        row['id']: (
            row['baseline_doc_ids']
            if model is None else rank_row(row, model)
        )
        for row in rows
    }


def cross_validate(rows, labels, feature_names, c, settings):
    splitter = KFold(
        n_splits=settings['folds'],
        shuffle=True,
        random_state=settings['random_state'],
    )
    oof = {}
    fold_metrics = []

    for train_indices, valid_indices in splitter.split(rows):
        train_rows = [rows[i] for i in train_indices]
        valid_rows = [rows[i] for i in valid_indices]
        model = fit_model(
            train_rows,
            labels,
            feature_names,
            c,
            settings['max_iter'],
            settings['random_state'],
        )
        predicted = rankings_for(valid_rows, model)
        oof.update(predicted)
        fold_metrics.append(
            evaluate(valid_rows, labels, predicted)
        )

    return evaluate(rows, labels, oof), fold_metrics, oof


def paired_rows(rows, labels, baseline, proposed):
    output = []
    for row in rows:
        qid = row['id']
        gold = set(labels[qid]['gold_doc_ids'])
        before = bool(set(baseline[qid][:10]) & gold)
        after = bool(set(proposed[qid][:10]) & gold)
        output.append({
            'id': qid,
            'baseline_hit': int(before),
            'ltr_hit': int(after),
            'outcome': (
                'win' if after and not before
                else ('loss' if before and not after else 'tie')
            ),
        })
    return output


def write_per_query(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['id', 'baseline_hit', 'ltr_hit', 'outcome']
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-features', type=Path, required=True)
    parser.add_argument('--dev-features', type=Path, required=True)
    parser.add_argument(
        '--config',
        type=Path,
        default=ROOT / 'configs' / 'pairwise_ltr_v1.json'
    )
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory')

    settings = json.loads(args.config.read_text(encoding='utf-8'))
    train_rows = read_jsonl(args.train_features)
    train_labels = {
        row['id']: row
        for row in read_jsonl(ROOT / 'datas' / 'train_labels.jsonl')
    }
    if set(train_labels) != {row['id'] for row in train_rows}:
        parser.error('Train feature IDs and train label IDs differ')

    primary = settings['primary_variant']
    primary_features = settings['variants'][primary]
    cv = {'grid': {}, 'folds': {}}

    for c in settings['c_grid']:
        metrics, folds, _ = cross_validate(
            train_rows, train_labels, primary_features, c, settings
        )
        cv['grid'][str(c)] = metrics
        cv['folds'][str(c)] = folds
        print(
            f'train CV {primary} C={c}: {json.dumps(metrics)}',
            flush=True
        )

    selected_c = max(settings['c_grid'], key=lambda c: (
        cv['grid'][str(c)]['gold_hit_at_10'],
        cv['grid'][str(c)]['gold_hit_at_5'],
        cv['grid'][str(c)]['evidence_recall_at_10'],
        cv['grid'][str(c)]['gold_hit_at_1'],
        -c,
    ))
    cv['selected_c'] = selected_c
    cv['selection_split'] = 'train'

    models = {}
    train_cv_metrics = {}
    for name, feature_names in settings['variants'].items():
        models[name] = fit_model(
            train_rows,
            train_labels,
            feature_names,
            selected_c,
            settings['max_iter'],
            settings['random_state'],
        )
        if name == primary:
            train_cv_metrics[name] = cv['grid'][str(selected_c)]
        else:
            metrics, _, _ = cross_validate(
                train_rows,
                train_labels,
                feature_names,
                selected_c,
                settings,
            )
            train_cv_metrics[name] = metrics
        print(
            f'fit {name}: {json.dumps(train_cv_metrics[name])}',
            flush=True
        )

    # dev 标签只在 train 完成选参和模型拟合后读取。
    dev_rows = read_jsonl(args.dev_features)
    dev_labels = {
        row['id']: row
        for row in read_jsonl(ROOT / 'datas' / 'dev_labels.jsonl')
    }
    if set(dev_labels) != {row['id'] for row in dev_rows}:
        parser.error('Dev feature IDs and dev label IDs differ')

    baseline = rankings_for(dev_rows)
    predicted = {
        name: rankings_for(dev_rows, model)
        for name, model in models.items()
    }
    report = {
        'selected_c': selected_c,
        'selection_split': 'train',
        'train_cv': train_cv_metrics,
        'train_baseline': evaluate(
            train_rows, train_labels, rankings_for(train_rows)
        ),
        'dev_baseline': evaluate(dev_rows, dev_labels, baseline),
        'dev': {
            name: evaluate(dev_rows, dev_labels, values)
            for name, values in predicted.items()
        },
    }

    paired = paired_rows(
        dev_rows, dev_labels, baseline, predicted[primary]
    )
    report['paired_dev'] = {
        outcome: sum(row['outcome'] == outcome for row in paired)
        for outcome in ('win', 'loss', 'tie')
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'cv_report.json').write_text(
        json.dumps(cv, indent=2) + '\n', encoding='utf-8'
    )
    (args.output / 'metrics.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8'
    )

    model = {
        'version': 1,
        'model_type': settings['model_type'],
        'training_split': 'train',
        'selection_split': 'train',
        'config_sha256': sha256(args.config),
        'train_features_sha256': sha256(args.train_features),
        **models[primary],
    }
    (args.output / 'model.json').write_text(
        json.dumps(model, indent=2) + '\n', encoding='utf-8'
    )
    write_per_query(args.output / 'dev_per_query.csv', paired)

    write_jsonl(args.output / 'dev_rankings.jsonl', [
        {
            'id': row['id'],
            'baseline_doc_ids': baseline[row['id']][:10],
            'ltr_doc_ids': predicted[primary][row['id']][:10],
            **{
                f'{name}_doc_ids': values[row['id']][:10]
                for name, values in predicted.items()
            },
        }
        for row in dev_rows
    ])

    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
