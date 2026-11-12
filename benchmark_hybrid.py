#!/usr/bin/env python3
"""Offline retrieval ablations; labels are used only to score produced rankings."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from agent.hybrid import CONFIG_PATH, INDEX_PATH as HYBRID_PATH, HybridIndex, fuse_rankings, load_config
from agent.solution import INDEX_PATH as BM25_PATH, ROOT, retrieve_bm25
from ir2026 import Corpus, CorpusTools, read_jsonl, write_jsonl


def metrics(rankings, labels):
    count = len(rankings)
    result = {'queries': count}
    for cutoff in [1, 5, 10]:
        result[f'gold_hit_at_{cutoff}'] = sum(
            bool(set(ids[:cutoff]) & set(labels[qid]['gold_doc_ids']))
            for qid, ids in rankings.items()) / count
    result['evidence_recall_at_10'] = sum(
        len(set(ids[:10]) & set(labels[qid]['evidence_doc_ids'])) / len(labels[qid]['evidence_doc_ids'])
        for qid, ids in rankings.items()) / count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'dev'], default='train')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--select-config', action='store_true', help='Select fusion weights on train only')
    args = parser.parse_args()
    if args.select_config and args.split != 'train':
        parser.error('Weight search is restricted to train; dev is a held-out evaluation')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(ROOT / 'corpus')
    manifest = hashlib.sha256((ROOT / 'corpus' / 'documents.jsonl').read_bytes()).hexdigest()
    index = HybridIndex(HYBRID_PATH, manifest)
    questions = read_jsonl(ROOT / 'datas' / f'{args.split}.jsonl')
    config = load_config()
    settings = {'bm25': None, 'tfidf': None, 'lsa': None, 'hybrid': config['weights']}
    if args.select_config:
        settings = {name: None for name in ['bm25', 'tfidf', 'lsa']}
        for bm25 in [0.5, 1.0, 2.0]:
            for lsa in [0.0, 0.25, 0.5, 1.0]:
                settings[f'fusion_b{bm25}_l{lsa}'] = {'bm25': bm25, 'tfidf': 1.0, 'lsa': lsa}
    predictions = {name: {} for name in settings}
    rows = []
    with sqlite3.connect(f'file:{BM25_PATH}?mode=ro', uri=True) as db:
        if db.execute("SELECT value FROM metadata WHERE key='manifest_sha256'").fetchone() != (manifest,):
            raise RuntimeError('BM25 index does not match the corpus')
        for position, question in enumerate(questions, 1):
            started = time.monotonic()
            tools = CorpusTools(corpus, question['id'], method_type='retrieval')
            bm25, _ = retrieve_bm25(db, question['question'], tools)
            lanes, _ = index.search(question['question'], config['candidate_limit'])
            lanes['bm25'] = bm25[:config['candidate_limit']]
            for name, weights in settings.items():
                hits = lanes[name] if weights is None else fuse_rankings(lanes, weights, config['rrf_constant'])
                predictions[name][question['id']] = [doc_id for doc_id, _ in hits[:10]]
            rows.append({'id': question['id'], 'rankings': lanes,
                         'elapsed_seconds': time.monotonic() - started})
            print(f'{position}/{len(questions)} id={question["id"]}', flush=True)
    # Labels are read after all retrieval calls are complete.
    labels = {row['id']: row for row in read_jsonl(ROOT / 'datas' / f'{args.split}_labels.jsonl')}
    report = {name: metrics(ids, labels) for name, ids in predictions.items()}
    selected = None
    if args.select_config:
        options = [name for name, weights in settings.items() if weights is not None]
        selected = max(options, key=lambda name: (
            report[name]['gold_hit_at_10'], report[name]['evidence_recall_at_10'],
            report[name]['gold_hit_at_5'], -settings[name]['lsa'],
        ))
        config.update(weights=settings[selected], selection_split='train', selection_queries=len(questions))
        CONFIG_PATH.write_text(json.dumps(config, indent=2) + '\n')
    output = {'split': args.split, 'selected': selected, 'config': config, 'ablations': report}
    write_jsonl(args.output / 'rankings.jsonl', rows)
    (args.output / 'ablation.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
