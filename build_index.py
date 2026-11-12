#!/usr/bin/env python3
"""Build a passage-level SQLite FTS5 index from the released local corpus."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from ir2026 import Corpus


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / 'index' / 'passages.sqlite3'
CHUNK_CHARS = 1800
OVERLAP_CHARS = 250


def passages(text):
    step = CHUNK_CHARS - OVERLAP_CHARS
    for start in range(0, len(text), step):
        body = text[start:start + CHUNK_CHARS]
        if body.strip():
            yield start, body
        if start + CHUNK_CHARS >= len(text):
            break


def build(path):
    corpus = Corpus(ROOT / 'corpus')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.passages-', suffix='.sqlite3', dir=path.parent)
    os.close(fd)
    try:
        db = sqlite3.connect(temporary)
        try:
            db.execute("CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED, start UNINDEXED, body, tokenize='porter unicode61')")
            db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            count = 0
            for doc_id in corpus.docs:
                rows = [(doc_id, start, body) for start, body in passages(corpus.text(doc_id))]
                db.executemany('INSERT INTO passages(doc_id, start, body) VALUES (?, ?, ?)', rows)
                count += len(rows)
            manifest_hash = hashlib.sha256((ROOT / 'corpus' / 'documents.jsonl').read_bytes()).hexdigest()
            db.executemany('INSERT INTO metadata(key, value) VALUES (?, ?)', [
                ('manifest_sha256', manifest_hash),
                ('chunk_chars', str(CHUNK_CHARS)),
                ('overlap_chars', str(OVERLAP_CHARS)),
                ('documents', str(len(corpus.docs))),
                ('passages', str(count)),
            ])
            db.commit()
        finally:
            db.close()
        os.replace(temporary, path)
        print(json.dumps({'index': str(path), 'documents': len(corpus.docs), 'passages': count,
                          'bytes': path.stat().st_size}, ensure_ascii=False))
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    build(args.output.resolve())
