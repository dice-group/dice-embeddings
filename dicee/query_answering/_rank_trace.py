"""Transactional rank records for inference-free tie-policy analysis."""

import json
import sqlite3
from pathlib import Path

from .context import fingerprint


class RankTrace:
    """One row per query; answer records are [entity, rank, greater, tie_size]."""

    def __init__(self, path, identity, completed):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        try:
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)')
            self.connection.execute('CREATE TABLE IF NOT EXISTS queries (position INTEGER PRIMARY KEY, query_id TEXT UNIQUE, shape TEXT, answers TEXT)')
            rows = self.connection.execute('SELECT identity FROM metadata').fetchall()
            if rows and rows != [(fingerprint(identity),)]:
                raise ValueError('Rank trace identity changed')
            if not rows:
                self.connection.execute('INSERT INTO metadata VALUES (?)', (fingerprint(identity),))
            count, last = self.connection.execute('SELECT COUNT(*), MAX(position) FROM queries WHERE position < ?', (completed,)).fetchone()
            if count != completed or (completed and last != completed - 1):
                raise ValueError('Rank trace is missing checkpointed queries')
            self.connection.execute('DELETE FROM queries WHERE position >= ?', (completed,))
            self.connection.commit()
        except BaseException:
            self.connection.close()
            raise

    def add(self, position, query, answers):
        self.connection.execute('INSERT INTO queries VALUES (?, ?, ?, ?)',
                                (position, query.identity, query.shape, json.dumps(answers, separators=(',', ':'))))

    def commit(self):
        self.connection.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()
