"""Bounded CPU cache and persistent frozen rows for multi-hop adapter training."""

import hashlib
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .context import attached_context, fingerprint, state_fingerprint
from .engine import AtomicScorer


class TrainingRows:
    def __init__(self, model, context, *, cache_dir, row_batch_size, seed, samples,
                 cache_bytes=128 * 1024**2, disk_bytes=8 * 1024**3):
        from ..models._inference import float32_precision_token
        from .benchmark import _implementation_fingerprint

        self.model, self.context = model, context
        self.row_batch_size, self.seed, self.samples = row_batch_size, seed, samples
        self.cache_bytes, self.cache_used, self.cache = cache_bytes, 0, OrderedDict()
        self.disk_limit = disk_bytes if cache_dir is not None else min(disk_bytes, cache_bytes)
        settings = dict(version=1, backbone=state_fingerprint(model), context=context.identity,
                        implementation=_implementation_fingerprint(), architecture=str(model),
                        args={k: str(v) for k, v in getattr(model, 'args', {}).items()},
                        row_batch_size=row_batch_size, seed=seed, samples=samples,
                        precision=float32_precision_token(), torch=str(torch.__version__),
                        deterministic=torch.are_deterministic_algorithms_enabled(),
                        autocast=torch.is_autocast_enabled(next(model.parameters()).device.type),
                        device=str(next(model.parameters()).device))
        self.identity = fingerprint(settings)
        path = ':memory:'
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            path = str(Path(cache_dir) / f'{self.identity}.sqlite3')
        self.connection = sqlite3.connect(path)
        self.connection.execute('PRAGMA cache_size=-8192')
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (identity TEXT PRIMARY KEY, settings TEXT)')
        existing = self.connection.execute('SELECT identity, settings FROM metadata').fetchall()
        expected = (self.identity, json.dumps(settings, sort_keys=True))
        if existing and existing != [expected]:
            raise ValueError('Training row cache identity mismatch')
        self.connection.execute('INSERT OR IGNORE INTO metadata VALUES (?, ?)', expected)
        self.connection.execute('CREATE TABLE IF NOT EXISTS rows (head INTEGER, relation INTEGER, dtype TEXT, value BLOB, sha256 TEXT, PRIMARY KEY(head, relation))')
        self.connection.commit()
        self.disk_used = self.connection.execute('SELECT COALESCE(SUM(LENGTH(value)), 0) FROM rows').fetchone()[0]

    def prepared(self, adapter, conditions):
        mode = (adapter.feature_mode, adapter.normalization)
        ready, missing = {}, []
        for pair in dict.fromkeys(conditions):
            key = (mode, pair)
            if key in self.cache:
                ready[pair] = self.cache[key]
                self.cache.move_to_end(key)
                continue
            record = self.connection.execute('SELECT dtype, value, sha256 FROM rows WHERE head=? AND relation=?', pair).fetchone()
            if record is None:
                missing.append(pair)
            else:
                dtype, value, digest = record
                if hashlib.sha256(value).hexdigest() != digest:
                    raise ValueError('Corrupt training score row')
                raw = torch.from_numpy(np.frombuffer(value, dtype=dtype).copy())
                self._prepare(adapter, pair, raw, mode, ready)
        if missing:
            with attached_context(self.model, self.context):
                scorer = AtomicScorer(self.model, self.context, row_batch_size=self.row_batch_size,
                                      seed=self.seed, samples=self.samples)
                raw = scorer.rows(missing).cpu()
            for pair, row in zip(missing, raw):
                array = row.numpy()
                value = array.tobytes()
                self.connection.execute('INSERT INTO rows VALUES (?, ?, ?, ?, ?)',
                                        (*pair, array.dtype.str, value, hashlib.sha256(value).hexdigest()))
                self.disk_used += len(value)
                self._prepare(adapter, pair, row, mode, ready)
            while self.disk_used > self.disk_limit:
                oldest = self.connection.execute('SELECT rowid, LENGTH(value) FROM rows ORDER BY rowid LIMIT 128').fetchall()
                self.connection.executemany('DELETE FROM rows WHERE rowid=?', [(row,) for row, _ in oldest])
                self.disk_used -= sum(size for _, size in oldest)
            self.connection.commit()
        return tuple(torch.stack([ready[pair][i] for pair in conditions]) for i in range(3))

    def _prepare(self, adapter, pair, raw, mode, ready):
        if raw.shape != (self.context.num_entities,) or not torch.isfinite(raw).all():
            raise ValueError('Invalid training score row')
        observed, base = self.context.features([pair], device='cpu')
        values = tuple(v[0].detach() for v in adapter.prepare(raw[None], observed, base))
        ready[pair] = values
        size = sum(v.numel() * v.element_size() for v in values)
        if size <= self.cache_bytes:
            while self.cache and self.cache_used + size > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cache_used -= sum(v.numel() * v.element_size() for v in old)
            self.cache[(mode, pair)] = values
            self.cache_used += size

    def close(self):
        self.connection.close()
