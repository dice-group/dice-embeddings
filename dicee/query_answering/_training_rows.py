"""Bounded host/device caches for frozen scores and parameter-independent features."""

import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .context import attached_context, fingerprint, state_fingerprint
from .engine import AtomicScorer


def scoring_fingerprint(root=None):
    root = Path(root) if root is not None else Path(__file__).parents[1]
    paths = sorted((root / 'models').glob('*.py'))
    paths += [root / 'query_answering' / name for name in ('context.py', 'engine.py')]
    return fingerprint({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class TrainingRows:
    def __init__(self, model, context, *, cache_dir, row_batch_size, seed, samples,
                 cache_bytes=128 * 1024**2, disk_bytes=8 * 1024**3,
                 device='cpu', device_cache_bytes=0):
        from ..models._inference import float32_precision_token

        self.model, self.context = model, context
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if min(cache_bytes, disk_bytes, device_cache_bytes) < 0:
            raise ValueError('Cache budgets must be nonnegative')
        self.row_batch_size, self.seed, self.samples = row_batch_size, seed, samples
        self.cache_bytes, self.cache_used, self.cache = cache_bytes, 0, OrderedDict()
        self.device_cache_bytes, self.device_cache_used, self.device_cache = device_cache_bytes, 0, OrderedDict()
        self.device_rows, self.next_block = {}, 0
        self.feature_cache = OrderedDict()
        self.stats = dict(host_hits=0, device_hits=0, disk_rows=0, scored_rows=0, feature_preparations=0, seconds=0.)
        self.disk_limit = disk_bytes if cache_dir is not None else min(disk_bytes, cache_bytes)
        settings = dict(version=2, backbone=state_fingerprint(model), context=context.identity,
                        scoring_code=scoring_fingerprint(), architecture=str(model),
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
        self.connection.execute('CREATE TABLE IF NOT EXISTS features (mode TEXT, head INTEGER, relation INTEGER, value BLOB, sha256 TEXT, PRIMARY KEY(mode, head, relation))')
        self.feature_code = fingerprint({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                         for name in ('adapter.py', '_training_rows.py')})
        self.connection.commit()
        self.disk_used = self.connection.execute('SELECT COALESCE(SUM(LENGTH(value)), 0) FROM rows').fetchone()[0]

    def prepared(self, adapter, conditions, *, device=None):
        start = time.perf_counter()
        target = self.device if device is None else torch.device(device)
        if target.type == 'cuda' and target.index is None:
            target = torch.device('cuda', torch.cuda.current_device())
        if target.type != 'cpu' and target != self.device:
            raise ValueError('Prepared rows require CPU or the configured cache device')
        mode = json.dumps((self.feature_code, adapter.feature_mode, adapter.normalization))
        ready, pending = {}, []
        for pair in dict.fromkeys(conditions):
            key = (mode, pair)
            if target.type != 'cpu' and key in self.device_rows:
                block, index = self.device_rows[key]
                ready[pair] = tuple(v[index] for v in self.device_cache[block][1])
                self.device_cache.move_to_end(block)
                self.stats['device_hits'] += 1
            elif key in self.cache:
                ready[pair] = self.cache[key]
                self.cache.move_to_end(key)
                self.stats['host_hits'] += 1
            else:
                pending.append(pair)
        records, features = {}, {}
        for offset in range(0, len(pending), 256):
            batch = pending[offset:offset + 256]
            where = ','.join('(?,?)' for _ in batch)
            args = [v for pair in batch for v in pair]
            records.update({(h, r): (dtype, value, digest) for h, r, dtype, value, digest in self.connection.execute(
                f'WITH wanted(head,relation) AS (VALUES {where}) '
                'SELECT r.head,r.relation,r.dtype,r.value,r.sha256 FROM wanted w '
                'JOIN rows r ON r.head=w.head AND r.relation=w.relation', args)})
            features.update({(h, r): (value, digest) for h, r, value, digest in self.connection.execute(
                f'WITH wanted(head,relation) AS (VALUES {where}) '
                'SELECT f.head,f.relation,f.value,f.sha256 FROM wanted w '
                'JOIN features f ON f.head=w.head AND f.relation=w.relation WHERE f.mode=?', [*args, mode])})
        missing = []
        for pair in pending:
            if pair in records:
                dtype, value, digest = records[pair]
                if hashlib.sha256(value).hexdigest() != digest:
                    raise ValueError('Corrupt training score row')
                raw = torch.from_numpy(np.frombuffer(value, dtype=dtype).copy())
                self.stats['disk_rows'] += 1
                self._prepare(adapter, pair, raw, mode, ready, features.get(pair))
            else:
                missing.append(pair)
        if missing:
            # Leave transient backbone activations room on a shared GPU.
            self.device_cache.clear()
            self.device_rows.clear()
            self.device_cache_used = 0
            with attached_context(self.model, self.context):
                scorer = AtomicScorer(self.model, self.context, row_batch_size=self.row_batch_size,
                                      seed=self.seed, samples=self.samples)
                raw = scorer.rows(missing).cpu()
            self.stats['scored_rows'] += len(missing)
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
        if self.connection.in_transaction:
            self.connection.commit()
        if target.type != 'cpu':
            pairs = [pair for pair, values in ready.items() if values[0].device != target]
            if pairs:
                block = tuple(torch.stack([ready[pair][i] for pair in pairs]).to(target) for i in range(3))
                self._remember_device([(mode, pair) for pair in pairs], block)
                for index, pair in enumerate(pairs):
                    ready[pair] = tuple(v[index] for v in block)
        self.stats['seconds'] += time.perf_counter() - start
        values = tuple(torch.stack([ready[pair][i] for pair in conditions]) for i in range(3))
        return values[0].to(torch.float64), values[1], values[2]

    def _prepare(self, adapter, pair, raw, mode, ready, saved=None):
        if raw.shape != (self.context.num_entities,) or not torch.isfinite(raw).all():
            raise ValueError('Invalid training score row')
        stored = raw.clone() if raw._base is not None else raw
        raw = raw.to(torch.float64)
        key = (mode, pair)
        fixed = self.feature_cache.get(key)
        if fixed is None and saved is not None:
            value, digest = saved
            if hashlib.sha256(value).hexdigest() != digest:
                raise ValueError('Corrupt training features')
            fixed = torch.from_numpy(np.frombuffer(value, dtype=np.float64).copy())
        if fixed is None:
            observed, base = self.context.features([pair], device='cpu')
            _, _, features = adapter.prepare(raw[None], observed, base)
            mean, std = raw.new_zeros(1), raw.new_ones(1)
            if adapter.normalization == 'standard':
                mean = raw[None].mean(1)
                std = raw[None].std(1, correction=0).clamp_min(1e-6)
            fixed = torch.cat((features[0], mean, std)).detach()
            value = fixed.numpy().tobytes()
            self.connection.execute('INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?, ?)',
                                    (mode, *pair, value, hashlib.sha256(value).hexdigest()))
            self.stats['feature_preparations'] += 1
        self.feature_cache[key] = fixed
        self.feature_cache.move_to_end(key)
        while len(self.feature_cache) > 4096:
            self.feature_cache.popitem(last=False)
        observed = torch.zeros(self.context.num_entities, dtype=torch.bool)
        observed[sorted(self.context.outgoing.get(pair[0], {}).get(pair[1], set()))] = True
        if adapter.normalization == 'standard':
            raw = (raw - fixed[-2]) / fixed[-1]
        else:
            raw = stored
        values = raw, observed, fixed[:-2]
        ready[pair] = values
        self._remember('', key, values)

    def _remember(self, prefix, key, values):
        cache = getattr(self, prefix + 'cache')
        used, limit = getattr(self, prefix + 'cache_used'), getattr(self, prefix + 'cache_bytes')
        size = sum(v.numel() * v.element_size() for v in values)
        if size <= limit:
            old = cache.pop(key, ())
            used -= sum(v.numel() * v.element_size() for v in old)
            while cache and used + size > limit:
                _, old = cache.popitem(last=False)
                used -= sum(v.numel() * v.element_size() for v in old)
            cache[key] = values
            setattr(self, prefix + 'cache_used', used + size)

    def _remember_device(self, keys, values):
        # Charge whole allocations: evicted row views must not hide live blocks.
        size = sum(v.numel() * v.element_size() for v in values)
        if size > self.device_cache_bytes:
            return
        while self.device_cache and self.device_cache_used + size > self.device_cache_bytes:
            block, (old_keys, old) = self.device_cache.popitem(last=False)
            self.device_cache_used -= sum(v.numel() * v.element_size() for v in old)
            for key in old_keys:
                if self.device_rows.get(key, (None,))[0] == block:
                    del self.device_rows[key]
        block = self.next_block
        self.next_block += 1
        self.device_cache[block] = (keys, values)
        self.device_cache_used += size
        for index, key in enumerate(keys):
            self.device_rows[key] = block, index

    def close(self):
        self.connection.close()
        self.cache.clear()
        self.device_cache.clear()
        self.device_rows.clear()
        self.feature_cache.clear()
