"""Shared beam and exact query execution over complete atomic rows."""

import math
from collections import OrderedDict
from contextlib import contextmanager

import torch

from ._query import atomic_conditions, compile_query, execute_query, positive, validate_tree
from .adapter import QueryScoreAdapter
from .context import QueryContext, evaluation_mode, fingerprint, state_token


class AtomicScorer:
    """Use DICE's existing all-tail API; only stochastic scoring needs special care."""

    def __init__(self, model, context=None, *, row_batch_size=8, seed=0, samples=None):
        from ..models.graph_model import GraphKGE, RelationGraphKGE
        if isinstance(model, RelationGraphKGE) or getattr(model, 'relation_prediction', False) or getattr(model, 'name', None) == 'Shallom':
            raise ValueError('Entity query answering requires an entity-prediction model')
        if type(row_batch_size) is not int or row_batch_size < 1 or (samples is not None and (type(samples) is not int or samples < 1)) or type(seed) is not int:
            raise ValueError('Positive row/sample counts and an integer seed are required')
        self.model, self.explicit_context = model, context
        self.graph_model = isinstance(model, GraphKGE)
        self.row_batch_size, self.seed, self.samples = row_batch_size, seed, samples
        self.refresh()

    def refresh(self):
        self.n, self.nr = self.model.num_entities, self.model.num_relations
        if not isinstance(self.n, int) or not isinstance(self.nr, int) or min(self.n, self.nr) < 1:
            raise ValueError('Query answering requires a complete, positive entity/relation vocabulary')
        attached = None
        if self.graph_model:
            from ..models._fused_message import tensor_version
            self.model._require_graph()
            tensors = (self.model.graph_triples, self.model.relation_id_map)
            token = (self.n, self.nr, self.model.num_direct_relations,
                     *((id(t), tensor_version(t)) for t in tensors))
            cacheable = all(not t.is_inference() for t in tensors)
            if not cacheable or token != getattr(self, '_context_token', None):
                self._attached_context = QueryContext.from_model(self.model)
                self._context_token = token
            attached = self._attached_context
        if attached is not None and self.explicit_context is not None and attached.identity != self.explicit_context.identity:
            raise ValueError('Observed context must match the graph attached to the model')
        self.context = attached if self.explicit_context is None else self.explicit_context
        if self.context is not None and (self.context.num_entities, self.context.num_relations) != (self.n, self.nr):
            raise ValueError('Context and model vocabularies disagree')

    @property
    def device(self):
        return next(self.model.parameters(), torch.empty(0)).device

    @contextmanager
    def evaluation(self):
        """Enter evaluation mode once for a query, including all scoring batches."""
        if getattr(self, '_evaluating', False):
            yield
            return
        with evaluation_mode(self.model):
            self._evaluating = True
            try:
                yield
            finally:
                self._evaluating = False

    def cache_token(self):
        from ..models._inference import float32_precision_token
        from ..models.flock import Flock
        # Training callers can still reuse rows within a query, but never carry
        # them across predictions. Autocast and unversioned tensors also opt out.
        if any(m.training for m in self.model.modules()) or torch.is_autocast_enabled(self.device.type):
            return None
        token = self.model.inference_token() if self.graph_model else state_token(self.model)
        if token is None:
            return None
        sampling = None
        if isinstance(self.model, Flock):
            sampling = (self.seed, self.model.test_samples if self.samples is None else self.samples,
                        self.model.walk_num, self.model.walk_len, self.model.refinements,
                        self.model.compact_state, self.model.pack_walks, self.model.compile_sampler)
        return (id(self.model), token, self.n, self.nr, self.context.identity if self.context else None,
                self.row_batch_size, getattr(self.model, 'query_batch_size', None), sampling,
                float32_precision_token(), torch.are_deterministic_algorithms_enabled())

    @torch.no_grad()
    def rows(self, conditions):
        from ..models.flock import Flock
        conditions = [tuple(map(int, pair)) for pair in conditions]
        if not conditions:
            return torch.empty((0, self.n), device=self.device)
        if any(not (0 <= h < self.n and 0 <= r < self.nr) for h, r in conditions):
            raise ValueError('Atomic query outside the public vocabulary')
        with self.evaluation():
            if isinstance(self.model, Flock):
                samples = self.model.test_samples if self.samples is None else self.samples
                seeds = [int(fingerprint([self.context.identity, self.seed, samples, h, r])[:15], 16) for h, r in conditions]
                result = self.model.forward_k_vs_all_seeded(torch.tensor(conditions, device=self.device), seeds, samples=samples)
            else:
                result = torch.cat([self.model.forward_k_vs_all(torch.tensor(conditions[start:start + self.row_batch_size], device=self.device))
                                    for start in range(0, len(conditions), self.row_batch_size)])
        if result.shape != (len(conditions), self.n) or not torch.isfinite(result).all():
            raise ValueError('Atomic scorer must return finite complete [rows, entities] logits')
        return result.detach()


class QueryAnswerer:
    """Answer indexed query tuples with the same evaluator for KGEs and KGFMs.

    Graph models supply their attached context automatically. For a transductive
    model, pass QueryContext to use context features or observed memberships.
    ``predict`` returns memberships (or log-memberships with ``return_log_scores``).
    Evaluation-mode predictions share a bounded row cache, invalidated by graph,
    weight, adapter or inference-setting changes. Training callers cache only
    within a prediction. ``last_info`` reports reuse and whether a beam pruned.
    """

    def __init__(self, model, *, context=None, adapter=None, observed_mix=None,
                 row_batch_size=8, cache_bytes=64 * 1024 * 1024, seed=0, samples=None):
        if cache_bytes < 0:
            raise ValueError('Cache size must be nonnegative')
        self.scorer = AtomicScorer(model, context, row_batch_size=row_batch_size, seed=seed, samples=samples)
        if adapter is not None and observed_mix is not None and observed_mix != adapter.observed_mix:
            raise ValueError('Set observed_mix on the adapter instead of overriding it')
        self.adapter = adapter if adapter is not None else QueryScoreAdapter('global', observed_mix or 0.)
        self.custom_adapter = adapter is not None
        self.cache_bytes = cache_bytes
        self.last_info = {}
        self.clear_cache()

    def clear_cache(self):
        """Release rows, also required after custom non-tensor scoring changes."""
        self._cache, self._cache_used, self._cache_token = OrderedDict(), 0, None

    def _prepare_cache(self, use_logits):
        token, adapter_token = self.scorer.cache_token(), state_token(self.adapter)
        token = ((token, adapter_token, tuple(self.adapter.configuration.items()), use_logits)
                 if token is not None and adapter_token is not None else None)
        if token is None or token != self._cache_token:
            self.clear_cache()
        self._cache_token = token
        while self._cache and self._cache_used > self.cache_bytes:
            _, old = self._cache.popitem(last=False)
            self._cache_used -= old.numel() * old.element_size()

    def _condition_batches(self, conditions, use_logits, stats):
        cache = self._cache
        for start in range(0, len(conditions), self.scorer.row_batch_size):
            batch = conditions[start:start + self.scorer.row_batch_size]
            ready, missing = {}, []
            for key in dict.fromkeys(batch):
                if key in cache:
                    ready[key] = cache[key]
                    cache.move_to_end(key)
                    stats['cache_hits'] += 1
                else:
                    missing.append(key)
            if missing:
                raw = self.scorer.rows(missing).to(torch.float64)
                stats['raw_rows'] += len(missing)
                if use_logits:
                    transformed = raw
                else:
                    context = self.scorer.context
                    needs_context = self.adapter.feature_mode != 'global' or self.adapter.observed_mix
                    obs, base = context.features(missing, device=raw.device) if context and needs_context else (None, None)
                    transformed = self.adapter(raw, obs, base)
                for key, value in zip(missing, transformed):
                    # Copies keep evicted batches from being retained by one row.
                    value = value.clone()
                    ready[key] = value
                    size = value.numel() * value.element_size()
                    if size <= self.cache_bytes:
                        while self._cache_used + size > self.cache_bytes:
                            _, old = cache.popitem(last=False)
                            self._cache_used -= old.numel() * old.element_size()
                        cache[key] = value
                        self._cache_used += size
            # Stacking also isolates returned scores from mutable cache storage.
            yield batch, torch.stack([ready[key] for key in batch])

    def _row_batches(self, heads, relation, use_logits, stats):
        for batch, values in self._condition_batches([(head, relation) for head in heads], use_logits, stats):
            yield [head for head, _ in batch], values

    @torch.no_grad()
    def prefetch(self, queries, *, use_logits=False):
        """Warm reusable anchor rows without inspecting answer labels."""
        self.scorer.refresh()
        self.adapter.verify_model(self.scorer.model)
        self._prepare_cache(use_logits)
        stats = dict(raw_rows=0, cache_hits=0)
        if self._cache_token is None or self.cache_bytes < self.scorer.n * 8:
            return stats
        conditions = []
        for query in queries:
            tree = compile_query(query)
            validate_tree(tree, self.scorer.n, self.scorer.nr)
            conditions.extend(atomic_conditions(tree))
        missing = [key for key in dict.fromkeys(conditions) if key not in self._cache]
        missing = missing[:self.cache_bytes // (self.scorer.n * 8)]
        with self.scorer.evaluation():
            for _ in self._condition_batches(missing, use_logits, stats):
                pass
        return stats

    @torch.no_grad()
    def predict(self, query, *, beam_size=64, tnorm='prod', neg_norm='standard', lambda_=0.,
                use_logits=False, return_log_scores=False, executor='cqd'):
        """QTO evaluates every potentially improving head without a beam limit."""
        if executor not in ('cqd', 'qto'):
            raise ValueError('Choose cqd or qto executor')
        if executor == 'qto' and use_logits:
            raise ValueError('QTO requires bounded memberships, not legacy logits')
        if type(beam_size) is not int or beam_size < 1:
            raise ValueError('beam_size must be a positive integer')
        if tnorm not in ('prod', 'min') or neg_norm not in ('standard', 'sugeno', 'yager'):
            raise ValueError('Unknown conjunction or negation norm')
        if not math.isfinite(lambda_) or (neg_norm == 'sugeno' and lambda_ <= -1) or (neg_norm == 'yager' and lambda_ <= 0):
            raise ValueError('Sugeno requires lambda_ > -1; Yager requires lambda_ > 0')
        if use_logits and (self.custom_adapter or self.adapter.observed_mix or return_log_scores):
            raise ValueError('Legacy logits cannot be combined with adapters, observed overrides, or log-membership output')
        self.scorer.refresh()
        self.adapter.verify_model(self.scorer.model)
        n, nr = self.scorer.n, self.scorer.nr
        context = self.scorer.context
        if context is None and (self.adapter.feature_mode != 'global' or self.adapter.observed_mix):
            raise ValueError('Adapter features and observed overrides require a context graph')
        tree = compile_query(query)
        validate_tree(tree, n, nr)
        self._prepare_cache(use_logits)
        stats = dict(raw_rows=0, cache_hits=0, pruned=False, negated_pruning=False, bound_skipped=0)

        with self.scorer.evaluation():
            result, stats['pruned'] = execute_query(
                tree, lambda heads, relation: self._row_batches(heads, relation, use_logits, stats),
                n=n, device=self.scorer.device, row_batch_size=self.scorer.row_batch_size,
                beam_size=beam_size, tnorm=tnorm, neg_norm=neg_norm, lambda_=lambda_,
                use_logits=use_logits, executor=executor, stats=stats)
        if not use_logits and self.adapter.observed_mix == 1 and positive(tree):
            result = result.clone()
            result[sorted(context.answers(tree))] = 0.
        if (torch.isnan(result) | torch.isposinf(result)).any():
            raise FloatingPointError('Query composition produced invalid scores')
        self.last_info = dict(stats, search='global-prefix' if executor == 'cqd' else 'exact', cache_bytes=self._cache_used)
        return result if use_logits or return_log_scores else result.exp()
