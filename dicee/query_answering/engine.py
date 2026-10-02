"""Shared beam and exact query execution over complete atomic rows."""

import math
import weakref
from collections import OrderedDict
from contextlib import contextmanager

import torch

from ..models._inference import to_device
from ._deferred import deferred_checks, require
from ._query import atomic_conditions, compile_query, execute_query, positive, validate_tree
from .context import QueryContext, evaluation_mode, fingerprint, state_token
from .score_adapter import QueryScoreAdapter

# Public contexts derived from graph models, released with the model.
_MODEL_CONTEXTS = weakref.WeakKeyDictionary()
_UNSET = object()


def _model_context(model, token, tensors):
    """Share one derived context among all scorers of a model (e.g. adapter controls).

    The entry holds the graph tensors themselves, so a reused object ID cannot match.
    """
    shared = _MODEL_CONTEXTS.get(model)
    if shared is None or shared[0] != token or any(a is not b for a, b in zip(shared[1], tensors)):
        shared = _MODEL_CONTEXTS[model] = token, tensors, QueryContext.from_model(model)
    return shared[2]


class AtomicBatchCache:
    """Share bounded logits across adapters without changing batch geometry.

    ``model`` keeps scores on their producing device, avoiding host round trips
    when GPU memory is available. CPU storage remains the portable default.
    ``granularity='row'`` stores single rows and computes only missing rows, in
    batches of their own. Rows are then reused whatever batch requests them, so
    a row's last bits can depend on the batch that first computed it.
    """

    def __init__(self, max_bytes, *, device='cpu', granularity='batch'):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Atomic cache size must be a nonnegative integer')
        if device not in ('cpu', 'model'):
            raise ValueError('Atomic cache device must be cpu or model')
        if granularity not in ('batch', 'row'):
            raise ValueError('Atomic cache granularity must be batch or row')
        self.max_bytes = max_bytes
        self.device = device
        self.granularity = granularity
        self.rows, self.token, self.used = OrderedDict(), None, 0
        self.hits, self.computed, self.peak = 0, 0, 0

    def clear(self):
        self.rows.clear()
        self.used, self.token = 0, None

    def get(self, token, conditions):
        if token is None or token != self.token:
            self.clear()
            self.token = token
        key = tuple(conditions)
        if token is not None and key in self.rows:
            self.rows.move_to_end(key)
            self.hits += len(conditions)
            return self.rows[key]
        return None

    def put(self, conditions, values):
        self.computed += len(conditions)
        size = values.numel() * values.element_size()
        if self.token is None or size > self.max_bytes:
            return
        key = tuple(conditions)
        if key in self.rows:
            old = self.rows.pop(key)
            self.used -= old.numel() * old.element_size()
        while self.rows and self.used + size > self.max_bytes:
            _, old = self.rows.popitem(last=False)
            self.used -= old.numel() * old.element_size()
        target = values.device if self.device == 'model' else 'cpu'
        self.rows[key] = values.detach().to(target, copy=True)
        self.used += size
        self.peak = max(self.peak, self.used)

    def lookup(self, token, conditions):
        """Row granularity: cached rows by condition and the distinct missing conditions."""
        if token is None or token != self.token:
            self.clear()
            self.token = token
        found, missing = {}, []
        for key in dict.fromkeys(conditions):
            if token is not None and key in self.rows:
                self.rows.move_to_end(key)
                found[key] = self.rows[key]
            else:
                missing.append(key)
        self.hits += sum(key in found for key in conditions)
        return found, missing

    def put_rows(self, conditions, values):
        """Row granularity: store each row of a computed batch on its own."""
        self.computed += len(conditions)
        if self.token is None or not len(conditions):
            return
        size = values[0].numel() * values.element_size()
        if size > self.max_bytes:
            return
        stored = values.detach().to(values.device if self.device == 'model' else 'cpu')
        for key, row in zip(conditions, stored):
            if key in self.rows:
                self.used -= self.rows.pop(key).numel() * row.element_size()
            while self.rows and self.used + size > self.max_bytes:
                _, old = self.rows.popitem(last=False)
                self.used -= old.numel() * old.element_size()
            # A copy per row lets eviction release each row's memory.
            self.rows[key] = row.clone()
            self.used += size
        self.peak = max(self.peak, self.used)


class AtomicScorer:
    """Use DICE's existing all-tail API; only stochastic scoring needs special care."""

    def __init__(self, model, context=None, *, row_batch_size=8, seed=0, samples=None, raw_cache=None):
        from ..models.graph_model import GraphKGE, RelationGraphKGE
        if isinstance(model, RelationGraphKGE) or getattr(model, 'relation_prediction', False) or getattr(model, 'name', None) == 'Shallom':
            raise ValueError('Entity query answering requires an entity-prediction model')
        if type(row_batch_size) is not int or row_batch_size < 1 or (samples is not None and (type(samples) is not int or samples < 1)) or type(seed) is not int:
            raise ValueError('Positive row/sample counts and an integer seed are required')
        self.model, self.explicit_context = model, context
        self.graph_model = isinstance(model, GraphKGE)
        self.row_batch_size, self.seed, self.samples = row_batch_size, seed, samples
        self.raw_cache = raw_cache
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
                self._attached_context = (_model_context(self.model, token, tensors) if cacheable
                                          else QueryContext.from_model(self.model))
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
            self._evaluating, self._token_memo = True, _UNSET
            try:
                yield
            finally:
                self._evaluating, self._token_memo = False, _UNSET

    def cache_token(self):
        """Walking the model's tensors is costly; one evaluation context cannot change them, so it walks once."""
        if getattr(self, '_evaluating', False):
            if self._token_memo is _UNSET:
                self._token_memo = self._cache_token()
            return self._token_memo
        return self._cache_token()

    def _cache_token(self):
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
        conditions = [tuple(map(int, pair)) for pair in conditions]
        if not conditions:
            return torch.empty((0, self.n), device=self.device)
        if any(not (0 <= h < self.n and 0 <= r < self.nr) for h, r in conditions):
            raise ValueError('Atomic query outside the public vocabulary')
        token = self.cache_token() if self.raw_cache is not None else None
        with self.evaluation():
            if self.raw_cache is not None and self.raw_cache.granularity == 'row':
                return self._cached_rows(token, conditions)
            cached = self.raw_cache.get(token, conditions) if self.raw_cache is not None else None
            if cached is not None:
                return cached.to(self.device, copy=True)
            result = self._compute(conditions)
        if self.raw_cache is not None:
            self.raw_cache.put(conditions, result)
        return result.detach()

    def _cached_rows(self, token, conditions):
        """Row-granular reuse: compute only rows missing from the cache, then restore the requested order."""
        found, missing = self.raw_cache.lookup(token, conditions)
        fresh = {}
        if missing:
            computed = self._compute(missing)
            self.raw_cache.put_rows(missing, computed)
            fresh = {key: i for i, key in enumerate(missing)}
        reused = [i for i, key in enumerate(conditions) if key not in fresh]
        if not reused:
            return computed[[fresh[key] for key in conditions]].detach()
        # Stacking copies, so callers never alias cached storage.
        rows = torch.stack([found[conditions[i]] for i in reused]).to(self.device)
        if not fresh:
            return rows
        result = torch.empty((len(conditions), self.n), dtype=computed.dtype, device=self.device)
        result[reused] = rows
        new = [i for i, key in enumerate(conditions) if key in fresh]
        result[new] = computed[[fresh[conditions[i]] for i in new]]
        return result.detach()

    def _compute(self, conditions):
        from ..models.flock import Flock
        from ..models.graph_model import GraphKGE
        if isinstance(self.model, Flock):
            samples = self.model.test_samples if self.samples is None else self.samples
            seeds = [int(fingerprint([self.context.identity, self.seed, samples, h, r])[:15], 16) for h, r in conditions]
            result = self.model.forward_k_vs_all_seeded(torch.tensor(conditions, device=self.device), seeds, samples=samples)
        else:
            # Graph models validate and map host queries before an asynchronous copy.
            host = self.graph_model and type(self.model).forward_k_vs_sample is GraphKGE.forward_k_vs_sample
            batches = (torch.tensor(conditions[start:start + self.row_batch_size])
                       for start in range(0, len(conditions), self.row_batch_size))
            result = torch.cat([self.model.forward_k_vs_all(batch if host else to_device(batch, self.device))
                                for batch in batches])
        if result.shape != (len(conditions), self.n):
            raise ValueError('Atomic scorer must return finite complete [rows, entities] logits')
        require(torch.isfinite(result).all(), ValueError('Atomic scorer must return finite complete [rows, entities] logits'))
        return result


class QueryAnswerer:
    """Answer indexed query tuples with the same evaluator for KGEs and KGFMs.

    Graph models supply their attached context automatically. For a transductive
    model, pass QueryContext to use context features or observed memberships.
    ``predict`` returns memberships (or log-memberships with ``return_log_scores``).
    Evaluation-mode predictions share a bounded row cache, invalidated by graph,
    weight, adapter or inference-setting changes. Training callers cache only
    within a prediction. ``last_info`` reports reuse and whether a beam pruned.
    ``restore_observed=False`` disables final proof restoration while retaining
    the adapter's atomic observed-fact setting.
    """

    def __init__(self, model, *, context=None, adapter=None, observed_mix=None,
                 row_batch_size=8, cache_bytes=64 * 1024 * 1024, seed=0, samples=None, raw_cache=None,
                 restore_observed=True):
        if cache_bytes < 0:
            raise ValueError('Cache size must be nonnegative')
        if type(restore_observed) is not bool:
            raise ValueError('Observed-answer restoration must be a boolean')
        self.scorer = AtomicScorer(model, context, row_batch_size=row_batch_size, seed=seed, samples=samples, raw_cache=raw_cache)
        if adapter is not None and observed_mix is not None and observed_mix != adapter.observed_mix:
            raise ValueError('Set observed_mix on the adapter instead of overriding it')
        self.adapter = adapter if adapter is not None else QueryScoreAdapter('global', observed_mix or 0.)
        self.custom_adapter = adapter is not None
        self.restore_observed = restore_observed
        self.cache_bytes = cache_bytes
        self.last_info = {}
        self.clear_cache()

    def clear_cache(self):
        """Release rows, also required after custom non-tensor scoring changes."""
        self._cache, self._cache_used, self._cache_token = OrderedDict(), 0, None

    def _prepare_cache(self, use_logits):
        token, adapter_token = self.scorer.cache_token(), state_token(self.adapter)
        if token is None and self.scorer.raw_cache is not None:
            self.scorer.raw_cache.clear()
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
        with self.scorer.evaluation(), self._checked():
            for _ in self._condition_batches(missing, use_logits, stats):
                pass
        return stats

    @contextmanager
    def _checked(self):
        """Read device-side validity checks once; drop rows cached before a failure."""
        try:
            with deferred_checks():
                yield
        except BaseException:
            self.clear_cache()
            if self.scorer.raw_cache is not None:
                self.scorer.raw_cache.clear()
            raise

    @torch.no_grad()
    def predict(self, query, *, beam_size=64, tnorm='prod', neg_norm='standard', lambda_=0.,
                use_logits=False, return_log_scores=False, executor='cqd'):
        """QTO evaluates every potentially improving head without a beam limit."""
        if executor not in ('cqd', 'qto'):
            raise ValueError('Choose cqd or qto executor')
        if executor == 'qto' and use_logits:
            raise ValueError('QTO requires bounded memberships, not raw logits')
        if type(beam_size) is not int or beam_size < 1:
            raise ValueError('beam_size must be a positive integer')
        if tnorm not in ('prod', 'min') or neg_norm not in ('standard', 'sugeno', 'yager'):
            raise ValueError('Unknown conjunction or negation norm')
        if not math.isfinite(lambda_) or (neg_norm == 'sugeno' and lambda_ <= -1) or (neg_norm == 'yager' and lambda_ <= 0):
            raise ValueError('Sugeno requires lambda_ > -1; Yager requires lambda_ > 0')
        if use_logits and (self.custom_adapter or self.adapter.observed_mix or return_log_scores):
            raise ValueError('Raw logits cannot be combined with adapters, observed overrides, or log-membership output')
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

        # Skipped cache writes can change future miss batches; require scalar backend rows.
        skip_zero_prefix = (not self.scorer.model.training and
                            getattr(self.scorer.model, 'query_batch_size', self.scorer.row_batch_size) == 1)
        with self.scorer.evaluation(), self._checked():
            result, stats['pruned'] = execute_query(
                tree, lambda heads, relation: self._row_batches(heads, relation, use_logits, stats),
                n=n, device=self.scorer.device, row_batch_size=self.scorer.row_batch_size,
                beam_size=beam_size, tnorm=tnorm, neg_norm=neg_norm, lambda_=lambda_,
                use_logits=use_logits, executor=executor, stats=stats, skip_zero_prefix=skip_zero_prefix)
            if self.restore_observed and not use_logits and self.adapter.observed_mix == 1 and positive(tree):
                result = result.clone()
                answers = sorted(context.answers(tree))
                if answers:
                    result.index_fill_(0, to_device(torch.tensor(answers), result.device), 0.)
            require(~(torch.isnan(result) | torch.isposinf(result)).any(),
                    FloatingPointError('Query composition produced invalid scores'))
        self.last_info = dict(stats, search='global-prefix' if executor == 'cqd' else 'exact', cache_bytes=self._cache_used)
        return result if use_logits or return_log_scores else result.exp()
