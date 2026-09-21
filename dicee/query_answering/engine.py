"""Model-independent global-prefix query answering over complete atomic rows."""

from collections import OrderedDict

import torch

from ._query import combine, compile_query, negate, positive, validate_tree
from .adapter import QueryScoreAdapter
from .context import QueryContext, fingerprint


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
        if attached is not None and self.explicit_context is not None and attached != self.explicit_context:
            raise ValueError('Observed context must match the graph attached to the model')
        self.context = attached if self.explicit_context is None else self.explicit_context
        if self.context is not None and (self.context.num_entities, self.context.num_relations) != (self.n, self.nr):
            raise ValueError('Context and model vocabularies disagree')

    @property
    def device(self):
        return next(self.model.parameters(), torch.empty(0)).device

    @torch.no_grad()
    def rows(self, conditions):
        from ..models.flock import Flock
        conditions = [tuple(map(int, pair)) for pair in conditions]
        if not conditions:
            return torch.empty((0, self.n), device=self.device)
        if any(not (0 <= h < self.n and 0 <= r < self.nr) for h, r in conditions):
            raise ValueError('Atomic query outside the public vocabulary')
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            if isinstance(self.model, Flock):
                old_seed, old_samples = self.model.seed, self.model.test_samples
                try:
                    self.model.test_samples = old_samples if self.samples is None else self.samples
                    values = []
                    for h, r in conditions:
                        # Flock's local CPU generator leaves caller CPU/CUDA RNGs alone.
                        self.model.seed = int(fingerprint([self.context.identity, self.seed, self.model.test_samples, h, r])[:15], 16)
                        values.append(self.model.forward_k_vs_all(torch.tensor([[h, r]], device=self.device)))
                    result = torch.cat(values)
                finally:
                    self.model.seed, self.model.test_samples = old_seed, old_samples
            else:
                result = torch.cat([self.model.forward_k_vs_all(torch.tensor(conditions[start:start + self.row_batch_size], device=self.device))
                                    for start in range(0, len(conditions), self.row_batch_size)])
        finally:
            self.model.train(modes[0][1])
            for module, training in modes:
                module.training = training
        if result.shape != (len(conditions), self.n) or not torch.isfinite(result).all():
            raise ValueError('Atomic scorer must return finite complete [rows, entities] logits')
        return result.detach()


class QueryAnswerer:
    """Answer indexed query tuples with the same evaluator for KGEs and KGFMs.

    Graph models supply their attached context automatically. For a transductive
    model, pass QueryContext to use context features or observed memberships.
    ``predict`` returns memberships (or log-memberships with ``return_log_scores``).
    The cache is bounded and scoped to one prediction, so graph/weight changes
    cannot silently reuse old rows. ``last_info`` reports whether a beam pruned.
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

    @torch.no_grad()
    def predict(self, query, *, beam_size=64, tnorm='prod', neg_norm='standard', lambda_=0.,
                use_logits=False, return_log_scores=False):
        if type(beam_size) is not int or beam_size < 1:
            raise ValueError('beam_size must be a positive integer')
        if tnorm not in ('prod', 'min') or neg_norm not in ('standard', 'sugeno', 'yager'):
            raise ValueError('Unknown conjunction or negation norm')
        if not torch.isfinite(torch.tensor(lambda_)) or (neg_norm == 'sugeno' and lambda_ <= -1) or (neg_norm == 'yager' and lambda_ <= 0):
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
        cache, used = OrderedDict(), 0
        stats = dict(raw_rows=0, cache_hits=0, pruned=False, negated_pruning=False)

        def rows(heads, relation):
            nonlocal used
            batch_size = self.scorer.row_batch_size
            for start in range(0, len(heads), batch_size):
                batch = heads[start:start + batch_size]
                ready, missing = {}, []
                for head in batch:
                    key = (head, relation)
                    if key in cache:
                        ready[head] = cache[key]
                        cache.move_to_end(key)
                        stats['cache_hits'] += 1
                    else:
                        missing.append(head)
                if missing:
                    conditions = [(head, relation) for head in missing]
                    raw = self.scorer.rows(conditions).to(torch.float64)
                    stats['raw_rows'] += len(missing)
                    obs, base = context.features(conditions, device=raw.device) if context else (None, None)
                    transformed = raw if use_logits else self.adapter(raw, obs, base)
                    for head, value in zip(missing, transformed):
                        # Copy each row: cached views must not retain a whole batch.
                        value = value.clone()
                        ready[head] = value
                        size = value.numel() * value.element_size()
                        if size <= self.cache_bytes:
                            while used + size > self.cache_bytes:
                                _, old = cache.popitem(last=False)
                                used -= old.numel() * old.element_size()
                            cache[head, relation] = value
                            used += size
                for head in batch:
                    yield ready[head]

        def execute(node):
            op = node[0]
            if op == 'anchor':
                result = torch.full((n,), 0. if use_logits else -torch.inf, dtype=torch.float64, device=self.scorer.device)
                result[node[1]] = 1. if use_logits else 0.
                return result, False
            if op == 'project' and node[2][0] == 'anchor':
                return next(rows([node[2][1]], node[1])), False
            if op in ('and', 'or'):
                branches = [execute(child) for child in node[1:]]
                return combine([v for v, _ in branches], op, tnorm, logits=use_logits), any(p for _, p in branches)
            if op == 'not':
                values, pruned = execute(node[1])
                stats['negated_pruning'] |= pruned
                return negate(values, neg_norm, lambda_, logits=use_logits), pruned
            prefix, pruned = execute(node[2])
            heads = torch.argsort(prefix, descending=True, stable=True)[:min(beam_size, n)].tolist()
            result = torch.full_like(prefix, -torch.inf)
            for head, edge in zip(heads, rows(heads, node[1])):
                joined = combine([prefix[head].expand_as(edge), edge], 'and', tnorm, logits=use_logits)
                result = torch.maximum(result, joined)
            return result, pruned or beam_size < n

        result, stats['pruned'] = execute(tree)
        if not use_logits and self.adapter.observed_mix == 1 and positive(tree):
            result = result.clone()
            result[sorted(context.answers(tree))] = 0.
        if torch.isnan(result).any() or torch.isposinf(result).any():
            raise FloatingPointError('Query composition produced invalid scores')
        self.last_info = dict(stats, search='global-prefix', cache_bytes=used)
        return result if use_logits or return_log_scores else result.exp()
