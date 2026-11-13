"""Shared inference contracts for native, checkpoint-compatible methods."""

from collections import OrderedDict
from functools import reduce
from typing import Any

import torch
from torch import nn

from ...models._inference import float32_precision_token
from .._query import compile_query, validate_tree
from ..context import evaluation_mode, state_token
from ..score_adapter import SOFTMAX_DEGREE_CEILING


class QueryMethod(nn.Module):
    provenance: dict[str, Any]  # Checkpoint identity, set by ``load_method``.

    def __init__(self, context):
        super().__init__()
        self.context = context

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(self, query):
        tree = compile_query(query)
        validate_tree(tree, self.context.num_entities, self.context.num_relations)
        with evaluation_mode(self):
            scores = self.score_tree(tree)
        if scores.shape != (self.context.num_entities,) or not torch.isfinite(scores).all():
            raise ValueError('QueryMethod returned invalid scores')
        return scores.clone()


def cache_token(model):
    """Validity token of cached inference tensors; None when nothing may be cached (inference tensors, autocast)."""
    token = state_token(model)
    if torch.is_autocast_enabled(next(model.parameters()).device.type):
        return None
    return None if token is None else (token, float32_precision_token())


class TensorCache:
    """Bounded inference cache, invalidated by parameter and device changes."""

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Cache capacity must be a nonnegative byte count')
        self.capacity, self.bytes, self.token = capacity, 0, None
        self.values = OrderedDict()
        self.optional = OrderedDict()

    def refresh(self, model):
        token = cache_token(model)
        if token is None or token != self.token:
            self.values.clear()
            self.optional.clear()
            self.bytes, self.token = 0, token

    def get(self, key):
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
            if key in self.optional:
                self.optional.move_to_end(key)
        return value

    def put(self, key, value, *, optional=False):
        """Optional entries use spare bytes and are evicted before primary entries."""
        size = value.numel() * value.element_size()
        if self.token is None or size > self.capacity:
            return
        previous = self.values.get(key)
        previous_size = 0 if previous is None else previous.numel() * previous.element_size()
        if optional and self.bytes - previous_size + size > self.capacity:
            return
        previous = self.values.pop(key, None)
        self.optional.pop(key, None)
        if previous is not None:
            self.bytes -= previous.numel() * previous.element_size()
        while self.bytes + size > self.capacity:
            if self.optional:
                victim, _ = self.optional.popitem(last=False)
                old = self.values.pop(victim)
            else:
                _, old = self.values.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.values[key] = value.detach().clone()
        self.bytes += size
        if optional:
            self.optional[key] = None


def fuzzy_combine(values, op, logic):
    if logic == 'product':
        fn = (lambda a, b: a * b) if op == 'and' else (lambda a, b: a + b - a * b)
    elif logic == 'godel':
        fn = torch.minimum if op == 'and' else torch.maximum
    elif logic == 'lukasiewicz':
        fn = (lambda a, b: (a + b - 1).clamp_min(0)) if op == 'and' else (lambda a, b: (a + b).clamp_max(1))
    else:
        raise ValueError('Choose product, godel or lukasiewicz logic')
    return reduce(fn, values)


class FuzzyGNN(QueryMethod):
    """Fuzzy-set execution with a learned projection.

    ``observed_traversal`` (off by default, and not part of any upstream method) also traverses the observed facts
    exactly at every projection: a tail receives at least the highest membership of a head linked to it by the
    projected relation, as with symbolic traversal (max over heads of a product with one).

    ``calibration='softmax-degree'`` (off by default, not part of any upstream method) replaces the sigmoid of every
    projection by QTO's calibration, generalized to fuzzy head sets: the softmax of the projection logits over all
    entities times d, the membership mass of the observed tails that the traversal reaches (at least one), capped at
    0.9999. For a single anchor head, d is the number of its observed tails, as in the beam executor's calibration.
    """

    def __init__(self, context, logic='product', observed_traversal=False, calibration=None):
        super().__init__(context)
        fuzzy_combine([torch.tensor(0.)], 'and', logic)
        if type(observed_traversal) is not bool:
            raise ValueError('observed_traversal must be a boolean')
        if calibration not in (None, 'softmax-degree'):
            raise ValueError('calibration must be None or softmax-degree')
        self.logic = logic
        self.observed_traversal = observed_traversal
        self.calibration = calibration
        self._relation_edges = None

    def membership(self, tree):
        return self.membership_batch([tree])[0]

    def score_tree(self, tree):
        probability = self.membership(tree)
        return ((probability + 1e-10) / (1 - probability + 1e-10)).log()

    def membership_batch(self, trees):
        op = trees[0][0]
        if any(tree[0] != op or len(tree) != len(trees[0]) for tree in trees):
            raise ValueError('GNN batches must share a query structure')
        if op == 'anchor':
            heads = torch.tensor([tree[1] for tree in trees], device=self.device)
            return torch.nn.functional.one_hot(heads, self.context.num_entities).float()
        if op == 'project':
            relation_ids = tuple(tree[1] for tree in trees)
            relations = torch.tensor(relation_ids, device=self.device)
            source = self.membership_batch([tree[2] for tree in trees])
            if self.calibration is None:
                projected = self.project_batch(source, relations, relation_ids=relation_ids)
                return torch.maximum(projected, self.traverse(source, relations)) if self.observed_traversal else projected
            known = self.traverse(source, relations)
            logits = self.project_batch(source, relations, relation_ids=relation_ids, logits=True)
            degree = known.sum(1, keepdim=True).clamp_min(1)
            projected = (logits.log_softmax(1) + degree.log()).exp().clamp_max(SOFTMAX_DEGREE_CEILING)
            return torch.maximum(projected, known) if self.observed_traversal else projected
        if op == 'not':
            return 1 - self.membership_batch([tree[1] for tree in trees])
        return fuzzy_combine([self.membership_batch(children) for children in zip(*(tree[1:] for tree in trees))], op, self.logic)

    def traverse(self, membership, relations):
        """Exact traversal of observed facts: each tail's highest membership among heads with (head, relation, tail)."""
        device = membership.device
        if self._relation_edges is None or self._relation_edges[0].device != device:
            triples = torch.tensor(self.context.triples, dtype=torch.long).reshape(-1, 3)
            triples = triples[torch.argsort(triples[:, 1], stable=True)]
            counts = torch.bincount(triples[:, 1], minlength=self.context.num_relations)
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            self._relation_edges = tuple(t.contiguous().to(device) for t in (offsets, triples[:, 0], triples[:, 2]))
        offsets, heads, tails = self._relation_edges
        starts = offsets[relations]
        lengths = offsets[relations + 1] - starts
        result = torch.zeros_like(membership)
        total = int(lengths.sum())
        if total:
            rows = torch.repeat_interleave(torch.arange(len(relations), device=device), lengths, output_size=total)
            positions = torch.arange(total, device=device) - (lengths.cumsum(0) - lengths)[rows] + starts[rows]
            values = membership[rows, heads[positions]]
            # A maximum does not depend on the reduction order, so this is deterministic on every device.
            result.view(-1).scatter_reduce_(0, rows * membership.shape[1] + tails[positions], values, 'amax')
        return result

    @torch.no_grad()
    def predict_batch(self, queries):
        trees = [compile_query(query) for query in queries]
        if not trees:
            return next(self.parameters()).new_empty((0, self.context.num_entities))
        for tree in trees:
            validate_tree(tree, self.context.num_entities, self.context.num_relations)
        with evaluation_mode(self):
            probability = self.membership_batch(trees)
            scores = ((probability + 1e-10) / (1 - probability + 1e-10)).log()
        if not torch.isfinite(scores).all():
            raise ValueError('GNN returned invalid scores')
        return scores


def union_branches(tree):
    """Distribute projection over DNF unions for embedding query models."""
    if tree[0] == 'or':
        return [branch for child in tree[1:] for branch in union_branches(child)]
    if tree[0] == 'project':
        return [('project', tree[1], child) for child in union_branches(tree[2])]
    if any(child[0] == 'or' for child in tree[1:] if isinstance(child, tuple)):
        raise ValueError('Only benchmark DNF unions are supported')
    return [tree]


def complex_rows(model, heads, relations):
    """ComplEx with the released CQD/QTO two-matmul evaluation order."""
    heads = torch.as_tensor(heads, device=model.entity_embeddings.weight.device, dtype=torch.long)
    relations = torch.as_tensor(relations, device=heads.device, dtype=torch.long)
    hr, hi = model.entity_embeddings(heads).chunk(2, -1)
    rr, ri = model.relation_embeddings(relations).chunk(2, -1)
    tr, ti = model.entity_embeddings.weight.chunk(2, -1)
    return (hr * rr - hi * ri) @ tr.T + (hi * rr + hr * ri) @ ti.T


def require_complex(model, context):
    from ...models.complex import ComplEx
    if not isinstance(model, ComplEx):
        raise TypeError('The released method requires a ComplEx backbone')
    if (model.num_entities, model.num_relations) != (context.num_entities, context.num_relations):
        raise ValueError('Checkpoint and graph vocabularies must match')
    if model.embedding_dim % 2:
        raise ValueError('ComplEx needs equal real and imaginary dimensions')
