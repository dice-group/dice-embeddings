"""Shared inference contracts for native, checkpoint-compatible methods."""

from collections import OrderedDict
from functools import reduce
from typing import Any

import torch
from torch import nn

from ...models._inference import float32_precision_token
from .._query import compile_query, validate_tree
from ..context import evaluation_mode, state_token


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
    def __init__(self, context, logic='product'):
        super().__init__(context)
        fuzzy_combine([torch.tensor(0.)], 'and', logic)
        self.logic = logic

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
            return self.project_batch(self.membership_batch([tree[2] for tree in trees]), relations, relation_ids=relation_ids)
        if op == 'not':
            return 1 - self.membership_batch([tree[1] for tree in trees])
        return fuzzy_combine([self.membership_batch(children) for children in zip(*(tree[1:] for tree in trees))], op, self.logic)

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
