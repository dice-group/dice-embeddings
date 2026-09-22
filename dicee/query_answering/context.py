"""Public-ID graph context; answers and evaluation labels never enter this object."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ._query import exact_answers


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def state_fingerprint(model_or_state):
    state = model_or_state.state_dict() if hasattr(model_or_state, 'state_dict') else model_or_state
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_token(model):
    """Cheap mutation token; inference tensors cannot safely be retained in caches.

    Use normal PyTorch parameter/buffer updates (copy_, load_state_dict, optimizers),
    not writes through .data, which bypass PyTorch's mutation counters.
    """
    tensors = (*model.named_parameters(), *model.named_buffers())
    if any(t.is_inference() for _, t in tensors):
        return None
    return (id(model), tuple((name, id(t), t._version, t.device, t.dtype, tuple(t.shape))
                            for name, t in tensors))


@dataclass(frozen=True)
class QueryContext:
    """A complete public vocabulary and a deduplicated set of observed triples.

    Explicit inverse pairs use public direct/inverse IDs, never an assumed ID parity.
    Supplied inverse pairs add reciprocal facts to the context.
    """

    triples: tuple
    num_entities: int
    num_relations: int
    inverse_relations: tuple = ()

    def __post_init__(self):
        if type(self.num_entities) is not int or type(self.num_relations) is not int or min(self.num_entities, self.num_relations) < 1:
            raise ValueError('Positive integer vocabulary sizes are required')
        pairs = dict(self.inverse_relations)
        ids = [r for pair in pairs.items() for r in pair]
        if len(set(ids)) != len(ids) or any(type(r) is not int or not 0 <= r < self.num_relations for r in ids):
            raise ValueError('Inverse pairs must be disjoint public relation IDs')
        triples = self.triples.tolist() if hasattr(self.triples, 'tolist') else self.triples
        edges = set()
        inverse = {a: b for h, t in pairs.items() for a, b in ((h, t), (t, h))}
        for triple in triples:
            if len(triple) != 3 or any(type(v) is not int for v in triple):
                raise ValueError('Context triples must contain three integer IDs')
            h, r, t = triple
            if not (0 <= h < self.num_entities and 0 <= t < self.num_entities and 0 <= r < self.num_relations):
                raise ValueError('Context triple outside the public vocabulary')
            edges.add((h, r, t))
            if r in inverse:
                edges.add((t, inverse[r], h))
        object.__setattr__(self, 'triples', tuple(sorted(edges)))
        object.__setattr__(self, 'inverse_relations', tuple(sorted(pairs.items())))
        out = defaultdict(lambda: defaultdict(set))
        for h, r, t in edges:
            out[h][r].add(t)
        object.__setattr__(self, 'outgoing', {h: dict(rs) for h, rs in out.items()})
        object.__setattr__(self, 'heads', Counter(h for h, _, _ in edges))
        object.__setattr__(self, 'relations', Counter(r for _, r, _ in edges))
        object.__setattr__(self, '_identity', fingerprint(self.to_dict()))

    @classmethod
    def from_model(cls, model):
        """Translate GraphKGE's internal reciprocal IDs back into its public IDs."""
        model._require_graph()
        mapping = model.relation_id_map.detach().cpu().tolist()
        public = {internal: external for external, internal in enumerate(mapping)}
        direct_count = model.num_direct_relations
        pairs = {external: public[internal + direct_count] for internal, external in public.items()
                 if internal < direct_count and internal + direct_count in public}
        triples = [(h, public[r], t) for h, r, t in model.graph_triples.detach().cpu().tolist() if r in public]
        return cls(triples, model.num_entities, model.num_relations, tuple(pairs.items()))

    @property
    def identity(self):
        return self._identity

    def to_dict(self):
        return dict(triples=self.triples, num_entities=self.num_entities, num_relations=self.num_relations,
                    inverse_relations=self.inverse_relations)

    def answers(self, tree):
        return exact_answers(tree, self.outgoing, self.num_entities)

    def features(self, conditions, *, device):
        observed = torch.zeros((len(conditions), self.num_entities), dtype=torch.bool, device=device)
        base = []
        n, nr = self.num_entities, self.num_relations
        for i, (h, r) in enumerate(conditions):
            tails = self.outgoing.get(h, {}).get(r, set())
            observed[i, sorted(tails)] = True
            base.append([1., math.log1p(len(tails)) / math.log1p(n),
                         math.log1p(self.heads[h]) / math.log1p(n * nr),
                         math.log1p(self.relations[r]) / math.log1p(n * n)])
        return observed, torch.tensor(base, dtype=torch.float64, device=device)


@contextmanager
def attached_context(model, context):
    """Attach an inference context temporarily and restore the caller's graph."""
    from ..models.graph_model import GraphKGE
    if not isinstance(model, GraphKGE):
        if (model.num_entities, model.num_relations) != (context.num_entities, context.num_relations):
            raise ValueError('Transductive source must use the fixed backbone vocabulary')
        yield
        return
    old = QueryContext.from_model(model) if model.graph_triples is not None else None
    buffers = dict(model._buffers)
    walk_graph = getattr(model, '_walk_graph', None)
    sizes = model.num_entities, model.num_relations, model.num_direct_relations
    try:
        model.set_graph(context.triples, num_entities=context.num_entities, num_relations=context.num_relations,
                        inverse_relations=dict(context.inverse_relations))
        yield
    finally:
        if old is not None:
            model.set_graph(old.triples, num_entities=old.num_entities, num_relations=old.num_relations,
                            inverse_relations=dict(old.inverse_relations))
        else:
            for name, value in buffers.items():
                setattr(model, name, value)
            if hasattr(model, '_walk_graph'):
                model._walk_graph = walk_graph
            model.num_entities, model.num_relations, model.num_direct_relations = sizes
            model.clear_inference_cache()


@contextmanager
def evaluation_mode(model):
    """Restore all module modes, including mixed train/eval configurations."""
    modes = [(module, module.training) for module in model.modules()]
    if not any(training for _, training in modes):
        yield
        return
    model.eval()
    try:
        yield
    finally:
        model.train(modes[0][1])
        for module, training in modes:
            module.training = training
