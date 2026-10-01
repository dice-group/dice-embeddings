"""Released incoming-relation heuristic with reproducible random rankings."""

import hashlib
import json

import torch
from torch import nn

from .._query import compile_query, validate_tree
from ._common import FuzzyGNN


def query_seed(query, seed):
    def flatten(value):
        return [i for part in value for i in flatten(part)] if isinstance(value, tuple) else [value]
    payload = json.dumps([seed, flatten(query)], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63)


class IncomingRelationHeuristic(FuzzyGNN):
    """Ignore projection heads; use Boolean logic and shuffle within two classes.

    Each query has its own seeded random stream, making subsets, batching and
    interrupted/resumed runs reproduce the same ranking.
    """

    def __init__(self, context, *, seed=0):
        super().__init__(context, logic='godel')
        if type(seed) is not int:
            raise ValueError('Heuristic seed must be an integer')
        self.seed = seed
        self.dummy_param = nn.Parameter(torch.zeros(1))
        triples = torch.tensor(context.triples, dtype=torch.long).reshape(-1, 3)
        self.register_buffer('relations', triples[:, 1], persistent=False)
        self.register_buffer('tails', triples[:, 2], persistent=False)

    def project_batch(self, membership, relation, *, relation_ids=None):
        result = membership.new_zeros(membership.shape)
        for i, r in enumerate(relation):
            result[i, self.tails[self.relations == r]] = 1
        return result

    @torch.no_grad()
    def predict_batch(self, queries):
        if not queries:
            return self.dummy_param.new_empty((0, self.context.num_entities))
        trees = [compile_query(q) for q in queries]
        for tree in trees:
            validate_tree(tree, self.context.num_entities, self.context.num_relations)
        probability = self.membership_batch(trees)
        noise = torch.stack([torch.rand(self.context.num_entities, device=self.device,
                            generator=torch.Generator(device=self.device).manual_seed(query_seed(q, self.seed)))
                             for q in queries])
        probability = (2 * probability + noise) / 3
        return ((probability + 1e-10) / (1 - probability + 1e-10)).log()

    def predict(self, query):
        return self.predict_batch([query])[0]
