"""QTO's calibrated neural adjacency and max-product query execution."""

import torch

from ._common import QueryMethod, TensorCache, complex_rows, require_complex


class QTO(QueryMethod):
    def __init__(self, model, context, *, threshold, negation_scale, row_batch_size=100, cache_bytes=512 * 2**20,
                 reference_batching=False):
        super().__init__(context)
        require_complex(model, context)
        if not 0 <= threshold <= 1 or negation_scale <= 0 or row_batch_size < 1:
            raise ValueError('Invalid QTO calibration or batch size')
        self.model, self.threshold, self.negation_scale = model, threshold, negation_scale
        self.row_batch_size = row_batch_size
        self.reference_batching = reference_batching
        self.cache = TensorCache(cache_bytes)

    def rows(self, heads, relation):
        self.cache.refresh(self.model)
        requested = heads.tolist()
        values = {head: self.cache.get((head, relation)) for head in requested}
        missing = [head for head, value in values.items() if value is None]
        groups = ([list(range(start, min(start + 100, self.context.num_entities))) for start in sorted({head // 100 * 100 for head in missing})]
                  if self.reference_batching else [missing] if missing else [])
        for group in groups:
            raw = complex_rows(self.model, group, [relation] * len(group))
            degrees = raw.new_tensor([max(1, len(self.context.outgoing.get(head, {}).get(relation, ()))) for head in group])
            calibrated = raw.softmax(-1) * degrees[:, None]
            calibrated = calibrated * (calibrated >= self.threshold)
            calibrated = torch.where(calibrated >= 1, .9999, calibrated)
            for head, row in zip(group, calibrated):
                row[sorted(self.context.outgoing.get(head, {}).get(relation, ()))] = 1
                if head in values:
                    values[head] = row.clone() if self.reference_batching else row
                self.cache.put((head, relation), row)
        # Keep requested rows ahead of opportunistic canonical-block prefetches.
        if groups:
            for head, row in values.items():
                if self.cache.get((head, relation)) is None:
                    self.cache.put((head, relation), row)
        return torch.stack([values[head] for head in requested])

    def projection(self, prefix, relation, negate=False):
        result = torch.zeros_like(prefix)
        heads = prefix.nonzero().flatten()
        if not len(heads):
            return result
        for chunk in heads.split(self.row_batch_size):
            rows = self.rows(chunk, relation)
            if negate:
                rows = 1 - torch.minimum(torch.ones_like(rows), self.negation_scale * rows)
            result = torch.maximum(result, (prefix[chunk, None] * rows).max(0).values)
        return result

    def score_tree(self, tree):
        op = tree[0]
        if op == 'anchor':
            return torch.nn.functional.one_hot(torch.tensor(tree[1], device=self.device), self.context.num_entities).float()
        if op == 'project':
            return self.projection(self.score_tree(tree[2]), tree[1])
        if op == 'not':
            child = tree[1]
            if child[0] != 'project':
                raise ValueError('Official QTO supports negation of relation atoms only')
            return self.projection(self.score_tree(child[2]), child[1], negate=True)
        values = torch.stack([self.score_tree(child) for child in tree[1:]])
        return values.prod(0) if op == 'and' else 1 - (1 - values).prod(0)
