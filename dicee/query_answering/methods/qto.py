"""QTO's calibrated neural adjacency and max-product query execution."""

from collections import OrderedDict

import torch

from ._common import QueryMethod, TensorCache, cache_token, complex_rows, require_complex

# Rows times entities densified at once when a projection reads a relation matrix.
DENSE_ELEMENTS = 2**24
# Projections from at most this many heads compute their rows unless the relation matrix is already on the device.
ROW_HEADS = 8


def _size(matrix):
    return sum(tensor.numel() * tensor.element_size() for tensor in matrix)


class RelationMatrices:
    """Thresholded calibrated relation matrices in CSR form, least recently used out first.

    Matrices stay on the model's device up to ``device_bytes``; evicted ones move to host memory, up to
    ``host_bytes``, and return on their next use. Parameter changes invalidate both, as in ``TensorCache``.
    """

    def __init__(self, device_bytes, host_bytes):
        if any(type(capacity) is not int or capacity < 0 for capacity in (device_bytes, host_bytes)):
            raise ValueError('Matrix cache capacities must be nonnegative byte counts')
        self.capacity = dict(device=device_bytes, host=host_bytes)
        self.tiers = dict(device=OrderedDict(), host=OrderedDict())
        self.bytes, self.token = dict(device=0, host=0), None

    def refresh(self, model):
        token = cache_token(model)
        if token is None or token != self.token:
            for tier in self.tiers.values():
                tier.clear()
            self.bytes, self.token = dict(device=0, host=0), token

    def resident(self, relation):
        """The matrix if it is on the device; host memory is not consulted."""
        if relation in self.tiers['device']:
            self.tiers['device'].move_to_end(relation)
            return self.tiers['device'][relation]
        return None

    def get(self, relation, device):
        matrix = self.resident(relation)
        if matrix is not None:
            return matrix
        matrix = self.tiers['host'].pop(relation, None)
        if matrix is None:
            return None
        self.bytes['host'] -= _size(matrix)
        matrix = tuple(tensor.to(device) for tensor in matrix)
        self.put(relation, matrix)
        return matrix

    def put(self, relation, matrix):
        self._store('device', relation, matrix)

    def _store(self, tier, relation, matrix):
        if self.token is None:
            return
        size = _size(matrix)
        if size > self.capacity[tier]:
            self._spill(tier, relation, matrix)
            return
        values = self.tiers[tier]
        while self.bytes[tier] + size > self.capacity[tier]:
            victim, old = values.popitem(last=False)
            self.bytes[tier] -= _size(old)
            self._spill(tier, victim, old)
        values[relation] = matrix
        self.bytes[tier] += size

    def _spill(self, tier, relation, matrix):
        if tier == 'device' and matrix[0].device.type != 'cpu':
            self._store('host', relation, tuple(tensor.cpu() for tensor in matrix))


class QTO(QueryMethod):
    """QTO with the released canonical 100-head calibration.

    With ``matrix_bytes`` > 0, projections read each relation's thresholded calibrated matrix, built once from
    the same canonical blocks and kept in CSR form (see ``RelationMatrices``), instead of recomputing rows per
    query. Scores are identical; only time and memory change.
    """

    def __init__(self, model, context, *, threshold, negation_scale, row_batch_size=100, cache_bytes=512 * 2**20,
                 reference_batching=False, matrix_bytes=0, host_matrix_bytes=0):
        super().__init__(context)
        require_complex(model, context)
        if not 0 <= threshold <= 1 or negation_scale <= 0 or row_batch_size < 1:
            raise ValueError('Invalid QTO calibration or batch size')
        if (matrix_bytes or host_matrix_bytes) and not (reference_batching and matrix_bytes):
            raise ValueError('Relation matrices need reference_batching and matrix_bytes > 0')
        self.model, self.threshold, self.negation_scale = model, threshold, negation_scale
        self.row_batch_size = row_batch_size
        self.reference_batching = reference_batching
        self.cache = TensorCache(cache_bytes)
        self.matrices = RelationMatrices(matrix_bytes, host_matrix_bytes) if matrix_bytes else None

    def calibrated(self, group, relation):
        """Calibrated rows of the heads in ``group``, scored together as one block."""
        raw = complex_rows(self.model, group, [relation] * len(group))
        observed = [self.context.outgoing.get(head, {}).get(relation, ()) for head in group]
        degrees = raw.new_tensor([max(1, len(tails)) for tails in observed])
        calibrated = raw.softmax(-1) * degrees[:, None]
        calibrated = calibrated * (calibrated >= self.threshold)
        calibrated = torch.where(calibrated >= 1, .9999, calibrated)
        rows = [i for i, tails in enumerate(observed) for _ in tails]
        if rows:
            calibrated[rows, [tail for tails in observed for tail in tails]] = 1
        return calibrated

    def rows(self, heads, relation):
        self.cache.refresh(self.model)
        requested = heads.tolist()
        values = {head: self.cache.get((head, relation)) for head in requested}
        missing = [head for head, value in values.items() if value is None]
        groups = ([list(range(start, min(start + 100, self.context.num_entities))) for start in sorted({head // 100 * 100 for head in missing})]
                  if self.reference_batching else [missing] if missing else [])
        for group in groups:
            for head, row in zip(group, self.calibrated(group, relation)):
                if head in values:
                    values[head] = row.clone() if self.reference_batching else row
                self.cache.put((head, relation), row)
        # Keep requested rows ahead of opportunistic canonical-block prefetches.
        if groups:
            for head, row in values.items():
                if self.cache.get((head, relation)) is None:
                    self.cache.put((head, relation), row)
        return torch.stack([values[head] for head in requested])

    def matrix(self, relation):
        """``relation``'s calibrated matrix as CSR (row pointers, int32 columns, values), from canonical blocks."""
        self.matrices.refresh(self.model)
        matrix = self.matrices.get(relation, self.device)
        if matrix is None:
            n = self.context.num_entities
            counts, columns, values = [], [], []
            for start in range(0, n, 100):
                block = self.calibrated(list(range(start, min(start + 100, n))), relation)
                rows, cols = block.nonzero(as_tuple=True)
                counts.append(torch.bincount(rows, minlength=len(block)))
                columns.append(cols.to(torch.int32))
                values.append(block[rows, cols])
            pointers = torch.cat([counts[0].new_zeros(1), torch.cat(counts).cumsum(0)])
            matrix = pointers, torch.cat(columns), torch.cat(values)
            self.matrices.put(relation, matrix)
        return matrix

    def matrix_rows(self, matrix, heads):
        """Dense calibrated rows of ``heads`` from a CSR matrix."""
        pointers, columns, values = matrix
        starts = pointers[heads]
        lengths = pointers[heads + 1] - starts
        rows = values.new_zeros(len(heads), self.context.num_entities)
        total = int(lengths.sum())
        if total:
            local = torch.repeat_interleave(torch.arange(len(heads), device=heads.device), lengths, output_size=total)
            offsets = torch.arange(total, device=heads.device) - (lengths.cumsum(0) - lengths)[local] + starts[local]
            rows[local, columns[offsets].long()] = values[offsets]
        return rows

    def projection(self, prefix, relation, negate=False):
        result = torch.zeros_like(prefix)
        heads = prefix.nonzero().flatten()
        if not len(heads):
            return result
        matrix = None
        if self.matrices is not None:
            # Both paths give the same rows; a few heads rarely justify loading or building a whole matrix.
            self.matrices.refresh(self.model)
            matrix = self.matrices.resident(relation)
            if matrix is None and len(heads) > ROW_HEADS:
                matrix = self.matrix(relation)
        if matrix is None:
            chunks = ((chunk, self.rows(chunk, relation)) for chunk in heads.split(self.row_batch_size))
        else:
            size = max(1, DENSE_ELEMENTS // self.context.num_entities)
            chunks = ((chunk, self.matrix_rows(matrix, chunk)) for chunk in heads.split(size))
        # The maximum over heads does not depend on how they are chunked.
        for chunk, rows in chunks:
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
