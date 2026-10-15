"""Released +H CQD semantics with bounded intermediate score matrices."""

import torch

from ._common import QueryMethod, TensorCache, complex_rows, require_complex


class CQD(QueryMethod):
    def __init__(self, model, context, *, beam_size=64, tnorm='prod', hybrid=False, max_norm=None,
                 max_k=512, normalize=True, sigmoid=False, reference_quirks=True, row_batch_size=100, cache_bytes=64 * 2**20,
                 reference_batching=False, atomic_negation=False, final_batch_size=None):
        super().__init__(context)
        require_complex(model, context)
        if min(beam_size, row_batch_size) < 1 or (max_k is not None and max_k < 1):
            raise ValueError('Beam and batch sizes must be positive')
        if tnorm not in ('prod', 'min'):
            raise ValueError('Choose prod or min')
        self.model, self.beam_size, self.tnorm = model, beam_size, tnorm
        self.max_norm = (.9 if hybrid else 1.) if max_norm is None else max_norm
        if not 0 < self.max_norm <= 1 or hybrid != (self.max_norm != 1):
            raise ValueError('Hybrid requires max_norm < 1; plain CQD requires max_norm = 1')
        self.hybrid, self.max_k = hybrid, max_k
        self.normalize, self.sigmoid = normalize, sigmoid
        self.reference_quirks, self.row_batch_size = reference_quirks, row_batch_size
        self.reference_batching = reference_batching
        self.atomic_negation = atomic_negation
        if final_batch_size is not None and (type(final_batch_size) is not int or final_batch_size < 1):
            raise ValueError('Final projection batch size must be positive')
        self.final_batch_size = final_batch_size
        self.cache = TensorCache(cache_bytes)
        self._tails = ({h: {r: tuple(sorted(ts)) for r, ts in rels.items()} for h, rels in context.outgoing.items()}
                       if hybrid else {})

    def conjunction(self, left, right):
        return left * right if self.tnorm == 'prod' else torch.minimum(left, right)

    def _raw(self, heads, relation):
        self.cache.refresh(self.model)
        key = relation, tuple(heads.tolist())
        scores = self.cache.get(key)
        if scores is None:
            if heads.is_cuda:
                # Keep reference matrix geometry; fail before a predictable OOM.
                element = self.model.entity_embeddings.weight.element_size()
                dim = self.model.entity_embeddings.weight.shape[1]
                scratch = len(heads) * (3 * self.context.num_entities + 6 * dim) * element
                free, _ = torch.cuda.mem_get_info(heads.device)
                reusable = torch.cuda.memory_reserved(heads.device) - torch.cuda.memory_allocated(heads.device)
                if scratch + 256 * 2**20 > free + reusable:
                    raise MemoryError(
                        f'CQD reference stage with {len(heads)} heads needs approximately '
                        f'{scratch / 2**30:.2f} GiB scratch plus 0.25 GiB reserve; '
                        f'{(free + reusable) / 2**30:.2f} GiB is available. '
                        'Use more VRAM or explicitly re-verify a bounded batching configuration.')
            scores = complex_rows(self.model, heads, torch.full_like(heads, relation))
            self.cache.put(key, scores)
        return scores.sigmoid() * self.max_norm if self.sigmoid else scores

    def _rows(self, heads, relation, observed_relation=None, batch_size=None):
        # The release normalizes the entire stage, rather than each atomic row.
        low, high = None, None
        batch_size = batch_size or (len(heads) if self.reference_batching else self.row_batch_size)
        single = len(heads) <= batch_size
        raw = None
        if self.normalize:
            for chunk in heads.split(batch_size):
                raw = self._raw(chunk, relation)
                lo, hi = raw.min(), raw.max()
                low = lo if low is None else torch.minimum(low, lo)
                high = hi if high is None else torch.maximum(high, hi)
                if not single:
                    raw = None
            if high == low:
                raise ValueError('Released CQD min-max normalization is undefined for constant scores')
        for start in range(0, len(heads), batch_size):
            chunk = heads[start:start + batch_size]
            scores = raw if raw is not None else self._raw(chunk, relation)
            raw = None
            if self.normalize:
                scores = self.max_norm * (scores - low) / (high - low)
            if self.hybrid:
                if not self.normalize:
                    scores = scores.clone()
                rel = relation if observed_relation is None else observed_relation
                self._restore_facts(scores, chunk, rel)
            yield start, scores

    def _restore_facts(self, scores, heads, relation):
        rows, tails = [], []
        for row, head in enumerate(heads.tolist()):
            facts = self._tails.get(head, {}).get(relation, ())
            for start in range(0, len(facts), 65536):
                part = facts[start:start + 65536]
                if len(rows) + len(part) > 65536:
                    scores[rows, tails] = 1
                    rows, tails = [], []
                rows.extend([row] * len(part))
                tails.extend(part)
        if rows:
            scores[rows, tails] = 1

    def _width(self, observed):
        width = min(self.beam_size + (observed if self.hybrid else 0), self.context.num_entities)
        return min(width, self.max_k) if self.hybrid and self.max_k is not None else width

    def _select_atomic(self, heads, relation):
        widths = ([self._width(len(self._tails.get(h, {}).get(relation, ()))) for h in heads.tolist()]
                  if self.hybrid else [self._width(0)] * len(heads))
        width = max(widths)
        values, indices = [], []
        for start, rows in self._rows(heads, relation):
            if self.reference_batching and not self.hybrid:
                return rows.topk(width, dim=1)
            for offset, row in enumerate(rows):
                count = widths[start + offset]
                score, index = row.unsqueeze(0).topk(count, dim=1)
                score, index = score[0], index[0]
                # The upstream hybrid pads shorter branches with entity 0 / score 0.
                values.append(torch.nn.functional.pad(score, (0, width - count)))
                indices.append(torch.nn.functional.pad(index, (0, width - count)))
        return torch.stack(values), torch.stack(indices)

    def _last_projection(self, heads, relation, prefix, observed_relation=None, negate=False, batch_size=None):
        result = None
        for start, rows in self._rows(heads, relation, observed_relation, batch_size):
            if negate:
                rows = 1 - rows
            scores = self.conjunction(prefix[start:start + len(rows), None], rows).max(0).values
            result = scores if result is None else torch.maximum(result, scores)
        return result

    def _path(self, head, relations, negate_last=False):
        heads = torch.tensor([head], device=self.device)
        if len(relations) == 1:
            scores = next(self._rows(heads, relations[0]))[1][0]
            return 1 - scores if negate_last else scores
        scores = []
        for relation in relations[:-1]:
            values, tails = self._select_atomic(heads, relation)
            scores.append(values)
            heads = tails.flatten()
        last_relation = relations[-1]
        if self.reference_quirks and len(relations) == 4:
            prefix, last_relation = scores[-1].flatten(), relations[-2]
        else:
            prefix = scores[0].flatten()
            for index, values in enumerate(scores[1:], 1):
                expanded = (prefix.repeat(values.shape[1]) if self.reference_quirks and len(relations) == 3 and index == 1
                            else prefix.repeat_interleave(values.shape[1]))
                prefix = self.conjunction(expanded, values.flatten())
                del expanded
        del scores
        return self._last_projection(heads, last_relation, prefix, relations[-1], negate=negate_last,
                                     batch_size=self.final_batch_size if len(relations) == 4 else None)

    def score_tree(self, tree):
        op = tree[0]
        if op == 'not':
            if not self.atomic_negation:
                raise ValueError('CQD negated queries require atomic_negation=true')
            child, relations = tree[1], []
            while child[0] == 'project':
                relations.append(child[1])
                child = child[2]
            if child[0] != 'anchor' or not relations or len(relations) > 2:
                raise ValueError('Atomic negation supports anchored one- and two-edge paths')
            return self._path(child[1], list(reversed(relations)), negate_last=True)
        if op in ('and', 'or'):
            values = [self.score_tree(child) for child in tree[1:]]
            result = values[0]
            for value in values[1:]:
                result = (self.conjunction(result, value) if op == 'and' else
                          1 - (1 - result) * (1 - value) if self.tnorm == 'prod' else torch.maximum(result, value))
            return result
        if op != 'project':
            raise ValueError('CQD needs anchored relation projections')
        relations, child = [], tree
        while child[0] == 'project':
            relations.append(child[1])
            child = child[2]
        relations.reverse()
        if child[0] == 'anchor':
            return self._path(child[1], relations)
        if len(relations) != 1:
            raise ValueError('Only published CQD structures are supported')
        values = self.score_tree(child)
        count = self._width(int((values == 1).sum()) if self.hybrid else 0)
        if self.max_k is not None:
            count = min(count, self.max_k)
        prefix, heads = values.topk(count)
        return self._last_projection(heads, relations[0], prefix)
