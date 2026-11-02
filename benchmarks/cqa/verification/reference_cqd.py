"""Bound the released 4p oracle's final projection; no DICE imports."""

import torch


def query_4p(model, source, queries, batch_size):
    if len(queries) != 1:
        raise ValueError('The released hybrid executor requires single-query batches')
    entity, predicate = model.embeddings

    def raw(lhs, rel, rhs):
        result = model.score_o(lhs, rel, rhs)[0]
        return result.sigmoid() * model.max_norm if model.do_sigmoid else result

    def scoring(lhs, rel, rhs):
        result = raw(lhs, rel, rhs)
        if model.do_normalize:
            result = model.max_norm * (result - result.min()) / (result.max() - result.min())
        return result

    ids = queries[:, 0]
    # Each stage gathers by ID, so selected embedding tensors are unnecessary.
    for position in (1, 2, 3):
        relation = queries[:, position]
        atoms = torch.stack((ids, relation.expand_as(ids)), -1)
        prefix, _, ids = source.score_candidates(
            queries=atoms, filters=model.filters, s_emb=entity(ids), p_emb=predicate(relation),
            candidates_emb=entity.weight, k=model.k, max_norm=model.max_norm, max_k=model.max_k,
            entity_embeddings=lambda indices: None, scoring_function=scoring)
        ids = ids.flatten().long()

    # Preserve both release quirks: r3 embeddings for atom 4, and only atom 3's prefix.
    relation = predicate(queries[:, 3])
    prefix = prefix.flatten()
    low, high = None, None
    if model.do_normalize:
        for heads in ids.split(batch_size):
            values = raw(entity(heads), relation.expand(len(heads), -1), entity.weight)
            lo, hi = values.min(), values.max()
            low = lo if low is None else torch.minimum(low, lo)
            high = hi if high is None else torch.maximum(high, hi)
    result = None
    for offset in range(0, len(ids), batch_size):
        heads = ids[offset:offset + batch_size]
        values = raw(entity(heads), relation.expand(len(heads), -1), entity.weight)
        if model.do_normalize:
            values = model.max_norm * (values - low) / (high - low)
        if model.max_norm != 1:
            for row, head in enumerate(heads.tolist()):
                values[row, model.filters.get((head, int(queries[0, 4])), [])] = 1
        before = prefix[offset:offset + len(heads), None]
        combined = before * values if model.t_norm_name == 'prod' else torch.minimum(before, values)
        best = combined.max(0).values
        result = best if result is None else torch.maximum(result, best)
    return result[None]


class _BoundedStages:
    """Partition score_o calls while delegating fact overrides/top-k to upstream.

    The profile fixes both GEMM row groups and row-wise candidate selection.
    Extrema cover an entire logical atom stage, before any observed-fact override.
    Only selected IDs/scores survive between stages, never selected embeddings.
    """

    def __init__(self, model, source, batch_size):
        self.model, self.source, self.batch_size = model, source, batch_size
        self.entity, self.predicate = model.embeddings
        self.combine = torch.mul if model.t_norm_name == 'prod' else torch.minimum

    def raw(self, heads, relation):
        values = self.model.score_o(self.entity(heads), self.predicate(relation).expand(len(heads), -1), self.entity.weight)[0]
        return values.sigmoid() * self.model.max_norm if self.model.do_sigmoid else values

    def rows(self, heads, relation):
        low, high, retained = None, None, None
        if self.model.do_normalize:
            for group in heads.split(self.batch_size):
                values = self.raw(group, relation)
                low = values.min() if low is None else torch.minimum(low, values.min())
                high = values.max() if high is None else torch.maximum(high, values.max())
                if len(heads) <= self.batch_size:
                    retained = values
                del values
            if high == low:
                raise ValueError('Upstream min-max normalization is undefined for constant scores')
        for offset in range(0, len(heads), self.batch_size):
            group = heads[offset:offset + self.batch_size]
            values = retained if retained is not None else self.raw(group, relation)
            retained = None
            if self.model.do_normalize:
                values = self.model.max_norm * (values - low) / (high - low)
            yield offset, group, values

    def candidates(self, heads, relation, values, k):
        # score_candidates remains responsible for observed-fact replacement,
        # per-row beam width, and the upstream topk implementation.
        return self.source.score_candidates(
            queries=torch.stack((heads, relation.expand_as(heads)), -1),
            filters=self.model.filters, s_emb=self.entity(heads), p_emb=self.predicate(relation),
            candidates_emb=self.entity.weight, k=k, max_norm=self.model.max_norm, max_k=self.model.max_k,
            entity_embeddings=lambda indices: None, scoring_function=lambda *_: values)

    def select(self, heads, relation):
        selected = []
        for _, group, values in self.rows(heads, relation):
            for index in range(len(group)):
                scores, _, ids = self.candidates(group[index:index + 1], relation, values[index:index + 1], self.model.k)
                selected.append((scores, ids))
        # Upstream hybrid pads each parent's selections to the largest width
        # across the full stage, not to a per-partition width.
        width = max(scores.shape[1] for scores, _ in selected)
        scores = torch.cat([torch.nn.functional.pad(s, (0, width - s.shape[1])) for s, _ in selected])
        ids = torch.cat([torch.nn.functional.pad(i, (0, width - i.shape[1])) for _, i in selected])
        return scores, ids

    def finish(self, heads, relation, prefix, observed_relation=None):
        result = None
        for offset, group, values in self.rows(heads, relation):
            scores, _, _ = self.candidates(group, relation if observed_relation is None else observed_relation, values, None)
            best = self.combine(prefix[offset:offset + len(group), None], scores).max(0).values
            result = best if result is None else torch.maximum(result, best)
        return result[None]

    def path(self, queries):
        heads = queries[:, 0]
        if queries.shape[1] == 2:
            _, group, values = next(self.rows(heads, queries[:, 1]))
            return self.candidates(group, queries[:, 1], values, None)[0]
        stages = []
        for position in range(1, queries.shape[1] - 1):
            scores, heads = self.select(heads.flatten(), queries[:, position])
            stages.append(scores)
        if queries.shape[1] == 5:
            # Released 4p discards earlier prefixes and scores r3 again, while
            # its observed-fact lookup still uses the requested r4.
            return self.finish(heads.flatten(), queries[:, 3], stages[-1].flatten(), queries[:, 4])
        prefix = stages[0].flatten()
        if len(stages) == 2:
            # Literal upstream 3p repeat (not repeat_interleave).
            prefix = self.combine(prefix.repeat(stages[1].shape[1]), stages[1].flatten())
        return self.finish(heads.flatten(), queries[:, -1], prefix)


def query_bounded(model, source, shape, queries, batch_size):
    """Independent bounded oracle for all eleven released positive shapes.

    This is a declared arithmetic profile, not a claim of bitwise equivalence
    with the release's unpartitioned GEMMs. Both oracle and implementation must
    use the declared row grouping; numerical/rank/tie checks remain unchanged.
    """
    if len(queries) != 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Bounded CQD requires one query and a positive integer row batch size')
    stages = _BoundedStages(model, source, batch_size)
    if shape in ('1p', '2p', '3p', '4p'):
        return stages.path(queries)
    if shape == 'pi':
        return stages.combine(stages.path(queries[:, :3]), stages.path(queries[:, 3:5]))

    # Retain upstream composition for all intersection/union base branches.
    def scoring(lhs, rel, rhs):
        values = model.score_o(lhs, rel, rhs)[0]
        if model.do_sigmoid:
            values = values.sigmoid() * model.max_norm
        if model.do_normalize:
            values = model.max_norm * (values - values.min()) / (values.max() - values.min())
        return values

    union = shape in ('2u', 'up')
    function = source.query_2u_dnf if union else getattr(source, 'query_' + ('2i' if shape == 'ip' else shape))
    combine = (lambda a, b: 1 - (1 - a) * (1 - b)) if model.t_norm_name == 'prod' else torch.maximum
    base = function(entity_embeddings=model.embeddings[0], predicate_embeddings=model.embeddings[1],
                    queries=queries[:, :4] if shape in ('ip', 'up') else queries,
                    filters=model.filters, max_k=model.max_k, max_norm=model.max_norm,
                    scoring_function=scoring, **{'t_conorm' if union else 't_norm': combine if union else stages.combine})
    if shape not in ('ip', 'up'):
        return base
    existing = int((base == 1).sum()) if model.max_norm != 1 else 0
    count = min(model.k + existing, model.nentity)
    if model.max_k is not None:
        count = min(count, model.max_k)
    prefix, heads = base.topk(count, dim=1)
    return stages.finish(heads.flatten(), queries[:, 5 if union else 4], prefix.flatten())
