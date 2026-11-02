"""Reference-order sparse reductions for checkpoint reproduction."""

import torch


def cached_layout(cache, name, edges, types, nodes, *, relation_order=False):
    # Inference tensors have no version counter, so their layout cannot be cached safely.
    if edges.is_inference() or types.is_inference():
        return layout(edges, types, nodes, relation_order=relation_order)
    token = edges.device, edges.data_ptr(), edges._version, types.data_ptr(), types._version
    previous = cache.get(name)
    if previous is None or previous[0] != token:
        cache[name] = token, layout(edges, types, nodes, relation_order=relation_order)
    return cache[name][1]


def layout(edges, types, nodes, *, relation_order=False):
    key = edges[0] * (edges[1].max() + 1) + edges[1] if types.numel() else types
    if relation_order and types.numel():
        key = key * (types.max() + 1) + types
    order = key.argsort()
    rows, sources = edges[:, order]
    pointers = torch.cat((rows.new_zeros(1), rows.bincount(minlength=nodes).cumsum(0)))
    return pointers, sources, types[order]


def aggregate(states, relations, edges, reduction='sum', *, num_warps=1):
    """Reduce neighbours sequentially, matching the upstream CUDA RSPMM loop.

    ``pna`` returns (sum, sum of squares, max, min) from one pass over the edges;
    squares equal separate reductions of ``states.square()`` and ``relations.square()``.
    Warps only change scheduling: each feature is still accumulated by one lane.
    """
    if states.is_cuda:
        from ._ordered_aggregation_cuda import ordered_reduce
        return ordered_reduce(states, relations, edges, reduction, num_warps=num_warps)
    if reduction == 'pna':
        return (aggregate(states, relations, edges), aggregate(states.square(), relations.square(), edges),
                aggregate(states, relations, edges, 'max'), aggregate(states, relations, edges, 'min'))
    from ._ordered_aggregation_cpu import ordered_reduce
    result = ordered_reduce(states, relations, edges, reduction)
    if result is not None:
        return result
    pointers, sources, types = edges
    fill = 0 if reduction == 'sum' else -torch.finfo(states.dtype).max if reduction == 'max' else torch.finfo(states.dtype).max
    output = torch.full_like(states, fill)
    for row in range(len(states)):
        for edge in range(int(pointers[row]), int(pointers[row + 1])):
            value = states[sources[edge]] * relations[types[edge]]
            output[row] = (output[row] + value if reduction == 'sum' else
                           torch.maximum(output[row], value) if reduction == 'max' else torch.minimum(output[row], value))
    return output
