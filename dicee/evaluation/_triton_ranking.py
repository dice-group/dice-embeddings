"""Filtered rank bounds without copying candidate scores or dense masks."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bounds(S, TARGETS, ROWS, FILTERS, PTR, O, STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr,
            N: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    query, tile = tl.program_id(0), tl.program_id(1)
    row = tl.load(ROWS + query)
    target = tl.load(TARGETS + query)
    target_score = tl.load(S + row * STRIDE + target * COL_STRIDE).to(tl.float32)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(S + row * STRIDE + columns * COL_STRIDE, columns < N, other=0).to(tl.float32)
    better = tl.sum(((values > target_score) & (columns < N)).to(tl.int32), 0)
    tied = tl.sum(((values == target_score) & (columns < N)).to(tl.int32), 0)
    nan = tl.sum(((values != values) & (columns < N)).to(tl.int32), 0)
    target_nan = tl.full((), 0, tl.int32)
    if tile == 0:
        first, end = tl.load(PTR + query), tl.load(PTR + query + 1)
        for start in range(first, end, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            index = tl.load(FILTERS + offsets, offsets < end, other=0)
            excluded = tl.load(S + row * STRIDE + index * COL_STRIDE, offsets < end, other=0).to(tl.float32)
            better -= tl.sum(((excluded > target_score) & (offsets < end)).to(tl.int32), 0)
            tied -= tl.sum(((excluded == target_score) & (offsets < end)).to(tl.int32), 0)
            nan -= tl.sum(((excluded != excluded) & (offsets < end)).to(tl.int32), 0)
        better += 1
        target_nan = (target_score != target_score).to(tl.int32)
    output = O + (query * TILES + tile) * 4
    tl.store(output, better)
    tl.store(output + 1, tied)
    tl.store(output + 2, nan)
    tl.store(output + 3, target_nan)


def bounds(scores, targets, filters, row_indices=None):
    target_ids = targets.detach().cpu().tolist() if isinstance(targets, torch.Tensor) else [int(target) for target in targets]
    if not target_ids:
        return []
    nodes = scores.shape[1]
    if len(filters) != len(target_ids):
        raise ValueError('One filter list is required for each target')
    if min(target_ids) < 0 or max(target_ids) >= nodes:
        raise IndexError('Target IDs are outside the score vocabulary')
    if row_indices is None:
        row_ids = list(range(len(target_ids)))
    else:
        row_ids = row_indices.detach().cpu().tolist() if isinstance(row_indices, torch.Tensor) else list(row_indices)
    if len(row_ids) != len(target_ids) or min(row_ids) < 0 or max(row_ids) >= scores.shape[0]:
        raise IndexError('Score row indices do not match the targets')
    columns, offsets = [], [0]
    for target, excluded in zip(target_ids, filters):
        excluded = [int(value) for value in excluded]
        if excluded and (min(excluded) < -nodes or max(excluded) >= nodes):
            raise IndexError('Filter IDs are outside the score vocabulary')
        columns.extend(sorted({value % nodes for value in excluded} | {target}))
        offsets.append(len(columns))
    device = scores.device
    targets = torch.tensor(target_ids, device=device)
    rows = torch.tensor(row_ids, device=device, dtype=torch.long)
    filters = torch.tensor(columns, device=device, dtype=torch.long)
    offsets = torch.tensor(offsets, device=device)
    tiles = triton.cdiv(scores.shape[1], 1024)
    partial = torch.empty(len(target_ids), tiles, 4, device=device, dtype=torch.int64)
    with torch.cuda.device(device):
        _bounds[(len(target_ids), tiles)](scores, targets, rows, filters, offsets, partial, *scores.stride(),
                                         scores.shape[1], tiles, 1024, num_warps=4)
    values = partial.sum(1).cpu().tolist()
    if any(nan or target_nan for _, _, nan, target_nan in values):
        raise ValueError('Cannot rank NaN prediction scores')
    return [(better, tied) for better, tied, _, _ in values]
