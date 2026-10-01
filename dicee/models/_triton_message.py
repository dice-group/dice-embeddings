"""Fused CSR relational message passing without an edge-feature allocation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_sum(S, R, SRC, TYPE, edge, end, batch, feature,
               SB: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
               RB: tl.constexpr, RN: tl.constexpr, RD: tl.constexpr, D: tl.constexpr):
    # One 32-edge chunk; the tile shape and warps fix its reduction tree.
    source = tl.load(SRC + edge, edge < end, other=0)
    relation = tl.load(TYPE + edge, edge < end, other=0)
    mask = (edge[:, None] < end) & (feature[None, :] < D)
    state = tl.load(S + batch * SB + source[:, None] * SN + feature[None, :] * SD, mask, other=0)
    rel = tl.load(R + batch * RB + relation[:, None] * RN + feature[None, :] * RD, mask, other=0)
    return tl.sum(state.to(tl.float32) * rel.to(tl.float32), axis=0)


@triton.jit
def _tile_sums(S, R, OUT, PTR, SRC, TYPE, ROWS, STARTS, TILES,
               SB: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
               RB: tl.constexpr, RN: tl.constexpr, RD: tl.constexpr,
               N: tl.constexpr, D: tl.constexpr, FEATURES: tl.constexpr, EDGES: tl.constexpr, ATOMIC: tl.constexpr):
    # One balanced program per tile: add it to its row atomically, or keep it
    # for the row program to accumulate in order.
    tile, batch = tl.program_id(0), tl.program_id(1)
    row = tl.load(ROWS + tile)
    end = tl.load(PTR + row + 1)
    edge = tl.load(STARTS + tile) + tl.arange(0, EDGES)
    feature = tl.arange(0, FEATURES)
    total = _chunk_sum(S, R, SRC, TYPE, edge, end, batch, feature, SB, SN, SD, RB, RN, RD, D)
    if ATOMIC:
        tl.atomic_add(OUT + (batch * N + row) * D + feature, total, feature < D, sem='relaxed')
    else:
        tl.store(OUT + (batch * TILES + tile) * D + feature, total, feature < D)


@triton.jit
def _distmult_sum(S, R, B, O, PTR, SRC, TYPE, FIRST, P, TILES,
                  SB: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
                  RB: tl.constexpr, RN: tl.constexpr, RD: tl.constexpr,
                  BB: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                  N: tl.constexpr, D: tl.constexpr, FEATURES: tl.constexpr,
                  EDGES: tl.constexpr, UNROLL: tl.constexpr):
    row, batch = tl.program_id(0), tl.program_id(1)
    feature = tl.arange(0, FEATURES)
    start, end = tl.load(PTR + row), tl.load(PTR + row + 1)
    first = tl.load(FIRST + row)
    total = tl.full((FEATURES,), 0, tl.float32)
    if first < 0:
        for chunk in range(start, end, EDGES):
            edge = chunk + tl.arange(0, EDGES)
            total += _chunk_sum(S, R, SRC, TYPE, edge, end, batch, feature, SB, SN, SD, RB, RN, RD, D)
    else:
        # Same chunk sums, accumulated in the same order as the inline loop.
        count = (end - start + EDGES - 1) // EDGES
        partial = P + (batch * TILES + first) * D + feature
        for block in range(0, count, UNROLL):
            for offset in tl.static_range(UNROLL):
                valid = block + offset < count
                value = tl.load(partial + (block + offset) * D, (feature < D) & valid, other=0)
                total = tl.where(valid, total + value, total)
    boundary = tl.load(B + batch * BB + row * BN + feature * BD, feature < D, other=0)
    tl.store(O + (batch * N + row) * D + feature, total + boundary, feature < D)


def distmult_sum(states, boundary, relations, layout):
    batch, nodes, dim = states.shape
    edges = (layout.offsets, layout.sources, layout.types)
    shape = (*states.stride(), *relations.stride())
    width = triton.next_power_of_2(dim)
    # Split high-degree rows into independent tiles. This avoids serial hub
    # loops and retains O(B*N*D) activation memory. Deterministic mode instead
    # keeps each row's sequential tile order, without floating-point atomics.
    if len(layout.sources) > 16 * nodes and not torch.are_deterministic_algorithms_enabled():
        # Tile atomics always accumulate in float32, including FP16/BF16 input.
        output = boundary.to(torch.float32).clone(memory_format=torch.contiguous_format)
        if batch and len(layout.tile_rows):
            _tile_sums[(len(layout.tile_rows), batch)](
                states, relations, output, *edges, layout.tile_rows, layout.tile_starts, 0,
                *shape, nodes, dim, width, 32, True, num_warps=4, enable_fp_fusion=False)
        return output.to(states.dtype)
    output = torch.empty_like(states, memory_format=torch.contiguous_format)
    if not batch or not nodes:
        return output
    # Hub rows read their tile sums from balanced programs instead of looping
    # serially; adding them in order is bitwise equal to the single-pass loop.
    hubs = len(layout.hub_rows)
    partials = torch.empty((batch, max(hubs, 1), dim), device=states.device, dtype=torch.float32)
    if hubs:
        _tile_sums[(hubs, batch)](
            states, relations, partials, *edges, layout.hub_rows, layout.hub_starts, hubs,
            *shape, nodes, dim, width, 32, False, num_warps=4, enable_fp_fusion=False)
    _distmult_sum[(nodes, batch)](
        states, relations, boundary, output, *edges, layout.hub_first, partials, max(hubs, 1),
        *shape, *boundary.stride(), nodes, dim, width, 32, 8, num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _norm_relu(X, S, W, B, O, D: tl.constexpr, EPS: tl.constexpr, RESIDUAL: tl.constexpr,
               FEATURES: tl.constexpr, ROWS: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    feature = tl.arange(0, FEATURES)
    mask = (row[:, None] < N) & (feature[None, :] < D)
    value = tl.load(X + row[:, None] * D + feature[None, :], mask, other=0)
    mean = tl.sum(value, 1) / D
    centered = tl.where(feature[None, :] < D, value - mean[:, None], 0)
    variance = tl.sum(centered * centered, 1) / D
    weight = tl.load(W + feature, feature < D, other=0)
    bias = tl.load(B + feature, feature < D, other=0)
    result = tl.maximum(centered * tl.rsqrt(variance[:, None] + EPS) * weight[None, :] + bias[None, :], 0)
    if RESIDUAL:
        result += tl.load(S + row[:, None] * D + feature[None, :], mask, other=0)
    tl.store(O + row[:, None] * D + feature[None, :], result, mask)


def norm_relu(value, states, weight, bias, eps, residual):
    output = torch.empty_like(value)
    dim = value.shape[-1]
    rows = value.numel() // dim
    if rows:
        _norm_relu[(triton.cdiv(rows, 4),)](value, states, weight, bias, output, dim, eps, residual,
                                          triton.next_power_of_2(dim), 4, rows, num_warps=4, enable_fp_fusion=False)
    return output
