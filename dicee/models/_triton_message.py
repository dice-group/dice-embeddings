"""Fused CSR relational message passing without an edge-feature allocation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _distmult_sum(S, R, B, O, PTR, SRC, TYPE,
                  SB: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
                  RB: tl.constexpr, RN: tl.constexpr, RD: tl.constexpr,
                  BB: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                  N: tl.constexpr, D: tl.constexpr, FEATURES: tl.constexpr,
                  EDGES: tl.constexpr):
    row, batch = tl.program_id(0), tl.program_id(1)
    feature = tl.arange(0, FEATURES)
    start, end = tl.load(PTR + row), tl.load(PTR + row + 1)
    total = tl.full((FEATURES,), 0, tl.float32)
    for first in range(start, end, EDGES):
        edge = first + tl.arange(0, EDGES)
        source = tl.load(SRC + edge, edge < end, other=0)
        relation = tl.load(TYPE + edge, edge < end, other=0)
        mask = (edge[:, None] < end) & (feature[None, :] < D)
        state = tl.load(S + batch * SB + source[:, None] * SN + feature[None, :] * SD, mask, other=0)
        rel = tl.load(R + batch * RB + relation[:, None] * RN + feature[None, :] * RD, mask, other=0)
        total += tl.sum(state * rel, axis=0)
    boundary = tl.load(B + batch * BB + row * BN + feature * BD, feature < D, other=0)
    tl.store(O + (batch * N + row) * D + feature, total + boundary, feature < D)


@triton.jit
def _distmult_tiles(S, R, O, PTR, SRC, TYPE, ROWS, STARTS,
                    SB: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
                    RB: tl.constexpr, RN: tl.constexpr, RD: tl.constexpr,
                    N: tl.constexpr, D: tl.constexpr, FEATURES: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    row = tl.load(ROWS + tile)
    edge = tl.load(STARTS + tile) + tl.arange(0, 32)
    end = tl.load(PTR + row + 1)
    feature = tl.arange(0, FEATURES)
    source = tl.load(SRC + edge, edge < end, other=0)
    relation = tl.load(TYPE + edge, edge < end, other=0)
    mask = (edge[:, None] < end) & (feature[None, :] < D)
    state = tl.load(S + batch * SB + source[:, None] * SN + feature[None, :] * SD, mask, other=0)
    rel = tl.load(R + batch * RB + relation[:, None] * RN + feature[None, :] * RD, mask, other=0)
    total = tl.sum(state * rel, axis=0)
    tl.atomic_add(O + (batch * N + row) * D + feature, total, feature < D, sem='relaxed')


def distmult_sum(states, boundary, relations, layout):
    batch, nodes, dim = states.shape
    # Split high-degree rows into independent tiles. This avoids serial hub
    # loops and retains O(B*N*D) activation memory. Deterministic mode instead
    # assigns each output row to one program, without floating-point atomics.
    if len(layout[1]) > 16 * nodes and not torch.are_deterministic_algorithms_enabled():
        output = boundary.clone(memory_format=torch.contiguous_format)
        if batch and len(layout[3]):
            _distmult_tiles[(len(layout[3]), batch)](
                states, relations, output, *layout,
                *states.stride(), *relations.stride(), nodes, dim,
                triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False,
            )
        return output
    output = torch.empty_like(states, memory_format=torch.contiguous_format)
    if batch and nodes:
        _distmult_sum[(nodes, batch)](
            states, relations, boundary, output, *layout[:3],
            *states.stride(), *relations.stride(), *boundary.stride(),
            nodes, dim, triton.next_power_of_2(dim), 32,
            num_warps=4, enable_fp_fusion=False,
        )
    return output
