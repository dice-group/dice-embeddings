"""Fused KG-ICL reasoning: attention-weighted CSR aggregation and GRU node updates.

Every (query, entity) row is computed by a fixed sequence of operations with
no atomics, so rows are deterministic and independent of the query batch.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

EDGES = 32


@triton.jit
def _chunk(H, AS, ACT, TBL, ATBL, SRC, TYPE, edge, end, base, table, qc, wa, ba, feature, attention,
           D: tl.constexpr, AP: tl.constexpr, SLOPE: tl.constexpr):
    # One chunk of incoming edges: sigmoid query-aware attention times (h_source + r).
    inside = edge < end
    source = tl.load(SRC + edge, inside, other=0)
    relation = tl.load(TYPE + edge, inside, other=0)
    valid = inside & (tl.load(ACT + base + source, inside, other=0) != 0)
    rows = (base + source)[:, None]
    pre = (tl.load(AS + rows * AP + attention[None, :], valid[:, None], other=0)
           + tl.load(ATBL + (table + relation)[:, None] * AP + attention[None, :], valid[:, None], other=0)
           + qc[None, :])
    pre = tl.where(pre >= 0, pre, pre * SLOPE)
    alpha = tl.sigmoid(tl.sum(pre * wa[None, :], axis=1) + ba)
    alpha = tl.where(valid, alpha, 0.)
    mask = valid[:, None] & (feature[None, :] < D)
    state = tl.load(H + rows * D + feature[None, :], mask, other=0)
    rel = tl.load(TBL + (table + relation)[:, None] * D + feature[None, :], mask, other=0)
    return tl.sum(alpha[:, None] * (state + rel), axis=0), tl.max(valid.to(tl.int32), axis=0)


@triton.jit
def _tile_partials(H, AS, ACT, TBL, ATBL, QC, QIDX, WA, BA, P, PR, PTR, SRC, TYPE, ROWS, STARTS, TILES,
                   N: tl.constexpr, D: tl.constexpr, RS: tl.constexpr, AP: tl.constexpr, FEATURES: tl.constexpr,
                   EDGES: tl.constexpr, SLOPE: tl.constexpr):
    # One program per 32-edge tile of a high in-degree row; the row program adds them in order.
    tile, batch = tl.program_id(0), tl.program_id(1).to(tl.int64)
    row = tl.load(ROWS + tile)
    end = tl.load(PTR + row + 1)
    edge = tl.load(STARTS + tile) + tl.arange(0, EDGES)
    feature, attention = tl.arange(0, FEATURES), tl.arange(0, AP)
    qi = tl.load(QIDX + batch).to(tl.int64)
    qc = tl.load(QC + qi * AP + attention)
    total, reach = _chunk(H, AS, ACT, TBL, ATBL, SRC, TYPE, edge, end, batch * N, qi * RS, qc, tl.load(WA + attention),
                          tl.load(BA), feature, attention, D, AP, SLOPE)
    tl.store(P + (batch * TILES + tile) * D + feature, total, feature < D)
    tl.store(PR + batch * TILES + tile, reach)


@triton.jit
def _aggregate(H, AS, ACT, TBL, ATBL, QC, QIDX, WA, BA, AGG, NEW, PTR, SRC, TYPE, FIRST, P, PR, TILES,
               N: tl.constexpr, D: tl.constexpr, RS: tl.constexpr, AP: tl.constexpr, FEATURES: tl.constexpr,
               EDGES: tl.constexpr, SLOPE: tl.constexpr):
    row, batch = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    feature, attention = tl.arange(0, FEATURES), tl.arange(0, AP)
    qi = tl.load(QIDX + batch).to(tl.int64)
    qc, wa, ba = tl.load(QC + qi * AP + attention), tl.load(WA + attention), tl.load(BA)
    start, end = tl.load(PTR + row), tl.load(PTR + row + 1)
    first = tl.load(FIRST + row)
    total = tl.zeros((FEATURES,), tl.float32)
    reach = tl.zeros((), tl.int32)
    if first < 0:
        for chunk in range(start, end, EDGES):
            value, hit = _chunk(H, AS, ACT, TBL, ATBL, SRC, TYPE, chunk + tl.arange(0, EDGES), end, batch * N, qi * RS,
                                qc, wa, ba, feature, attention, D, AP, SLOPE)
            total += value
            reach = tl.maximum(reach, hit)
    else:
        count = (end - start + EDGES - 1) // EDGES
        for block in range(0, count):
            total += tl.load(P + (batch * TILES + first + block) * D + feature, feature < D, other=0)
            reach = tl.maximum(reach, tl.load(PR + batch * TILES + first + block))
    current = batch * N + row
    active = tl.load(ACT + current)
    if active != 0:
        # The implicit self-loop edge with relation slot RS - 1.
        pre = tl.load(AS + current * AP + attention) + tl.load(ATBL + (qi * RS + RS - 1) * AP + attention) + qc
        pre = tl.where(pre >= 0, pre, pre * SLOPE)
        alpha = tl.sigmoid(tl.sum(pre * wa, axis=0) + ba)
        state = tl.load(H + current * D + feature, feature < D, other=0)
        rel = tl.load(TBL + (qi * RS + RS - 1) * D + feature, feature < D, other=0)
        total += alpha * (state + rel)
    tl.store(AGG + current * D + feature, total, feature < D)
    tl.store(NEW + current, ((active != 0) | (reach != 0)).to(tl.int8))


@triton.jit
def _matrix(W, rows, columns, R: tl.constexpr, C: tl.constexpr):
    # W[rows, columns] for an [R, C] row-major matrix, transposed for x @ W.T; padding is zero.
    return tl.load(W + columns[None, :] * C + rows[:, None], (rows[:, None] < C) & (columns[None, :] < R), other=0)


@triton.jit
def _gate(x, W, B, index: tl.constexpr, feature, D: tl.constexpr):
    # One GRU gate block: x @ W[index * D:(index + 1) * D].T + b[index * D:(index + 1) * D].
    value = tl.dot(x, _matrix(W + index * D * D, feature, feature, D, D), input_precision="ieee")
    return value + tl.load(B + index * D + feature, feature < D, other=0)[None, :]


@triton.jit
def _update(AGG, H, NEW, OUT, AS, SCORE, WH, LNW, LNB, WIH, WHH, BIH, BHH, WS, WF, TOTAL,
            D: tl.constexpr, A: tl.constexpr, AP: tl.constexpr, FEATURES: tl.constexpr, ROWS: tl.constexpr,
            EPS: tl.constexpr, SLOPE: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr):
    # RReLU(W_h agg) -> LayerNorm -> GRU cell with the previous state; inactive rows stay zero.
    # The first layer's GRU state is zero: the query embedding at the head is only a message input.
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    feature = tl.arange(0, FEATURES)
    inside = rows < TOTAL
    mask = inside[:, None] & (feature[None, :] < D)
    offsets = rows[:, None] * D + feature[None, :]
    x = tl.load(AGG + offsets, mask, other=0)
    if FIRST:
        previous = tl.zeros((ROWS, FEATURES), tl.float32)
    else:
        previous = tl.load(H + offsets, mask, other=0)
    active = tl.load(NEW + rows, inside, other=0) != 0
    y = tl.dot(x, _matrix(WH, feature, feature, D, D), input_precision="ieee")
    y = tl.where(y >= 0, y, y * SLOPE)
    y = tl.where(feature[None, :] < D, y, 0.)
    mean = tl.sum(y, axis=1) / D
    centered = tl.where(feature[None, :] < D, y - mean[:, None], 0.)
    variance = tl.sum(centered * centered, axis=1) / D
    norm_weight = tl.load(LNW + feature, feature < D, other=0)
    norm_bias = tl.load(LNB + feature, feature < D, other=0)
    y = centered * tl.rsqrt(variance + EPS)[:, None] * norm_weight[None, :] + norm_bias[None, :]
    reset = tl.sigmoid(_gate(y, WIH, BIH, 0, feature, D) + _gate(previous, WHH, BHH, 0, feature, D))
    update = tl.sigmoid(_gate(y, WIH, BIH, 1, feature, D) + _gate(previous, WHH, BHH, 1, feature, D))
    candidate = libdevice.tanh(_gate(y, WIH, BIH, 2, feature, D) + reset * _gate(previous, WHH, BHH, 2, feature, D))
    state = (1 - update) * candidate + update * previous
    state = tl.where(active[:, None] & (feature[None, :] < D), state, 0.)
    if LAST:
        weight = tl.load(WF + feature, feature < D, other=0)
        tl.store(SCORE + rows, tl.sum(state * weight[None, :], axis=1), inside)
    else:
        tl.store(OUT + offsets, state, mask)
        attention = tl.arange(0, AP)
        projection = tl.load(WS + attention[:, None] * D + feature[None, :],
                             (attention[:, None] < A) & (feature[None, :] < D), other=0)
        values = tl.sum(state[:, None, :] * projection[None, :, :], axis=2)
        tl.store(AS + rows[:, None] * AP + attention[None, :], values, inside[:, None])


@triton.jit
def _project(X, W, OUT, TOTAL, D: tl.constexpr, A: tl.constexpr, AP: tl.constexpr, FEATURES: tl.constexpr):
    # Attention projection of single rows, with the reduction used by _update.
    row = tl.program_id(0)
    feature, attention = tl.arange(0, FEATURES), tl.arange(0, AP)
    state = tl.load(X + row * D + feature, feature < D, other=0)
    projection = tl.load(W + attention[:, None] * D + feature[None, :], (attention[:, None] < A) & (feature[None, :] < D), other=0)
    values = tl.sum(state[None, :] * projection, axis=1)
    tl.store(OUT + row * AP + attention, values)


class Workspace:
    """Per-call activation buffers for one query batch."""

    def __init__(self, batch, nodes, dim, ap, device):
        self.states = [torch.zeros(batch, nodes, dim, device=device) for _ in range(2)]
        self.attention = [torch.zeros(batch, nodes, ap, device=device) for _ in range(2)]
        self.active = [torch.zeros(batch, nodes, dtype=torch.int8, device=device) for _ in range(2)]
        self.aggregate = torch.empty(batch, nodes, dim, device=device)
        self.scores = torch.empty(batch, nodes, device=device)


def project(rows, weight, ap):
    output = rows.new_zeros(len(rows), ap)
    if len(rows):
        _project[(len(rows),)](rows, weight, output, len(rows), rows.shape[-1], weight.shape[0], ap,
                               max(16, triton.next_power_of_2(rows.shape[-1])), num_warps=1, enable_fp_fusion=False)
    return output


def aggregate(space, current, tables, attention_tables, query_constants, query_index, alpha_weight, alpha_bias, layout, slope):
    batch, nodes, dim = space.states[current].shape
    ap = space.attention[current].shape[-1]
    relation_slots = tables.shape[1]
    features = max(16, triton.next_power_of_2(dim))
    edges = (layout.offsets, layout.sources, layout.types)
    hubs = len(layout.hub_rows)
    partials = torch.empty((batch, max(hubs, 1), dim), device=tables.device)
    flags = torch.empty((batch, max(hubs, 1)), dtype=torch.int32, device=tables.device)
    common = (space.states[current], space.attention[current], space.active[current], tables, attention_tables,
              query_constants, query_index, alpha_weight, alpha_bias)
    if hubs:
        _tile_partials[(hubs, batch)](*common, partials, flags, *edges, layout.hub_rows, layout.hub_starts, hubs,
                                      nodes, dim, relation_slots, ap, features, EDGES, slope, num_warps=4, enable_fp_fusion=False)
    _aggregate[(nodes, batch)](*common, space.aggregate, space.active[1 - current], *edges, layout.hub_first, partials, flags,
                               max(hubs, 1), nodes, dim, relation_slots, ap, features, EDGES, slope,
                               num_warps=4, enable_fp_fusion=False)


def update(space, current, layer, gate, norm, next_projection, final, slope, first, last):
    batch, nodes, dim = space.states[current].shape
    ap = space.attention[current].shape[-1]
    total = batch * nodes
    rows = 32
    projection = next_projection if next_projection is not None else final.weight
    _update[(triton.cdiv(total, rows),)](
        space.aggregate, space.states[current], space.active[1 - current], space.states[1 - current], space.attention[1 - current],
        space.scores, layer.W_h.weight, norm.weight, norm.bias, gate.weight_ih_l0, gate.weight_hh_l0, gate.bias_ih_l0,
        gate.bias_hh_l0, projection, final.weight, total, dim,
        next_projection.shape[0] if next_projection is not None else 1, ap,
        max(16, triton.next_power_of_2(dim)), rows, norm.eps, slope, first, last, num_warps=4, enable_fp_fusion=False)
