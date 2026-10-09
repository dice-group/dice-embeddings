"""Fused KG-ICL reasoning: attention-weighted CSR aggregation and GRU node updates.

States are node-major, ``[entities, queries, dim]``: one program aggregates the
incoming edges of one entity for a fixed block of queries, whose source states
are contiguous. The source term ``Ws_attn h`` of the attention is projected
once per node and layer rather than once per edge. Every (entity, query) row is
computed by a fixed sequence of operations with no atomics, so rows are
deterministic and independent of the query batch: blocks always hold
``QUERIES`` lanes, padded when the batch is smaller, and high in-degree rows
are split into partial sums that are added in order.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

EDGES = 4          # Edges per chunk; lane j accumulates edges j, j + EDGES, ... of a row in order.
QUERIES = 4        # Queries per aggregation program.
WARPS = 1          # Warps per aggregation program: many small programs hide the gather latency.
SEGMENT = 64       # Edges per partial sum of a hub row.
HUB_SEGMENTS = 4   # Rows with more segments are hubs, split across programs.
ATTENTION = tl.constexpr(16)  # Row width of the padded attention tables.


@triton.jit
def _alpha(projected, table, query_term, weight, bias, valid, SLOPE: tl.constexpr):
    # sigmoid(w_alpha . rrelu(Ws h + Wr r + Wqr q) + b_alpha) over the last axis; padded lanes are zero.
    pre = projected + table + query_term
    pre = tl.where(pre >= 0, pre, pre * SLOPE)
    return tl.where(valid, tl.sigmoid(tl.sum(pre * weight, axis=-1) + bias), 0.)


@triton.jit
def _queries(QIDX, QC, WA, B, A: tl.constexpr, AW: tl.constexpr, QUERIES: tl.constexpr):
    # The query block of this program: ids, mask, relation slots, attention query terms and w_alpha.
    queries = tl.program_id(1) * QUERIES + tl.arange(0, QUERIES)
    query_mask = queries < B
    lane = tl.arange(0, AW)
    qi = tl.load(QIDX + queries, query_mask, other=0).to(tl.int64)
    query_term = tl.load(QC + qi[:, None] * ATTENTION + lane[None, :], (lane < A)[None, :], other=0)
    return queries, query_mask, qi, query_term, tl.load(WA + lane, lane < A, other=0)


@triton.jit
def _segment(H, AS, ACT, TBL, ATBL, SRC, TYPE, start, end, queries, query_mask, qi, query_term, weight, bias,
             B, D: tl.constexpr, A: tl.constexpr, AW: tl.constexpr, RS, FEATURES: tl.constexpr,
             EDGES: tl.constexpr, QUERIES: tl.constexpr, SLOPE: tl.constexpr):
    # Sum of alpha * (h_source + r) over edges [start, end) for a query block, and which queries an edge reaches.
    feature = tl.arange(0, FEATURES)
    lane = tl.arange(0, AW)
    lanes = tl.zeros((EDGES, QUERIES, FEATURES), tl.float32)
    reach = tl.zeros((QUERIES,), tl.int32)
    for chunk in range(start, end, EDGES):
        edge = chunk + tl.arange(0, EDGES)
        inside = edge < end
        source = tl.load(SRC + edge, inside, other=0).to(tl.int64)
        pair = source[:, None] * B + queries[None, :]
        present = inside[:, None] & query_mask[None, :]
        valid = present & (tl.load(ACT + pair, present, other=0) != 0)
        hits = tl.max(valid.to(tl.int32), axis=0)
        # Chunks without an active source contribute nothing; skip them.
        if tl.max(hits, axis=0) != 0:
            slot = qi[None, :] * RS + tl.load(TYPE + edge, inside, other=0).to(tl.int64)[:, None]
            attention = valid[:, :, None] & (lane < A)[None, None, :]
            projected = tl.load(AS + pair[:, :, None] * AW + lane[None, None, :], attention, other=0)
            table = tl.load(ATBL + slot[:, :, None] * ATTENTION + lane[None, None, :], attention, other=0)
            alpha = _alpha(projected, table, query_term[None, :, :], weight[None, None, :], bias, valid, SLOPE)
            mask = valid[:, :, None] & (feature < D)[None, None, :]
            state = tl.load(H + pair[:, :, None] * D + feature[None, None, :], mask, other=0)
            rel = tl.load(TBL + slot[:, :, None] * D + feature[None, None, :], mask, other=0)
            lanes += alpha[:, :, None] * (state + rel)
            reach = tl.maximum(reach, hits)
    return tl.sum(lanes, axis=0), reach


@triton.jit
def _project(H, ACT, WS, AS, TOTAL, D: tl.constexpr, A: tl.constexpr, AW: tl.constexpr, FEATURES: tl.constexpr,
             ROWS: tl.constexpr, PRECISION: tl.constexpr):
    # Attention source terms Ws_attn h of active rows; rows that are inactive are never read.
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    inside = rows < TOTAL
    active = inside & (tl.load(ACT + rows, inside, other=0) != 0)
    if tl.max(active.to(tl.int32), axis=0) == 0:
        return
    feature = tl.arange(0, FEATURES)
    lane = tl.arange(0, ATTENTION)
    x = tl.load(H + rows[:, None] * D + feature[None, :], active[:, None] & (feature < D)[None, :], other=0)
    weight = tl.load(WS + lane[None, :] * D + feature[:, None], (feature < D)[:, None] & (lane < A)[None, :], other=0)
    projected = tl.dot(x, weight, input_precision=PRECISION)
    tl.store(AS + rows[:, None] * AW + lane[None, :], projected, inside[:, None] & (lane < AW)[None, :])


@triton.jit
def _hub_partials(H, AS, ACT, TBL, ATBL, QC, QIDX, WA, BA, P, PR, PTR, SRC, TYPE, SEGMENT_ROWS, SEGMENT_STARTS,
                  B, D: tl.constexpr, A: tl.constexpr, AW: tl.constexpr, RS, FEATURES: tl.constexpr,
                  EDGES: tl.constexpr, QUERIES: tl.constexpr, SEGMENT: tl.constexpr, SLOPE: tl.constexpr):
    # One program per segment of a hub row and query block; the row program adds the segments in order.
    segment = tl.program_id(0)
    queries, query_mask, qi, query_term, weight = _queries(QIDX, QC, WA, B, A, AW, QUERIES)
    row = tl.load(SEGMENT_ROWS + segment)
    start = tl.load(SEGMENT_STARTS + segment)
    end = tl.minimum(start + SEGMENT, tl.load(PTR + row + 1))
    total, reach = _segment(H, AS, ACT, TBL, ATBL, SRC, TYPE, start, end, queries, query_mask, qi, query_term, weight,
                            tl.load(BA), B, D, A, AW, RS, FEATURES, EDGES, QUERIES, SLOPE)
    feature = tl.arange(0, FEATURES)
    rows = segment.to(tl.int64) * B + queries
    tl.store(P + rows[:, None] * D + feature[None, :], total, query_mask[:, None] & (feature < D)[None, :])
    tl.store(PR + rows, reach, query_mask)


@triton.jit
def _aggregate(H, AS, ACT, TBL, ATBL, QC, QIDX, WA, BA, AGG, NEW, PTR, SRC, TYPE, FIRST, P, PR,
               B, D: tl.constexpr, A: tl.constexpr, AW: tl.constexpr, RS, FEATURES: tl.constexpr,
               EDGES: tl.constexpr, QUERIES: tl.constexpr, SEGMENT: tl.constexpr, SLOPE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    queries, query_mask, qi, query_term, weight = _queries(QIDX, QC, WA, B, A, AW, QUERIES)
    bias = tl.load(BA)
    feature = tl.arange(0, FEATURES)
    start, end = tl.load(PTR + row), tl.load(PTR + row + 1)
    first = tl.load(FIRST + row)
    if first < 0:
        total, reach = _segment(H, AS, ACT, TBL, ATBL, SRC, TYPE, start, end, queries, query_mask, qi, query_term, weight,
                                bias, B, D, A, AW, RS, FEATURES, EDGES, QUERIES, SLOPE)
    else:
        total = tl.zeros((QUERIES, FEATURES), tl.float32)
        reach = tl.zeros((QUERIES,), tl.int32)
        for segment in range(0, (end - start + SEGMENT - 1) // SEGMENT):
            rows = (first + segment).to(tl.int64) * B + queries
            total += tl.load(P + rows[:, None] * D + feature[None, :], query_mask[:, None] & (feature < D)[None, :], other=0)
            reach = tl.maximum(reach, tl.load(PR + rows, query_mask, other=0))
    current = row * B + queries
    active = query_mask & (tl.load(ACT + current, query_mask, other=0) != 0)
    # The implicit self-loop edge with relation slot RS - 1.
    slot = qi * RS + RS - 1
    lane = tl.arange(0, AW)
    attention = active[:, None] & (lane < A)[None, :]
    projected = tl.load(AS + current[:, None] * AW + lane[None, :], attention, other=0)
    table = tl.load(ATBL + slot[:, None] * ATTENTION + lane[None, :], attention, other=0)
    alpha = _alpha(projected, table, query_term, weight[None, :], bias, active, SLOPE)
    mask = active[:, None] & (feature < D)[None, :]
    state = tl.load(H + current[:, None] * D + feature[None, :], mask, other=0)
    rel = tl.load(TBL + slot[:, None] * D + feature[None, :], mask, other=0)
    total += alpha[:, None] * (state + rel)
    tl.store(AGG + current[:, None] * D + feature[None, :], total, query_mask[:, None] & (feature < D)[None, :])
    tl.store(NEW + current, (active | (reach != 0)).to(tl.int8), query_mask)


@triton.jit
def _matrix(W, rows, columns, R: tl.constexpr, C: tl.constexpr):
    # W[rows, columns] for an [R, C] row-major matrix, transposed for x @ W.T; padding is zero.
    return tl.load(W + columns[None, :] * C + rows[:, None], (rows[:, None] < C) & (columns[None, :] < R), other=0)


@triton.jit
def _gate(x, W, B, index: tl.constexpr, feature, D: tl.constexpr, PRECISION: tl.constexpr):
    # One GRU gate block: x @ W[index * D:(index + 1) * D].T + b[index * D:(index + 1) * D].
    value = tl.dot(x, _matrix(W + index * D * D, feature, feature, D, D), input_precision=PRECISION)
    return value + tl.load(B + index * D + feature, feature < D, other=0)[None, :]


@triton.jit
def _update(AGG, H, NEW, OUT, SCORE, WH, LNW, LNB, WIH, WHH, BIH, BHH, WF, TOTAL,
            D: tl.constexpr, FEATURES: tl.constexpr, ROWS: tl.constexpr, EPS: tl.constexpr, SLOPE: tl.constexpr,
            FIRST: tl.constexpr, LAST: tl.constexpr, PRECISION: tl.constexpr):
    # RReLU(W_h agg) -> LayerNorm -> GRU cell with the previous state; inactive rows stay zero.
    # The first layer's GRU state is zero: the query embedding at the head is only a message input.
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    inside = rows < TOTAL
    active = tl.load(NEW + rows, inside, other=0) != 0
    # Active sets only grow, so an inactive row's outputs are already zero.
    if tl.max(active.to(tl.int32), axis=0) == 0:
        return
    feature = tl.arange(0, FEATURES)
    mask = inside[:, None] & (feature[None, :] < D)
    offsets = rows[:, None] * D + feature[None, :]
    x = tl.load(AGG + offsets, mask & active[:, None], other=0)
    if FIRST:
        previous = tl.zeros((ROWS, FEATURES), tl.float32)
    else:
        previous = tl.load(H + offsets, mask, other=0)
    y = tl.dot(x, _matrix(WH, feature, feature, D, D), input_precision=PRECISION)
    y = tl.where(y >= 0, y, y * SLOPE)
    y = tl.where(feature[None, :] < D, y, 0.)
    mean = tl.sum(y, axis=1) / D
    centered = tl.where(feature[None, :] < D, y - mean[:, None], 0.)
    variance = tl.sum(centered * centered, axis=1) / D
    norm_weight = tl.load(LNW + feature, feature < D, other=0)
    norm_bias = tl.load(LNB + feature, feature < D, other=0)
    y = centered * tl.rsqrt(variance + EPS)[:, None] * norm_weight[None, :] + norm_bias[None, :]
    reset = tl.sigmoid(_gate(y, WIH, BIH, 0, feature, D, PRECISION) + _gate(previous, WHH, BHH, 0, feature, D, PRECISION))
    update = tl.sigmoid(_gate(y, WIH, BIH, 1, feature, D, PRECISION) + _gate(previous, WHH, BHH, 1, feature, D, PRECISION))
    candidate = libdevice.tanh(_gate(y, WIH, BIH, 2, feature, D, PRECISION)
                               + reset * _gate(previous, WHH, BHH, 2, feature, D, PRECISION))
    state = (1 - update) * candidate + update * previous
    state = tl.where(active[:, None] & (feature[None, :] < D), state, 0.)
    if LAST:
        weight = tl.load(WF + feature, feature < D, other=0)
        tl.store(SCORE + rows, tl.sum(state * weight[None, :], axis=1), inside)
    else:
        tl.store(OUT + offsets, state, mask)


class Workspace:
    """Node-major activation buffers of one query batch; outputs of inactive rows stay zero.

    Dense products use 3xTF32 tensor-core dots where available: float32-level
    accuracy, and like IEEE FMA dots a fixed per-row operation order.
    """

    def __init__(self, nodes, batch, dim, attention, device):
        self.states = [torch.zeros(nodes, batch, dim, device=device) for _ in range(2)]
        self.active = [torch.zeros(nodes, batch, dtype=torch.int8, device=device) for _ in range(2)]
        self.projected = torch.empty(nodes, batch, max(2, triton.next_power_of_2(attention)), device=device)
        self.aggregate = torch.empty(nodes, batch, dim, device=device)
        self.scores = torch.zeros(nodes, batch, device=device)
        tensor_cores = torch.cuda.get_device_capability(device) >= (8, 0)
        self.precision, self.rows = ('tf32x3', 64) if tensor_cores else ('ieee', 32)


def _features(dim):
    return max(16, triton.next_power_of_2(dim))


def aggregate(space, current, tables, attention_tables, query_constants, query_index, alpha_weight, alpha_bias,
              projection, layout, slope):
    nodes, batch, dim = space.states[current].shape
    attention, width = projection.shape[0], space.projected.shape[-1]
    total = nodes * batch
    _project[(triton.cdiv(total, space.rows),)](space.states[current], space.active[current], projection, space.projected,
                                                total, dim, attention, width, _features(dim), space.rows, space.precision,
                                                num_warps=4, enable_fp_fusion=False)
    shapes = (batch, dim, attention, width, tables.shape[1], _features(dim), EDGES, QUERIES, SEGMENT, slope)
    common = (space.states[current], space.projected, space.active[current], tables, attention_tables, query_constants,
              query_index, alpha_weight, alpha_bias)
    blocks = triton.cdiv(batch, QUERIES)
    segments = len(layout.hub_rows)
    partials = torch.empty((max(segments, 1), batch, dim), device=tables.device)
    reached = torch.empty((max(segments, 1), batch), dtype=torch.int32, device=tables.device)
    if segments:
        _hub_partials[(segments, blocks)](*common, partials, reached, layout.offsets, layout.sources, layout.types,
                                          layout.hub_rows, layout.hub_starts, *shapes, num_warps=WARPS, enable_fp_fusion=False)
    _aggregate[(nodes, blocks)](*common, space.aggregate, space.active[1 - current], layout.offsets, layout.sources,
                                layout.types, layout.hub_first, partials, reached, *shapes, num_warps=WARPS, enable_fp_fusion=False)


def update(space, current, layer, gate, norm, final, slope, first, last):
    nodes, batch, dim = space.states[current].shape
    total = nodes * batch
    _update[(triton.cdiv(total, space.rows),)](
        space.aggregate, space.states[current], space.active[1 - current], space.states[1 - current], space.scores,
        layer.W_h.weight, norm.weight, norm.bias, gate.weight_ih_l0, gate.weight_hh_l0, gate.bias_ih_l0,
        gate.bias_hh_l0, final.weight, total, dim, _features(dim), space.rows, norm.eps, slope, first, last,
        space.precision, num_warps=4, enable_fp_fusion=False)
