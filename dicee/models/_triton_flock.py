"""Fused Flock record embeddings and RMS normalization."""
import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_sum(WALK, NAMED, RESTART, NEIGHBOR, TYPE, NAMED_TYPE, DIRECTION,
                   W0, W1, W2, W3, W4, W5, W6, HEAD, QUERY, O,
                   HS: tl.constexpr, QS: tl.constexpr, D: tl.constexpr, PER_BATCH: tl.constexpr,
                   N: tl.constexpr, RELATION: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos, feature = index // D, index % D
    mask = index < N
    batch = pos // PER_BATCH
    named = tl.load(NAMED + pos, mask, other=1) - 1
    named_type = tl.load(NAMED_TYPE + pos, mask, other=1) - 1
    restart = tl.load(RESTART + pos, mask, other=0)
    neighbor = tl.load(NEIGHBOR + pos, mask, other=0)
    direction = tl.load(DIRECTION + pos, mask, other=0)
    node = tl.load(WALK + pos, mask, other=0)
    head = tl.load(HEAD + batch * HS, mask, other=-1)
    query = tl.load(QUERY + batch * QS, mask, other=-1)
    marker0 = (node == head).to(tl.int32)
    if RELATION:
        marker1 = (node == query).to(tl.int32)
    else:
        marker1 = (tl.load(TYPE + pos, mask, other=-1) == query).to(tl.int32)
    value = tl.load(W0 + named * D + feature, mask, other=0)
    value += tl.load(W1 + named_type * D + feature, mask, other=0)
    value += tl.load(W2 + restart * D + feature, mask, other=0)
    value += tl.load(W3 + neighbor * D + feature, mask, other=0)
    value += tl.load(W4 + direction * D + feature, mask, other=0)
    value += tl.load(W5 + marker0 * D + feature, mask, other=0)
    value += tl.load(W6 + marker1 * D + feature, mask, other=0)
    tl.store(O + index, value, mask)


def embedding_sum(records, weights, heads, query, relation_prediction):
    output = weights[0].new_empty(*records[0].shape, weights[0].shape[-1])
    _embedding_sum[(triton.cdiv(output.numel(), 512),)](
        *records, *weights, heads, query, output, heads.stride(0), query.stride(0), output.shape[-1],
        records[0].shape[1] * records[0].shape[2], output.numel(), relation_prediction, 512,
        num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _rms_norm(X, W, O, D: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
              FEATURES: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, FEATURES)
    mask = (row[:, None] < N) & (col[None, :] < D)
    x = tl.load(X + row[:, None] * D + col[None, :], mask, other=0)
    norm = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    weight = tl.load(W + col, col < D, other=0)
    value = x * norm[:, None] * weight[None, :]
    tl.store(O + row[:, None] * D + col[None, :], value, mask)


def rms_norm(x, weight, eps):
    output = torch.empty_like(x)
    dim = x.shape[-1]
    rows = x.numel() // dim
    if rows:
        _rms_norm[(triton.cdiv(rows, 4),)](x, weight, output, dim, rows, eps,
                                         triton.next_power_of_2(dim), 4, num_warps=4, enable_fp_fusion=False)
    return output
