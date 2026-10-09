"""CSR inference kernel preserving upstream neighbour accumulation order."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Rows above this multiple of the mean degree launch first, so their long serial
# chains overlap the remaining rows instead of trailing the launch.
HUB_FACTOR = 8
# Independent edge loads issued ahead of the strictly sequential accumulation.
UNROLL = 8


@triton.jit
def _reduce(S, R, O, O2, O3, O4, PTR, SRC, REL, ORDER, HUBS, ROWS,
            SN: tl.constexpr, SB: tl.constexpr, SD: tl.constexpr,
            RN: tl.constexpr, RB: tl.constexpr, RD: tl.constexpr,
            B: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr, KIND: tl.constexpr, UNROLL: tl.constexpr):
    # Hubs for every batch element first; the remaining rows stay batch-major
    # so concurrently running programs share one batch element's states in L2.
    program = tl.program_id(0)
    if program < HUBS * B:
        rank = program // B
        batch = program % B
    else:
        rest = program - HUBS * B
        batch = rest // tl.maximum(ROWS - HUBS, 1)
        rank = HUBS + rest % tl.maximum(ROWS - HUBS, 1)
    row = tl.load(ORDER + rank)
    feature = tl.arange(0, WIDTH)
    present = feature < D
    start, end = tl.load(PTR + row), tl.load(PTR + row + 1)
    value = tl.full((WIDTH,), 0. if KIND == 'sum' or KIND == 'pna' else
                    -3.4028234663852886e38 if KIND == 'max' else 3.4028234663852886e38, tl.float32)
    squares = tl.full((WIDTH,), 0., tl.float32)
    high = tl.full((WIDTH,), -3.4028234663852886e38, tl.float32)
    low = tl.full((WIDTH,), 3.4028234663852886e38, tl.float32)
    for block in range(start, end, UNROLL):
        # Loads do not depend on the running value, so they overlap; updates
        # remain one edge at a time in CSR order, as in the upstream loop.
        for offset in tl.static_range(UNROLL):
            edge = block + offset
            valid = edge < end
            source = tl.load(SRC + edge, valid, other=0)
            relation = tl.load(REL + edge, valid, other=0)
            state = tl.load(S + source * SN + batch * SB + feature * SD, present & valid, other=0)
            rel = tl.load(R + relation * RN + batch * RB + feature * RD, present & valid, other=0)
            # Explicit round-to-nearest operations also prevent packed-instruction
            # contraction when a lane owns multiple features on newer GPUs.
            message = libdevice.mul_rn(state, rel)
            if KIND == 'sum':
                value = tl.where(valid, libdevice.add_rn(value, message), value)
            elif KIND == 'max':
                value = tl.where(valid, tl.maximum(value, message), value)
            elif KIND == 'min':
                value = tl.where(valid, tl.minimum(value, message), value)
            else:
                # PNA reads each edge once; squares match separate h*h and r*r inputs.
                square = libdevice.mul_rn(libdevice.mul_rn(state, state), libdevice.mul_rn(rel, rel))
                value = tl.where(valid, libdevice.add_rn(value, message), value)
                squares = tl.where(valid, libdevice.add_rn(squares, square), squares)
                high = tl.where(valid, tl.maximum(high, message), high)
                low = tl.where(valid, tl.minimum(low, message), low)
    offset = (row * B + batch) * D + feature
    tl.store(O + offset, value, present)
    if KIND == 'pna':
        tl.store(O2 + offset, squares, present)
        tl.store(O3 + offset, high, present)
        tl.store(O4 + offset, low, present)


def _schedule(pointers):
    """Rows by launch rank, hubs first, cached on the CSR pointer tensor."""
    version = None if pointers.is_inference() else pointers._version
    cached = getattr(pointers, '_dicee_schedule', None)
    if cached is not None and version is not None and cached[0] == version:
        return cached[1:]
    degree = pointers[1:] - pointers[:-1]
    hubs = degree > HUB_FACTOR * degree.double().mean()
    rows = torch.arange(len(degree), device=pointers.device)
    order = torch.cat((rows[hubs][degree[hubs].argsort(descending=True, stable=True)], rows[~hubs])).contiguous()
    result = order, int(hubs.sum())
    if version is not None:
        pointers._dicee_schedule = (version, *result)
    return result


def ordered_reduce(states, relations, edges, reduction, *, num_warps=1):
    """Return one reduction, or (sum, sum of squares, max, min) for ``pna``."""
    if states.dtype != torch.float32 or relations.dtype != torch.float32:
        raise ValueError('Reference sparse inference requires float32')
    if num_warps not in (1, 2, 4, 8):
        raise ValueError('Ordered aggregation warps must be 1, 2, 4 or 8')
    nodes, batch, dim = states.shape
    outputs = [torch.empty((nodes, batch, dim), device=states.device, dtype=states.dtype)
               for _ in range(4 if reduction == 'pna' else 1)]
    if nodes and batch:
        order, hubs = _schedule(edges[0])
        targets = outputs + [outputs[0]] * (4 - len(outputs))
        with torch.cuda.device(states.device):
            _reduce[(nodes * batch,)](states, relations, *targets, *edges, order, hubs, nodes,
                                      *states.stride(), *relations.stride(),
                                      batch, dim, triton.next_power_of_2(dim), reduction, UNROLL,
                                      num_warps=num_warps, enable_fp_fusion=False)
    return tuple(outputs) if reduction == 'pna' else outputs[0]
