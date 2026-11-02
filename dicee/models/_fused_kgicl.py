"""Optional KG-ICL inference kernels; training and unsupported settings use PyTorch.

The fused path keeps a dense node-major ``[entities, batch, dim]`` state with an
active mask. Messages leave only active entities, so its scores equal
upstream's hop-by-hop expansion. Triton is imported only on the CUDA path.
"""
import torch
from torch.nn import functional as F

from ._fused_message import CSRLayout
from ._inference import to_device

ATTENTION = 16  # Padded attention width of the fused kernels.


def fused_reasoning_supported(model, edges):
    """Whether batched Triton reasoning applies; ``backend='triton'`` raises when it cannot."""
    backend = getattr(model, '_inference_backend', 'auto')
    weight = model.W_final.weight
    supported = (edges[0] is model.edge_index and edges[1] is model.edge_type and weight.is_cuda
                 and weight.dtype == torch.float32 and model.dim <= 128 and model.attn_dim <= ATTENTION
                 and model.num_entities < 2**31 and model.edge_type.numel() < 2**31)
    if backend == 'torch' or not supported:
        if backend == 'triton' and not supported:
            raise RuntimeError('The triton KG-ICL backend requires float32 CUDA inference on the attached graph')
        return False
    try:
        from . import _triton_kgicl  # noqa: F401
    except ImportError:
        if backend == 'triton':
            raise RuntimeError('The triton inference backend requires Triton') from None
        return False
    return True


def layer_tables(model, relation, prompt):
    """Per-layer relation states, their attention projections and the query term of one relation.

    Computed for a single relation at a time, so values never depend on which
    other relations share a query batch.
    """
    rel, result = prompt, []
    for i, layer in enumerate(model.gnn_layers):
        rel = model.layer_norms_rel[i](rel + F.relu(model.rel_transfer[i](rel)))
        result.append((rel, F.pad(layer.Wr_attn(rel), (0, ATTENTION - model.attn_dim)),
                       F.pad(layer.Wqr_attn(rel[relation]), (0, ATTENTION - model.attn_dim))))
        rel = model.layer_norms_rel[i](rel)
    return result


def tile_layout(targets, sources, types, num_nodes, edges, hub_tiles):
    """Edges grouped by target in ``edges``-wide tiles; rows with more than ``hub_tiles`` tiles are hubs.

    The same structure as ``csr_layout``, with 32-bit indices and a configurable tile width.
    """
    order = targets.argsort(stable=True)
    counts = torch.bincount(targets, minlength=num_nodes)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    tile_counts = (counts + edges - 1) // edges
    tile_offsets = tile_counts.cumsum(0) - tile_counts
    rows = torch.arange(num_nodes, device=counts.device).repeat_interleave(tile_counts)
    starts = offsets[rows] + (torch.arange(len(rows), device=rows.device) - tile_offsets[rows]) * edges
    hub = tile_counts > hub_tiles
    hub_counts = torch.where(hub, tile_counts, 0)
    return CSRLayout(*(tensor.to(torch.int32).contiguous() for tensor in (
        offsets, sources[order], types[order], rows, starts, rows[hub[rows]], starts[hub[rows]],
        torch.where(hub, hub_counts.cumsum(0) - hub_counts, -1))))


def destination_layout(model):
    """Incoming edges of every entity in segments, cached until the graph changes."""
    from ._triton_kgicl import HUB_SEGMENTS, SEGMENT
    if model._destination_layout is None:
        model._destination_layout = tile_layout(model.edge_index[1], model.edge_index[0], model.edge_type,
                                                model.num_entities, SEGMENT, HUB_SEGMENTS)
    return model._destination_layout


def fused_scores(model, heads, relations, ids, prompts, edges):
    """All-entity scores ``[B, N]`` and first-reach depths (or None) for a query batch."""
    from ._triton_kgicl import Workspace, aggregate, update
    from .kgicl import RRELU_LOWER, RRELU_UPPER

    slope = (RRELU_LOWER + RRELU_UPPER) / 2
    device, batch, nodes = heads.device, len(heads), model.num_entities
    unique = list(dict.fromkeys(ids))
    position = {q: i for i, q in enumerate(unique)}
    tables = model._layer_tables(unique, prompts)
    query_index = to_device(torch.tensor([position[q] for q in ids], dtype=torch.int32), device)
    layout = destination_layout(model)
    space = Workspace(nodes, batch, model.dim, model.attn_dim, device)
    rows = torch.arange(batch, device=device)
    space.states[0].index_put_((heads, rows), torch.stack([prompts[q][q] for q in ids]))
    space.active[0].index_put_((heads, rows), torch.ones((), dtype=torch.int8, device=device))
    depth = None
    if model.masked_distances:
        depth = torch.full((nodes, batch), -1, dtype=torch.int8, device=device)
        depth.index_put_((heads, rows), torch.zeros((), dtype=torch.int8, device=device))
    current = 0
    for i, layer in enumerate(model.gnn_layers):
        relation_states, attention_states, query_terms = tables[i]
        alpha = layer.w_alpha
        aggregate(space, current, relation_states, attention_states, query_terms, query_index,
                  F.pad(alpha.weight[0], (0, ATTENTION - model.attn_dim)), alpha.bias, layer.Ws_attn.weight, layout, slope)
        update(space, current, layer, model.gate, model.layer_norms[i], model.W_final, slope, i == 0,
               i == len(model.gnn_layers) - 1)
        if depth is not None:
            depth.masked_fill_((space.active[1 - current] != 0) & (space.active[current] == 0), i + 1)
        current = 1 - current
    return space.scores.T.contiguous(), None if depth is None else depth.T.contiguous()
