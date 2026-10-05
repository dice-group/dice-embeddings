"""Optional KG-ICL inference kernels; training and unsupported settings use PyTorch.

The fused path keeps a dense ``[batch, entities, dim]`` state with an active
mask. Messages leave only active entities, so its scores equal upstream's
hop-by-hop expansion. Triton is imported only on the CUDA path.
"""
import torch
from torch.nn import functional as F

from ._fused_message import csr_layout
from ._inference import to_device


def fused_reasoning_supported(model, edges):
    """Whether batched Triton reasoning applies; ``backend='triton'`` raises when it cannot."""
    backend = getattr(model, '_inference_backend', 'auto')
    weight = model.W_final.weight
    supported = (edges[0] is model.edge_index and edges[1] is model.edge_type and weight.is_cuda
                 and weight.dtype == torch.float32 and model.dim <= 128 and model.attn_dim <= 16)
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


def attention_width(model):
    return max(2, 1 << (model.attn_dim - 1).bit_length())


def layer_tables(model, relation, prompt):
    """Per-layer relation states, their attention projections and the query term of one relation.

    Computed for a single relation at a time, so values never depend on which
    other relations share a query batch.
    """
    width = attention_width(model)
    rel, result = prompt, []
    for i, layer in enumerate(model.gnn_layers):
        rel = model.layer_norms_rel[i](rel + F.relu(model.rel_transfer[i](rel)))
        result.append((rel, F.pad(layer.Wr_attn(rel), (0, width - model.attn_dim)),
                       F.pad(layer.Wqr_attn(rel[relation]), (0, width - model.attn_dim))))
        rel = model.layer_norms_rel[i](rel)
    return result


def fused_scores(model, heads, relations, ids, prompts, edges):
    """All-entity scores ``[B, N]`` and first-reach depths (or None) for a query batch."""
    from ._triton_kgicl import Workspace, aggregate, project, update
    from .kgicl import RRELU_LOWER, RRELU_UPPER

    slope = (RRELU_LOWER + RRELU_UPPER) / 2
    device, batch, nodes = heads.device, len(heads), model.num_entities
    unique = list(dict.fromkeys(ids))
    position = {q: i for i, q in enumerate(unique)}
    tables = model._layer_tables(unique, prompts)
    query_index = to_device(torch.tensor([position[q] for q in ids], dtype=torch.int32), device)
    if model._destination_edges is None:
        model._destination_edges = model.edge_index.flip(0).contiguous()
    layout = csr_layout(model._destination_edges, model.edge_type, nodes)
    width = attention_width(model)
    space = Workspace(batch, nodes, model.dim, width, device)
    rows = torch.arange(batch, device=device)
    initial = torch.stack([prompts[q][q] for q in ids])
    space.states[0].index_put_((rows, heads), initial)
    space.active[0].index_put_((rows, heads), torch.ones((), dtype=torch.int8, device=device))
    space.attention[0].index_put_((rows, heads), project(initial, model.gnn_layers[0].Ws_attn.weight, width))
    depth = None
    if model.masked_distances:
        depth = torch.full((batch, nodes), -1, dtype=torch.int8, device=device)
        depth.index_put_((rows, heads), torch.zeros((), dtype=torch.int8, device=device))
    current = 0
    for i, layer in enumerate(model.gnn_layers):
        relation_states, attention_states, query_terms = tables[i]
        alpha = layer.w_alpha
        aggregate(space, current, relation_states, attention_states, query_terms, query_index,
                  F.pad(alpha.weight[0], (0, width - model.attn_dim)), alpha.bias, layout, slope)
        last = i == len(model.gnn_layers) - 1
        update(space, current, layer, model.gate, model.layer_norms[i],
               None if last else model.gnn_layers[i + 1].Ws_attn.weight, model.W_final, slope, i == 0, last)
        if depth is not None:
            depth.masked_fill_((space.active[1 - current] != 0) & (space.active[current] == 0), i + 1)
        current = 1 - current
    return space.scores, depth
