"""Optional inference-only DistMult/sum kernels; training stays in PyTorch.

The CSR layout belongs to the runtime edge tensor, so all convolution layers
share it and moving/replacing a graph naturally drops it. No graph or compiled
kernel is included in a checkpoint. Triton is imported only on the CUDA path.
"""
import weakref

import torch


def tensor_version(tensor):
    return None if tensor.is_inference() else tensor._version


def csr_layout(edge_index, edge_type, num_nodes):
    signature = (tensor_version(edge_index), tensor_version(edge_type), num_nodes)
    cached = getattr(edge_index, '_dicee_csr', None)
    if cached is not None and cached[0]() is edge_type and cached[1] == signature:
        return cached[2]
    order = edge_index[0].argsort(stable=True)
    counts = torch.bincount(edge_index[0], minlength=num_nodes)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    tile_counts = (counts + 31) // 32
    tile_offsets = tile_counts.cumsum(0) - tile_counts
    rows = torch.arange(num_nodes, device=counts.device).repeat_interleave(tile_counts)
    starts = offsets[rows] + (torch.arange(len(rows), device=rows.device) - tile_offsets[rows]) * 32
    layout = (offsets, edge_index[1, order].contiguous(), edge_type[order].contiguous(), rows, starts)
    # Inference tensors lack mutation counters. Do not retain a potentially
    # stale layout for graphs constructed inside inference_mode().
    if not edge_index.is_inference() and not edge_type.is_inference():
        edge_index._dicee_csr = (weakref.ref(edge_type), signature, layout)
    return layout


def fused_distmult_sum(states, boundary, edge_index, edge_type, relations, backend='auto'):
    """Return a fused update, or None when the portable implementation is needed."""
    supported = (states.is_cuda and states.dtype == torch.float32
                 and relations.dtype == states.dtype and boundary.dtype == states.dtype
                 and (not torch.is_grad_enabled() or not any(x.requires_grad for x in (states, boundary, relations)))
                 and states.shape[-1] <= 128)
    if backend == 'torch' or not supported:
        return None
    try:
        from ._triton_message import distmult_sum
    except ImportError:
        if backend == 'triton':
            raise RuntimeError('The triton inference backend requires Triton') from None
        return None
    with torch.cuda.device(states.device):
        return distmult_sum(states, boundary, relations, csr_layout(edge_index, edge_type, states.shape[1]))
