"""Optional Flock inference kernels; training and unsupported dtypes use Torch."""
import torch


def embedding_sum(records, weights, heads, query, relation_prediction):
    if (not weights[0].is_cuda or any(w.dtype != torch.float32 or not w.is_contiguous() for w in weights)
            or any(not r.is_contiguous() for r in records)):
        return None
    try:
        from ._triton_flock import embedding_sum as run
    except ImportError:
        return None
    with torch.cuda.device(weights[0].device):
        return run(records, weights, heads, query, relation_prediction)


def rms_norm(x, weight, eps):
    if not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous() or x.shape[-1] > 128:
        return None
    try:
        from ._triton_flock import rms_norm as run
    except ImportError:
        return None
    with torch.cuda.device(x.device):
        return run(x, weight, eps)
