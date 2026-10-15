"""Small, state-dictionary-free helpers for graph-model inference."""
from functools import lru_cache

import torch
from torch.nn import functional as F


def float32_precision_backends():
    """Precision controls of this PyTorch build, even after cuDNN's RNN module is imported.

    Some PyTorch releases expose ``fp32_precision`` on only a subset of these backends.
    """
    if not hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        return ()
    cudnn, mkldnn = torch.backends.cudnn, torch.backends.mkldnn
    # Importing cudnn.rnn shadows the precision object on the module instance.
    rnn = getattr(type(cudnn), 'rnn', None)
    if rnn is None:
        rnn = getattr(cudnn, 'rnn', None)
    candidates = (torch.backends, torch.backends.cuda.matmul, cudnn, mkldnn,
                  getattr(type(cudnn), 'conv', None), rnn, getattr(mkldnn, 'matmul', None))
    return tuple(backend for backend in candidates if hasattr(backend, 'fp32_precision'))


def float32_precision_token():
    """Read precision without mixing PyTorch's legacy and per-backend APIs."""
    backends = float32_precision_backends()
    if backends:
        return tuple(backend.fp32_precision for backend in backends)
    return torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32


def to_device(tensor, device):
    """Copy a host tensor to CUDA without a stream synchronization.

    Blocking host-to-device copies wait for all queued GPU work. Staging small
    index tensors in pinned memory lets the host continue issuing kernels.
    """
    device = torch.device(device)
    if device.type != 'cuda' or tensor.device.type != 'cpu':
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def host_ids(values):
    """Integer IDs as a list, using a host copy attached by the caller when present."""
    ids = getattr(values, '_dicee_ids', None)
    return values.tolist() if ids is None else ids


def inference_only(module):
    return (not module.training and not torch.is_autocast_enabled(next(module.parameters()).device.type)
            and (not torch.is_grad_enabled() or not any(p.requires_grad for p in module.parameters())))


def conv_update(states, update, weight, bias, norm_weight, norm_bias, eps, residual):
    value = F.linear(torch.cat((states, update), -1), weight, bias)
    value = F.layer_norm(value, (weight.shape[0],), norm_weight, norm_bias, eps).relu()
    return value + states if residual else value


@lru_cache(maxsize=1)
def compiled_conv_update():
    # A functional callable avoids changing checkpoint keys or retaining a model.
    # Python graph/cache management stays outside this CUDA-graph-compatible region.
    return torch.compile(conv_update, dynamic=False, mode='reduce-overhead')


def candidate_features(hidden, candidates):
    """Skip the identity gather for internally generated all-entity candidates."""
    if getattr(candidates, '_dicee_all_entities', False):
        return hidden
    return hidden.gather(1, candidates.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))


def candidate_slice(candidates, sl):
    result = candidates[sl]
    if getattr(candidates, '_dicee_all_entities', False):
        result._dicee_all_entities = True
    return result


def cached_relation_batch(values, ids):
    """Broadcast one cached relation across a same-relation inference batch.

    Beam expansions commonly share a relation. A zero-stride view avoids
    copying its full graph representation for every head; consumers must not
    mutate this view. Mixed-relation batches retain their original order.
    """
    first = values[ids[0]]
    if all(key == ids[0] for key in ids):
        return first.unsqueeze(0).expand(len(ids), *first.shape)
    return torch.stack([values[key] for key in ids])


def conditioned_linear(linear, hidden, query):
    """Apply W[h,q]+b while projecting the shared query only once."""
    dim = hidden.shape[-1]
    return (F.linear(hidden, linear.weight[:, :dim], linear.bias)
            + F.linear(query, linear.weight[:, dim:])[:, None])


def conditioned_score(mlp, hidden, query):
    value = conditioned_linear(mlp[0], hidden, query)
    for layer in mlp[1:]:
        value = layer(value)
    return value
