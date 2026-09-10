"""Small, state-dictionary-free helpers for graph-model inference."""
from functools import lru_cache

import torch
from torch.nn import functional as F


def float32_precision_token():
    """Read precision without mixing PyTorch's legacy and per-backend APIs."""
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        # Include parent settings as well as operator overrides. The legacy
        # getters can raise when backends (or cuDNN conv/RNN) use different modes.
        cudnn, mkldnn = torch.backends.cudnn, torch.backends.mkldnn
        backends = (torch.backends, torch.backends.cuda.matmul, cudnn, mkldnn,
                    getattr(cudnn, 'conv'), getattr(cudnn, 'rnn'), getattr(mkldnn, 'matmul'))
        return tuple(getattr(backend, 'fp32_precision') for backend in backends)
    return torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32


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
