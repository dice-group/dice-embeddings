"""Bound attention workspace without splitting coupled query sequences."""

from types import FunctionType

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode


def _attention_forward(mode):
    function = F.multi_head_attention_forward
    scope = function.__globals__ | {
        'scaled_dot_product_attention': lambda *args, **kwargs: mode.__torch_function__(
            F.scaled_dot_product_attention, (), args, kwargs)}
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)


class BoundedAttention(TorchFunctionMode):
    def __init__(self, max_pairs=1024**2):
        self.max_pairs = max_pairs

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.multi_head_attention_forward:
            # The outer functional call suspends this mode inside its Python body.
            return _attention_forward(self)(*args, **kwargs)
        if func is F.scaled_dot_product_attention:
            query, key, value = args[:3]
            if query.is_cuda and query.shape[-3] > 1 and query.shape[-2] * key.shape[-2] > self.max_pairs:
                mask = args[3] if len(args) > 3 else kwargs.get('attn_mask')
                dropout = args[4] if len(args) > 4 else kwargs.get('dropout_p', 0)
                if mask is not None or dropout != 0 or kwargs.get('enable_gqa', False):
                    raise ValueError('Bounded inference attention requires no mask or dropout')
                return torch.cat([func(query[..., i:i+1, :, :], key[..., i:i+1, :, :], value[..., i:i+1, :, :], *args[3:], **kwargs)
                                  for i in range(query.shape[-3])], dim=-3)
        return func(*args, **kwargs)
