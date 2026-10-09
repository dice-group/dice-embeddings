"""Validity checks that share one host synchronization per query.

Reading a CUDA boolean on the host waits for every queued kernel. Inside
``deferred_checks`` such conditions are collected and read together when the
block exits; the first failing check raises its original error. Outside the
block, ``require`` checks immediately, so other callers keep eager errors.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_PENDING = ContextVar('dicee_pending_checks', default=None)


def require(condition, error):
    """Raise ``error`` unless the scalar ``condition`` holds."""
    pending = _PENDING.get()
    if pending is None or not isinstance(condition, torch.Tensor) or condition.device.type == 'cpu':
        if not bool(condition):
            raise error
    else:
        pending.append((condition.reshape(()), error))


def _raise_first(pending):
    if pending:
        device = pending[0][0].device
        flags = torch.stack([condition.to(device) for condition, _ in pending]).tolist()
        for passed, (_, error) in zip(flags, pending):
            if not passed:
                raise error


@contextmanager
def deferred_checks():
    """Collect device conditions; an earlier failed check takes precedence over later errors."""
    pending = []
    token = _PENDING.set(pending)
    try:
        yield
    except Exception:
        # Interrupts and exits propagate unchanged.
        _PENDING.reset(token)
        token = None
        _raise_first(pending)
        raise
    finally:
        if token is not None:
            _PENDING.reset(token)
    _raise_first(pending)
