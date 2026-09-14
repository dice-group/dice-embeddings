"""Reusable loss for positive-first groups of negative samples."""
import math

import torch
from torch.nn import functional as F


def grouped_adversarial_bce(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
    """Balance the positive against a weighted negative group (ULTRA convention).

    Positive temperature uses detached softmax(logits / temperature); zero uses
    uniform negative weights. This intentionally differs from ordinary mean BCE.
    """
    if logits.ndim != 2 or logits.shape[1] < 2 or logits.shape != targets.shape:
        raise ValueError('Adversarial BCE requires [batch, positive + negatives] logits and targets')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('adversarial_temperature must be finite and nonnegative')
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if temperature > 0:
        weights = (logits[:, 1:].detach() / temperature).softmax(dim=-1)
    else:
        weights = torch.full_like(logits[:, 1:], 1 / (logits.shape[1] - 1))
    return ((losses[:, 0] + (weights * losses[:, 1:]).sum(-1)) / 2).mean()
