"""Utility functions for evaluation module.

This module contains shared helper functions used across different
evaluation components.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm

# Standard hit levels for link prediction evaluation
DEFAULT_HITS_RANGE: List[int] = [1, 3, 10]
ALL_HITS_RANGE: List[int] = list(range(1, 11))


def make_iterable_verbose(
    iterable_object: Iterable,
    verbose: bool,
    desc: str = "Default",
    position: Optional[int] = None,
    leave: bool = True
) -> Iterable:
    """Wrap an iterable with tqdm progress bar if verbose is True.

    Args:
        iterable_object: The iterable to potentially wrap.
        verbose: Whether to show progress bar.
        desc: Description for the progress bar.
        position: Position of the progress bar.
        leave: Whether to leave the progress bar after completion.

    Returns:
        The original iterable or a tqdm-wrapped version.
    """
    if verbose:
        return tqdm(iterable_object, desc=desc, position=position, leave=leave)
    return iterable_object


def compute_metrics_from_ranks(
    ranks: List[int],
    num_triples: int,
    hits_dict: Dict[int, List[float]],
    scale_factor: int = 1
) -> Dict[str, float]:
    """Compute standard link prediction metrics from ranks.

    Args:
        ranks: List of ranks for each prediction.
        num_triples: Total number of triples evaluated.
        hits_dict: Dictionary mapping hit levels to lists of hits.
        scale_factor: Factor to scale the denominator (e.g., 2 for head+tail).

    Returns:
        Dictionary containing H@1, H@3, H@10, and MRR metrics.
    """
    divisor = float(num_triples * scale_factor)

    hit_1 = sum(hits_dict.get(1, [])) / divisor if 1 in hits_dict else 0.0
    hit_3 = sum(hits_dict.get(3, [])) / divisor if 3 in hits_dict else 0.0
    hit_10 = sum(hits_dict.get(10, [])) / divisor if 10 in hits_dict else 0.0
    mean_reciprocal_rank = sum(1.0 / r for r in ranks) / divisor

    return {
        'H@1': hit_1,
        'H@3': hit_3,
        'H@10': hit_10,
        'MRR': mean_reciprocal_rank
    }


def compute_metrics_from_ranks_simple(
    ranks: List[int],
    num_triples: int,
    hits_dict: Dict[int, List[float]]
) -> Dict[str, float]:
    """Compute link prediction metrics without scaling factor.

    Args:
        ranks: List of ranks for each prediction.
        num_triples: Total number of triples evaluated.
        hits_dict: Dictionary mapping hit levels to lists of hits.

    Returns:
        Dictionary containing H@1, H@3, H@10, and MRR metrics.
    """
    hit_1 = sum(hits_dict.get(1, [])) / num_triples if 1 in hits_dict else 0.0
    hit_3 = sum(hits_dict.get(3, [])) / num_triples if 3 in hits_dict else 0.0
    hit_10 = sum(hits_dict.get(10, [])) / num_triples if 10 in hits_dict else 0.0
    mean_reciprocal_rank = np.mean(1.0 / np.array(ranks))

    return {
        'H@1': hit_1,
        'H@3': hit_3,
        'H@10': hit_10,
        'MRR': mean_reciprocal_rank
    }


def update_hits(
    hits: Dict[int, List[float]],
    rank: int,
    hits_range: Optional[List[int]] = None
) -> None:
    """Update hits dictionary based on rank.

    Args:
        hits: Dictionary to update in-place.
        rank: The rank to check against hit levels.
        hits_range: List of hit levels to check (default: ALL_HITS_RANGE).
    """
    if hits_range is None:
        hits_range = ALL_HITS_RANGE

    for hits_level in hits_range:
        if rank <= hits_level:
            hits[hits_level].append(1.0)


def create_hits_dict(hits_range: Optional[List[int]] = None) -> Dict[int, List[float]]:
    """Create an initialized hits dictionary.

    Args:
        hits_range: List of hit levels to initialize (default: ALL_HITS_RANGE).

    Returns:
        Dictionary with empty lists for each hit level.
    """
    if hits_range is None:
        hits_range = ALL_HITS_RANGE
    return {i: [] for i in hits_range}


def efficient_zero_grad(model) -> None:
    """Efficiently zero gradients using parameter.grad = None.

    This is more efficient than optimizer.zero_grad() as it avoids
    memory operations.

    See: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

    Args:
        model: PyTorch model to zero gradients for.
    """
    for param in model.parameters():
        param.grad = None
