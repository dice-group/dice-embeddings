"""Evaluation for continual KGE experiments (Section 11).

Reuses DICE's existing filtered-ranking primitives
(``dicee.evaluation._filtering.FilteredRanker``) directly on integer entity
ids -- continual experiments operate purely on integer-indexed triples (no
string round-trip needed), so the string-keyed convenience wrappers in
``dicee.evaluation.link_prediction`` are not used here, but the exact same
filtered-ranking algorithm is.

Two kinds of metrics are produced:

* standard filtered link-prediction metrics (MRR, Hits@1/3/10) on a held-out
  triple set, via :func:`evaluate_snapshot`;
* preservation metrics between two model checkpoints on a shared "stable"
  query set, via :func:`preservation_metrics` (``Delta_MRR_stable`` and
  ``ScoreDrift``, Section 11).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..evaluation._filtering import FilteredRanker
from ..evaluation.utils import ALL_HITS_RANGE, compute_metrics_from_ranks_simple, create_hits_dict, update_hits


@torch.no_grad()
def evaluate_snapshot(
    model,
    triples: np.ndarray,
    er_vocab: Dict[Tuple[int, int], List[int]],
    re_vocab: Dict[Tuple[int, int], List[int]],
    *,
    tie_policy: str = "sort",
    tie_seed: int = 0,
) -> Dict[str, float]:
    """Filtered head+tail link-prediction evaluation on integer-indexed triples.

    Parameters
    ----------
    model : torch.nn.Module
        A ``BaseKGE`` instance exposing ``forward_triples``.
    triples : np.ndarray
        ``(n, 3)`` int array of held-out ``(head, relation, tail)`` ids.
    er_vocab, re_vocab : dict
        ``(head, relation) -> [tails]`` / ``(relation, tail) -> [heads]``
        filtering vocabularies built from the *currently visible* graph
        (e.g. via ``dicee.static_preprocess_funcs.get_er_vocab``/``get_re_vocab``
        on the cumulative training triples at the current step, unioned
        with the evaluation split itself for standard filtered evaluation).

    Returns
    -------
    Dict[str, float]
        ``{"MRR", "H@1", "H@3", "H@10", "runtime_seconds"}``.
    """
    start = time.time()
    model.eval()
    ranker = FilteredRanker(tie_policy, tie_seed)
    num_entities = model.num_entities

    ranks: List[int] = []
    hits = create_hits_dict(ALL_HITS_RANGE)

    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        all_e = torch.arange(num_entities, dtype=torch.long)

        tail_x = torch.stack([
            torch.full_like(all_e, h), torch.full_like(all_e, r), all_e
        ], dim=1)
        tail_scores = model.forward_triples(tail_x)
        filt_tails = er_vocab.get((h, r), [])
        rank_tail = ranker.rank(tail_scores, t, filt_tails)

        head_x = torch.stack([
            all_e, torch.full_like(all_e, r), torch.full_like(all_e, t)
        ], dim=1)
        head_scores = model.forward_triples(head_x)
        filt_heads = re_vocab.get((r, t), [])
        rank_head = ranker.rank(head_scores, h, filt_heads)

        ranks.append(rank_tail)
        update_hits(hits, rank_tail, ALL_HITS_RANGE)
        ranks.append(rank_head)
        update_hits(hits, rank_head, ALL_HITS_RANGE)

    metrics = compute_metrics_from_ranks_simple(ranks, len(ranks), hits)
    metrics["runtime_seconds"] = time.time() - start
    return metrics


@torch.no_grad()
def preservation_metrics(
    model_new,
    model_old,
    stable_triples: np.ndarray,
    er_vocab: Dict[Tuple[int, int], List[int]],
    re_vocab: Dict[Tuple[int, int], List[int]],
) -> Dict[str, float]:
    """Preservation metrics on a shared stable-query set (Section 11).

    Returns
    -------
    Dict[str, float]
        ``MRR_new``, ``MRR_old``, ``Delta_MRR_stable`` (signed),
        ``abs_Delta_MRR_stable``, and ``ScoreDrift`` (mean absolute
        per-triple score difference between the two models).
    """
    if len(stable_triples) == 0:
        return {
            "MRR_new": float("nan"), "MRR_old": float("nan"),
            "Delta_MRR_stable": float("nan"), "abs_Delta_MRR_stable": float("nan"),
            "ScoreDrift": float("nan"), "num_stable_triples": 0,
        }

    metrics_new = evaluate_snapshot(model_new, stable_triples, er_vocab, re_vocab)
    metrics_old = evaluate_snapshot(model_old, stable_triples, er_vocab, re_vocab)

    model_new.eval()
    model_old.eval()
    x = torch.tensor(stable_triples, dtype=torch.long)
    f_new = model_new.forward_triples(x)
    f_old = model_old.forward_triples(x)
    score_drift = (f_new - f_old).abs().mean().item()

    delta = metrics_new["MRR"] - metrics_old["MRR"]
    return {
        "MRR_new": metrics_new["MRR"],
        "MRR_old": metrics_old["MRR"],
        "Delta_MRR_stable": delta,
        "abs_Delta_MRR_stable": abs(delta),
        "ScoreDrift": score_drift,
        "num_stable_triples": len(stable_triples),
    }
