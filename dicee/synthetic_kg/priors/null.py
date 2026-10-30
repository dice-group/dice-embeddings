"""The structural null of arm N: degree-preserving edge swaps of a record world's complete truth (``s5_swap``, v2).

The swaps act between the latent world and its observation (``FamilySpec.null_swap_budget``); the observation, query
and negative draws keep their own streams, so a null world has the intact world's observed share and hidden count.
Within each relation, a round matches the movable facts at random and proposes to exchange the tails of every pair;
a proposal is rejected when it is a no-op, makes a self-loop, leaves the relation's domain or range, recreates a fact
of the relation, or collides with another proposal of the round. A relation stops after ``ceil(budget x m)``
accepted swaps, ``CAP x budget x m`` proposals or ``STALL`` idle rounds. A symmetric relation (its facts equal their
reverses) is swapped as undirected pairs with a second generator, after the directed draws that v1 made for it, so
that every directed relation stays as in v1. Preserved exactly: every relation's fact count, every entity's in- and
out-degree per relation, the permitted types, the self-loops and the symmetry of symmetric relations.
"""
import math
from typing import Any, Sequence

import numpy as np

from ..streams import NULL_VERSION, null_stream

CAP = 10.0
"""At most ``CAP x budget`` proposals per fact (``s5_swap.CAP``)."""

STALL = 3
"""Rounds in a row without an accepted swap that stop a relation (``s5_swap.STALL``)."""


def _member(keys: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
    """Element-wise membership of ``keys`` in the sorted unique ``sorted_keys``."""
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.searchsorted(sorted_keys, keys)
    found: np.ndarray = sorted_keys[np.minimum(pos, len(sorted_keys) - 1)] == keys
    return found


def _accept(k1: np.ndarray, k2: np.ndarray, ok: np.ndarray, room: int) -> np.ndarray:
    """The proposals that are valid and create two facts that no other proposal of the round creates, at most ``room``."""
    idx = np.flatnonzero(ok)
    if len(idx):
        new = np.concatenate([k1[idx], k2[idx]])
        _, inverse, counts = np.unique(new, return_inverse=True, return_counts=True)
        dup = counts[inverse] > 1
        idx = idx[~(dup[:len(idx)] | dup[len(idx):])]
        idx = idx[:room]
    return idx


def _swap_relation(H: np.ndarray, T: np.ndarray, n: int, rng: np.random.Generator, budget: float, dom_mask: np.ndarray,
                  ran_mask: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Directed swaps of one relation's facts ``(H, T)``; the heads stay in their slots (``s5_swap.swap_relation``).

    Returns:
        The new tails, the accepted swaps, and whether the budget was reached.
    """
    m = len(H)
    T = T.copy()
    target = int(math.ceil(budget * m))
    movable = np.flatnonzero(H != T)
    if target == 0 or len(movable) < 2:
        return T, 0, target == 0
    keys = np.sort(H * n + T)
    limit = CAP * budget * m
    accepted = attempts = idle = 0
    while accepted < target and attempts < limit and idle < STALL:
        perm = movable[rng.permutation(len(movable))]
        half = len(perm) // 2
        a, b = perm[:half], perm[half:2 * half]
        ha, ta, hb, tb = H[a], T[a], H[b], T[b]
        ok = (ha != hb) & (ta != tb) & (ha != tb) & (hb != ta)
        ok &= dom_mask[ha] & dom_mask[hb] & ran_mask[ta] & ran_mask[tb]
        k1, k2 = ha * n + tb, hb * n + ta
        ok &= ~_member(k1, keys) & ~_member(k2, keys)
        idx = _accept(k1, k2, ok, target - accepted)
        if len(idx):
            T[a[idx]] = tb[idx]
            T[b[idx]] = ta[idx]
            keys = np.sort(H * n + T)
            idle = 0
        else:
            idle += 1
        attempts += int(half)
        accepted += int(len(idx))
    return T, accepted, accepted >= target


def _swap_undirected(H: np.ndarray, T: np.ndarray, n: int, rng: np.random.Generator, budget: float,
                    mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Swaps of a symmetric relation as unordered pairs; each matched {a, b}, {c, d} proposes {a, c}, {b, d} or {a, d},
    {c, b} by a fair coin (``s5_swap.swap_undirected``).

    Returns:
        The new heads and tails (both directions of every pair, then the self-loops), the accepted swaps, and whether
        the budget was reached.
    """
    loop = H == T
    A, B = H[H < T].copy(), T[H < T].copy()
    m = int(len(A) + loop.sum())
    target = int(math.ceil(budget * m))
    accepted = 0
    if target > 0 and len(A) >= 2:
        keys = np.sort(A * n + B)
        limit = CAP * budget * m
        attempts = idle = 0
        while accepted < target and attempts < limit and idle < STALL:
            perm = rng.permutation(len(A))
            half = len(perm) // 2
            i, j = perm[:half], perm[half:2 * half]
            a, b, c, d = A[i], B[i], A[j], B[j]
            cross = rng.random(half) < 0.5
            x1, y1 = a, np.where(cross, c, d)
            x2, y2 = np.where(cross, b, c), np.where(cross, d, b)
            ok = (a != c) & (a != d) & (b != c) & (b != d)
            ok &= mask[a] & mask[b] & mask[c] & mask[d]
            lo1, hi1, lo2, hi2 = np.minimum(x1, y1), np.maximum(x1, y1), np.minimum(x2, y2), np.maximum(x2, y2)
            k1, k2 = lo1 * n + hi1, lo2 * n + hi2
            ok &= ~_member(k1, keys) & ~_member(k2, keys)
            idx = _accept(k1, k2, ok, target - accepted)
            if len(idx):
                A[i[idx]], B[i[idx]] = lo1[idx], hi1[idx]
                A[j[idx]], B[j[idx]] = lo2[idx], hi2[idx]
                keys = np.sort(A * n + B)
                idle = 0
            else:
                idle += 1
            attempts += int(half)
            accepted += int(len(idx))
    return np.concatenate([A, B, H[loop]]), np.concatenate([B, A, T[loop]]), accepted, accepted >= target


def swap_truth(facts: np.ndarray, domains: Sequence[np.ndarray], ranges: Sequence[np.ndarray], n: int, seed: int, index: int, family: str,
               budget: float) -> tuple[np.ndarray, dict[str, Any]]:
    """The null of a world's complete truth (``s5_swap.swap_truth``), with its own streams per (seed, index, family).

    Args:
        facts: Int64 ``[F, 3]`` sorted unique facts.
        domains: Per relation the permitted heads.
        ranges: Per relation the permitted tails.
        n: Entity count.
        seed: The run seed.
        index: The world's stream index.
        family: The generator name.
        budget: Accepted swaps per fact (per pair in a symmetric relation).

    Returns:
        The new facts (sorted unique, the same count) and the summary that the research added to the world's metadata
        (``swap_version``, ``swap_budget``, ``swap_moved_share``, ``swap_accepted_per_fact``, ``swap_reached_share``).

    Raises:
        ValueError: When the swaps created a duplicate fact.
    """
    facts = np.asarray(facts, dtype=np.int64).reshape(-1, 3)
    rng = np.random.default_rng(null_stream(family, seed, index, directed=True))
    urng = np.random.default_rng(null_stream(family, seed, index, directed=False))
    out = facts.copy()
    order = np.argsort(facts[:, 1], kind='stable')
    bounds = np.searchsorted(facts[order, 1], np.arange(len(domains) + 1))
    total = accepted = moved = reached = 0
    for r in range(len(domains)):
        slots = order[bounds[r]:bounds[r + 1]]
        H, T = facts[slots, 0], facts[slots, 2]
        dom_mask = np.zeros(n, dtype=bool)
        dom_mask[np.asarray(domains[r], dtype=np.int64)] = True
        ran_mask = np.zeros(n, dtype=bool)
        ran_mask[np.asarray(ranges[r], dtype=np.int64)] = True
        T2, count, done = _swap_relation(H, T, n, rng, budget, dom_mask, ran_mask)
        old_keys = np.unique(H * n + T)
        H2 = H
        if len(H) and np.array_equal(old_keys, np.unique(T * n + H)):   # symmetric: the directed draws above are discarded
            H2, T2, count, done = _swap_undirected(H, T, n, urng, budget, dom_mask & ran_mask)
            out[slots, 0] = H2
        out[slots, 2] = T2
        total += len(H)
        accepted += count
        moved += len(H) - len(np.intersect1d(old_keys, np.unique(H2 * n + T2), assume_unique=True))
        reached += len(H) if done else 0
    out = np.unique(out, axis=0)
    if len(out) != len(facts):
        raise ValueError('a swap created a duplicate fact')
    facts_total = max(1.0, float(total))
    summary = dict(swap_version=NULL_VERSION, swap_budget=float(budget), swap_moved_share=float(moved / facts_total),
                   swap_accepted_per_fact=accepted / facts_total, swap_reached_share=float(reached / facts_total))
    return out, summary
