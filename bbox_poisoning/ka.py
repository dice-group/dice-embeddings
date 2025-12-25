#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kemeny-based deletion (no backup checks):
- Build undirected, unlabeled backbone from triples.
- Factorize T = (1 + r) D - A.
- For every undirected edge (i<j), compute raw Kemeny edge damage c_r(i,j).
- Sort edges by c_r desc and delete WHOLE head–tail groups (all triples between head and tail)
  until the triple budget is reached. Optionally, allow partial deletion of the last group to
  hit the budget exactly.

Usage:
  python kemeny_attack.py \
      --input KGs/FB15k-237/train.txt \
      --budget 500 \
      --r 1e-8 \
      --jitter 1e-8 \
      --sep $'\\t' \
      --out-removed removed.txt \
      --out-poisoned poisoned_train.txt \
      [--scores-csv scores.csv] \
      [--allow-partial-groups]
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------- I/O ----------------

def read_triples(path, sep=None, comment_prefix="#"):
    triples, raw_lines, triple_line_idx = [], [], []
    entities, relations = set(), set()

    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            raw_lines.append(line)
            s = line.strip()
            if not s:
                continue
            if comment_prefix and s.startswith(comment_prefix):
                continue
            parts = s.split(sep) if sep is not None else s.split()
            if len(parts) < 3:
                # skip malformed lines
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
            triple_line_idx.append(ln - 1)
            entities.add(h); entities.add(t); relations.add(r)

    return triples, raw_lines, triple_line_idx, entities, relations


def save_lines(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


# ------------- Indexing & graph -------------

def build_ids(entities, relations):
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def index_triples(triples, ent2id, rel2id):
    """
    Returns:
      triples_id: [(hi, ri, ti)]
      undirected_pairs: set of (i,j) with i<j
      pair_to_indices: (i,j) -> list of triple indices
    """
    triples_id = []
    undirected_pairs = set()
    pair_to_indices = defaultdict(list)

    for idx, (h, r, t) in enumerate(triples):
        hi = ent2id[h]; ti = ent2id[t]; ri = rel2id[r]
        triples_id.append((hi, ri, ti))
        if hi == ti:
            continue  # ignore self-loops
        a, b = (hi, ti) if hi < ti else (ti, hi)
        undirected_pairs.add((a, b))
        pair_to_indices[(a, b)].append(idx)

    return triples_id, undirected_pairs, pair_to_indices


def build_A_undirected(n, undirected_pairs):
    if not undirected_pairs:
        raise RuntimeError("No undirected pairs found.")
    rows, cols = [], []
    for i, j in undirected_pairs:
        rows.extend([i, j]); cols.extend([j, i])
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.csr_matrix((data, (np.array(rows), np.array(cols))), shape=(n, n))
    A.sum_duplicates()
    return A


# ------------- Kemeny edge damage -------------

def factorize_T(A, r, jitter=0.0):
    """
    T = (1+r)D - A, with optional diagonal jitter to avoid exact singularities.
    Returns a solver function for T x = rhs, and the degree vector d.
    """
    d = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    T = (1.0 + r) * sp.diags(d) - A
    if jitter > 0.0:
        T = T + jitter * sp.eye(A.shape[0], format="csr")
    T = T.tocsc()
    try:
        solve_lu = spla.factorized(T)
        def solve_T(rhs):
            return solve_lu(rhs)
        return solve_T, d
    except Exception as e:
        sys.stderr.write(f"[WARN] LU factorization failed ({e}); falling back to CG.\n")
        try:
            ilu = spla.spilu(T)
            M = spla.LinearOperator(T.shape, matvec=lambda x: ilu.solve(x))
        except Exception:
            M = None
        def solve_T(rhs):
            x, info = spla.cg(T, rhs, M=M, atol=1e-10, rtol=1e-10, maxiter=5000)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info={info})")
            return x
        return solve_T, d


def kemeny_edge_scores_raw(A, r, solve_T, d):
    """
    Raw regularized Kemeny damage c_r(i,j) for each undirected edge (i<j).
    """
    n = A.shape[0]
    z = solve_T(d)
    gamma = float(d @ z + d.sum())

    scores = {}
    indptr, indices = A.indptr, A.indices
    v = np.zeros(n, dtype=np.float64)

    for i in range(n):
        row_start, row_end = indptr[i], indptr[i+1]
        for p in range(row_start, row_end):
            j = indices[p]
            if i < j:
                v.fill(0.0)
                v[i] = 1.0; v[j] = -1.0
                w = solve_T(v)
                delta = float(d @ w)
                x = w - (delta / gamma) * z
                alpha = float(x[i] - x[j])
                denom = 1.0 - alpha
                if abs(denom) < 1e-14:
                    c_r = 1e12  # near-bridge
                else:
                    beta = float(np.dot(x * x, d))
                    c_r = beta / denom
                scores[(i, j)] = c_r
    return scores


# ------------- Selection (no backup checks) -------------

def select_edges_by_kemeny_only(undirected_pairs, pair_to_indices, scores, budget_triples, allow_partial=False):
    """
    Sort pairs by Kemeny damage, remove whole pair groups until budget.
    If allow_partial=True, the last group can be partially removed to hit budget exactly.
    Returns list of triple indices to delete.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    to_delete = []
    remaining = int(budget_triples)

    for (i, j), _score in ranked:
        if remaining <= 0:
            break
        pair = (i, j)
        if pair not in undirected_pairs:
            continue
        idxs = pair_to_indices.get(pair, [])
        if not idxs:
            continue
        cost = len(idxs)
        if cost <= remaining:
            to_delete.extend(idxs)
            remaining -= cost
        elif allow_partial:
            to_delete.extend(idxs[:remaining])
            remaining = 0
            break
        else:
            # skip if full pair doesn't fit; set allow_partial=True to hit budget exactly
            continue

    return to_delete


# ------------- Pipeline -------------

def run_attack(input_path, sep, budget, r_reg, jitter, out_removed, out_poisoned, scores_csv=None, allow_partial=False):
    triples, raw_lines, triple_line_idx, entities, relations = read_triples(input_path, sep=sep)
    if not triples:
        raise RuntimeError("No valid triples parsed from input.")

    ent2id, rel2id = build_ids(entities, relations)
    triples_id, undirected_pairs, pair_to_indices = index_triples(triples, ent2id, rel2id)

    n = len(ent2id)
    print(f"[INFO] Loaded {len(triples)} triples, {n} entities, {len(undirected_pairs)} undirected pairs.")

    A = build_A_undirected(n, undirected_pairs)

    print("[INFO] Factorizing T = (1+r)D - A ...")
    solve_T, d = factorize_T(A, r_reg, jitter=jitter)
    print("[INFO] Computing raw Kemeny edge scores ...")
    scores = kemeny_edge_scores_raw(A, r_reg, solve_T, d)

    if scores_csv is not None:
        id2ent = {v: k for k, v in ent2id.items()}
        with open(scores_csv, "w", encoding="utf-8") as f:
            f.write("head,tail,score,group_size\n")
            for (i, j), s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                gsz = len(pair_to_indices.get((i, j), []))
                f.write(f"{id2ent[i]},{id2ent[j]},{s:.10g},{gsz}\n")

    if budget <= 0:
        # No deletions, just write original back (useful when only computing scores)
        save_lines([], out_removed)
        save_lines(raw_lines, out_poisoned)
        print("[INFO] Budget is 0: no deletions performed.")
        return

    print("[INFO] Selecting edges to delete (Kemeny-only ranking) ...")
    triple_idxs_to_delete = select_edges_by_kemeny_only(
        undirected_pairs,
        pair_to_indices,
        scores,
        budget_triples=budget,
        allow_partial=allow_partial,
    )

    selected_lines = set(triple_line_idx[idx] for idx in triple_idxs_to_delete)
    print(f"[INFO] Selected {len(selected_lines)} triples to delete (requested budget={budget}).")

    removed_lines = [raw_lines[ln] for ln in sorted(selected_lines)]
    kept_lines = [raw_lines[ln] for ln in range(len(raw_lines)) if ln not in selected_lines]

    save_lines(removed_lines, out_removed)
    save_lines(kept_lines, out_poisoned)

    print(f"[DONE] Removed triples written to: {out_removed}")
    print(f"[DONE] Poisoned train written to: {out_poisoned}")


def main():
    ap = argparse.ArgumentParser(
        description="Kemeny-based deletion: rank edges by damage, delete top until budget. No backup checks."
    )
    ap.add_argument("--input", required=True, help="Path to train triples file.")
    ap.add_argument("--sep", default=None, help="Field separator (default: any whitespace). For tab: --sep $'\\t'")
    ap.add_argument("--budget", type=int, required=True, help="Triple budget (delete by undirected head–tail groups).")
    ap.add_argument("--r", type=float, default=1e-8, help="Regularization r in T = (1+r)D - A.")
    ap.add_argument("--jitter", type=float, default=0.0, help="Small diagonal added to T.")
    ap.add_argument("--out-removed", required=True, help="Removed triples (original lines).")
    ap.add_argument("--out-poisoned", required=True, help="Poisoned train (original minus removed).")
    ap.add_argument("--scores-csv", default=None, help="Optional CSV of per-pair Kemeny scores.")
    ap.add_argument("--allow-partial-groups", action="store_true",
                    help="If set, partially delete last pair to hit budget exactly.")
    args = ap.parse_args()

    run_attack(
        input_path=args.input,
        sep=args.sep,
        budget=args.budget,
        r_reg=args.r,
        jitter=args.jitter,
        out_removed=args.out_removed,
        out_poisoned=args.out_poisoned,
        scores_csv=args.scores_csv,
        allow_partial=args.allow_partial_groups,
    )


if __name__ == "__main__":
    main()
