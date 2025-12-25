#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kemeny-based deletion attack (top-K by Kemeny damage, no backup checks).

Workflow:
  1) Build undirected, unlabeled backbone A from triples (ignore self-loops).
  2) Factorize T = (1+r)D - A.
  3) Compute raw regularized Kemeny edge damage c_r(i,j) for all undirected edges.
  4) Sort edges by c_r desc and remove WHOLE head–tail groups (all triples between i and j)
     until the triple budget is reached. The last group may overshoot to ensure we meet budget.

Use:
  python kemeny_no_backup_attack.py \
      --input KGs/KINSHIP/train.txt \
      --budget 500 \
      --r 1e-8 \
      --jitter 1e-8 \
      --sep $'\\t' \
      --out-removed KINSHIP_removed.txt \
      --out-poisoned KINSHIP_poisoned.txt \
      --scores-csv KINSHIP_kemeny_scores.csv
"""

import argparse
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# -------------------- I/O --------------------

def read_triples(path, sep=None, comment_prefix="#"):
    triples = []
    raw_lines = []
    triple_line_idx = []
    entities = set()
    relations = set()

    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f):
            line = raw.rstrip("\n")
            raw_lines.append(line)
            s = line.strip()
            if not s:
                continue
            if comment_prefix and s.startswith(comment_prefix):
                continue
            parts = s.split(sep) if sep is not None else s.split()
            if len(parts) < 3:
                # skip malformed
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
            triple_line_idx.append(ln)
            entities.add(h); entities.add(t); relations.add(r)

    return triples, raw_lines, triple_line_idx, entities, relations


def save_lines(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


# -------------------- Indexing & graphs --------------------

def build_ids(entities, relations):
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def index_triples(triples, ent2id, rel2id):
    """
    Returns:
      triples_id: list of (hi, ri, ti)
      undirected_pairs: set of (i,j) with i<j
      pair_to_triple_idx: (i,j) -> list of triple indices
    """
    triples_id = []
    undirected_pairs = set()
    pair_to_triple_idx = defaultdict(list)

    for idx, (h, r, t) in enumerate(triples):
        hi = ent2id[h]; ti = ent2id[t]; ri = rel2id[r]
        triples_id.append((hi, ri, ti))
        if hi == ti:
            continue
        a, b = (hi, ti) if hi < ti else (ti, hi)
        undirected_pairs.add((a, b))
        pair_to_triple_idx[(a, b)].append(idx)

    return triples_id, undirected_pairs, pair_to_triple_idx


def build_A_undirected(n, undirected_pairs):
    if not undirected_pairs:
        raise RuntimeError("No undirected pairs found (only self-loops or empty file).")
    rows, cols, data = [], [], []
    for i, j in undirected_pairs:
        rows.extend([i, j]); cols.extend([j, i]); data.extend([1.0, 1.0])
    A = sp.csr_matrix((np.array(data, dtype=np.float64),
                       (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
                      shape=(n, n))
    A.sum_duplicates()
    return A


# -------------------- Kemeny centrality (raw) --------------------

def factorize_T(A, r, jitter=0.0):
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
    Raw regularized Kemeny damage c_r(i,j), no filtering.
    """
    n = A.shape[0]
    z = solve_T(d)
    gamma = float(d @ z + d.sum())

    scores = {}
    indptr = A.indptr; indices = A.indices
    v = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for p in range(indptr[i], indptr[i+1]):
            j = indices[p]
            if i < j:
                v.fill(0.0); v[i] = 1.0; v[j] = -1.0
                w = solve_T(v)
                delta = float(d @ w)
                x = w - (delta / gamma) * z
                alpha = float(x[i] - x[j])
                denom = 1.0 - alpha
                if abs(denom) < 1e-14:
                    c_r = 1e12
                else:
                    beta = float(np.dot(x * x, d))
                    c_r = beta / denom
                scores[(i, j)] = c_r
    return scores


# -------------------- Selection (top-K by Kemeny) --------------------

def select_edges_to_delete_topk(
    undirected_pairs,
    pair_to_triple_idx,
    scores,
    budget_triples,
    allow_overshoot=True,
):
    """
    Pick edges by descending Kemeny damage, removing WHOLE groups.
    If allow_overshoot=True, the last chosen group may exceed the budget to ensure we hit it.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    to_delete = []
    removed = 0
    for (i, j), _s in ranked:
        if (i, j) not in undirected_pairs:
            continue
        idxs = pair_to_triple_idx.get((i, j), [])
        if not idxs:
            continue
        g = len(idxs)
        if removed + g > budget_triples and not allow_overshoot:
            # skip groups that don't fit
            continue
        to_delete.extend(idxs)
        removed += g
        if removed >= budget_triples:
            break

    return to_delete


# -------------------- Pipeline --------------------

def run_attack(input_path,
               sep,
               budget,
               r_reg,
               jitter,
               out_removed,
               out_poisoned,
               scores_csv=None):

    triples, raw_lines, triple_line_idx, entities, relations = read_triples(input_path, sep=sep)
    if not triples:
        raise RuntimeError("No triples parsed from input. Check file and separator.")

    ent2id, rel2id = build_ids(entities, relations)
    triples_id, undirected_pairs, pair_to_triple_idx = index_triples(triples, ent2id, rel2id)
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
                gsz = len(pair_to_triple_idx.get((i, j), []))
                f.write(f"{id2ent[i]},{id2ent[j]},{s:.10g},{gsz}\n")

    print("[INFO] Selecting edges to delete (top-K by Kemeny, no backup checks) ...")
    idxs_to_delete = select_edges_to_delete_topk(
        undirected_pairs=undirected_pairs,
        pair_to_triple_idx=pair_to_triple_idx,
        scores=scores,
        budget_triples=budget,
        allow_overshoot=True,   # ensure we meet the triple budget
    )

    selected_lines = set(triple_line_idx[idx] for idx in idxs_to_delete)
    print(f"[INFO] Selected {len(selected_lines)} triples to delete (requested budget={budget}).")

    removed_lines = [raw_lines[ln] for ln in sorted(selected_lines)]
    save_lines(removed_lines, out_removed)

    kept_lines = [raw_lines[ln] for ln in range(len(raw_lines)) if ln not in selected_lines]
    save_lines(kept_lines, out_poisoned)

    print(f"[DONE] Removed triples written to: {out_removed}")
    print(f"[DONE] Poisoned train written to: {out_poisoned}")


def main():
    ap = argparse.ArgumentParser(
        description="Kemeny-based deletion attack: top-K by Kemeny damage (no backup checks)."
    )
    ap.add_argument("--input", required=True, help="Path to train triples file.")
    ap.add_argument("--sep", default=None, help="Field separator (default: any whitespace). For tab: --sep $'\\t'")
    ap.add_argument("--budget", type=int, required=True, help="Triple budget (count of triples to remove).")
    ap.add_argument("--r", type=float, default=1e-8, help="Regularization r in T = (1+r)D - A.")
    ap.add_argument("--jitter", type=float, default=0.0, help="Diagonal jitter eps added to T.")
    ap.add_argument("--out-removed", required=True, help="Where to write removed triples.")
    ap.add_argument("--out-poisoned", required=True, help="Where to write poisoned train.")
    ap.add_argument("--scores-csv", default=None, help="Optional CSV of edge scores.")
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
    )


if __name__ == "__main__":
    main()
