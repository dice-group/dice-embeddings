import argparse
import sys
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Set

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def read_triples(path: str, sep: str = None, comment_prefix: str = "#"):
    triples = []
    raw_lines = []
    triple_line_idx = []
    entities = set()
    relations = set()

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f):
            raw = line.rstrip("\n")
            raw_lines.append(raw)
            s = raw.strip()
            if not s or (comment_prefix and s.startswith(comment_prefix)):
                continue
            parts = s.split(sep) if sep is not None else s.split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
            triple_line_idx.append(ln)
            entities.add(h); entities.add(t)
            relations.add(r)
    return triples, raw_lines, triple_line_idx, entities, relations


def build_ids(entities: Set[str], relations: Set[str]):
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def index_triples(triples, ent2id, rel2id):
    triples_id = []
    undirected_pairs = set()
    undirected_pair_to_indices = defaultdict(list)
    dir_pair_counts = defaultdict(int)
    out_neighbors = defaultdict(set)

    for idx, (h, r, t) in enumerate(triples):
        i, j = ent2id[h], ent2id[t]
        if i != j:
            out_neighbors[i].add(j)
            dir_pair_counts[(i, j)] += 1
        a, b = (i, j) if i <= j else (j, i)
        if a != b:
            undirected_pairs.add((a, b))
            undirected_pair_to_indices[(a, b)].append(idx)
        triples_id.append((i, rel2id[r], j))

    n = len(ent2id)
    out_list = [set() for _ in range(n)]
    for u, nbrs in out_neighbors.items():
        out_list[u] = set(nbrs)
    return triples_id, undirected_pairs, undirected_pair_to_indices, out_list, dir_pair_counts


def build_A_und(n: int, undirected_pairs: Set[Tuple[int, int]]):
    rows = []
    cols = []
    data = []
    for (i, j) in undirected_pairs:
        rows.extend([i, j]); cols.extend([j, i]); data.extend([1.0, 1.0])
    A = sp.csr_matrix((np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))), shape=(n, n))
    A.sum_duplicates()
    return A

"""
def factorize_T(A_csr: sp.csr_matrix, r: float):
    d = np.asarray(A_csr.sum(axis=1)).ravel().astype(np.float64)
    T = ((1.0 + r) * sp.diags(d) - A_csr).tocsc()
    solve_T = spla.factorized(T) 
    return solve_T, d
"""

def factorize_T(A, r, jitter=0.0):

    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel()
    T = (1.0 + r) * sp.diags(d) - A
    T = T.tocsc()

    if jitter > 0.0:
        T = T + jitter * sp.eye(n, format="csc")

    solve_T = spla.factorized(T)
    return solve_T, d


def filtered_kemeny_scores(A_csr: sp.csr_matrix, r: float, solve_T, d: np.ndarray):
    n = A_csr.shape[0]
    inv_r = 1.0 / r
    cut_threshold = 0.5 * inv_r

    z = solve_T(d)
    gamma = float(d @ z + d.sum())

    scores = {}
    indptr, indices = A_csr.indptr, A_csr.indices

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
                    c_r = inv_r   
                else:
                    beta = float(np.dot(x * x, d))
                    c_r = beta / denom
                scores[(i, j)] = (inv_r - c_r) if (c_r > cut_threshold) else c_r
    return scores


def k_hop_reachable_directed(src: int, dst: int, out_neighbors: List[Set[int]], pair_counts: Dict[Tuple[int,int], int],
                             removed_pairs: Set[Tuple[int,int]], k: int):
    """Directed unlabeled reachability within k, ignoring edges in removed_pairs completely."""
    if src == dst:
        return True
    seen = {src}
    frontier = [src]
    for _ in range(k):
        if not frontier:
            return False
        nxt = []
        for u in frontier:
            for v in out_neighbors[u]:
                if (u, v) in removed_pairs:
                    continue
                if pair_counts.get((u, v), 0) <= 0:
                    continue
                if v == dst:
                    return True
                if v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt
    return False


def select_candidates(triples_id, undirected_pair_to_indices, scores,
                      out_neighbors, dir_pair_counts,
                      budget: int, k: int,
                      unit: str = "pair",
                      gate: str = "hard",
                      lam: float = 0.0):

    ranked_pairs = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    to_delete_idxs = []
    removed_pairs = set()
    remaining = budget

    if unit == "pair":
        # Greedy over pairs. Cost = group size.
        for (i, j), s in ranked_pairs:
            idxs = undirected_pair_to_indices[(i, j)]
            if len(idxs) == 0:
                continue
            cost = len(idxs)
            if cost > remaining:
                continue  # skip if cannot afford
            # evaluate gate
            ok = True
            if gate != "none":
                # simulate removal of (i,j) and (j,i)
                simulate = {(i, j), (j, i)}
                # if either direction exists, require no k-hop backup
                has_ij = any(triples_id[idx][0] == i and triples_id[idx][2] == j for idx in idxs)
                has_ji = any(triples_id[idx][0] == j and triples_id[idx][2] == i for idx in idxs)
                if gate == "hard":
                    if has_ij and k_hop_reachable_directed(i, j, out_neighbors, dir_pair_counts, simulate, k):
                        ok = False
                    if ok and has_ji and k_hop_reachable_directed(j, i, out_neighbors, dir_pair_counts, simulate, k):
                        ok = False
                elif gate == "soft":
                    penalty = 0.0
                    if has_ij and k_hop_reachable_directed(i, j, out_neighbors, dir_pair_counts, simulate, k):
                        penalty += lam
                    if has_ji and k_hop_reachable_directed(j, i, out_neighbors, dir_pair_counts, simulate, k):
                        penalty += lam
                    ok = (s - penalty) > 0
            if not ok:
                continue
            # accept: remove the whole pair group
            to_delete_idxs.extend(idxs)
            remaining -= cost
            removed_pairs.add((i, j)); removed_pairs.add((j, i))
            # update directed counts to reflect actual removals
            for idx in idxs:
                hi, _, ti = triples_id[idx]
                if (hi, ti) in dir_pair_counts:
                    dir_pair_counts[(hi, ti)] -= 1
        return to_delete_idxs

    elif unit == "triple":
        # Greedy over individual triples in pair-score order
        cand_triples = []
        for (i, j), s in ranked_pairs:
            cand_triples.extend(undirected_pair_to_indices[(i, j)])

        for idx in cand_triples:
            if remaining <= 0:
                break
            h, _, t = triples_id[idx]
            # evaluate gate
            ok = True
            if gate != "none":
                simulate = {(h, t)}
                if gate == "hard":
                    if k_hop_reachable_directed(h, t, out_neighbors, dir_pair_counts, simulate, k):
                        ok = False
                elif gate == "soft":
                    penalty = lam if k_hop_reachable_directed(h, t, out_neighbors, dir_pair_counts, simulate, k) else 0.0
                    ok = (scores.get((min(h,t), max(h,t)), 0.0) - penalty) > 0
            if not ok:
                continue
            # accept
            to_delete_idxs.append(idx)
            remaining -= 1
            # update directed counts
            if (h, t) in dir_pair_counts:
                dir_pair_counts[(h, t)] -= 1
        return to_delete_idxs

    else:
        raise ValueError("unit must be 'pair' or 'triple'")


def run_pipeline(input_path: str, budget: int, k: int, r: float,
                 unit: str, gate: str, lam: float,
                 sep: str = None,
                 out_path: str = None,
                 poisoned_out_path: str = None,
                 scores_csv: str = None, jitter = 0.0):
    triples, raw_lines, triple_line_idx, entities, relations = read_triples(input_path, sep=sep)
    if not triples:
        raise RuntimeError("No triples parsed from input. Check your file format.")

    ent2id, rel2id = build_ids(entities, relations)
    triples_id, undirected_pairs, undirected_pair_to_indices, out_neighbors, dir_pair_counts = index_triples(triples, ent2id, rel2id)
    n = len(ent2id)

    if len(undirected_pairs) == 0:
        raise RuntimeError("No undirected pairs found. Is your file empty or full of self-loops?")

    A = build_A_und(n, undirected_pairs)
    #solve_T, d = factorize_T(A, r)
    solve_T, d = factorize_T(A, r, jitter)

    scores = filtered_kemeny_scores(A, r, solve_T, d)

    to_delete_idxs = select_candidates(triples_id, undirected_pair_to_indices, scores,
                                       out_neighbors, dir_pair_counts,
                                       budget, k, unit=unit, gate=gate, lam=lam)

    selected_line_nos = set(triple_line_idx[idx] for idx in to_delete_idxs)
    selected_lines = [raw_lines[ln] for ln in sorted(selected_line_nos)]

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for s in selected_lines:
                f.write(s + "\n")

    if poisoned_out_path:
        with open(poisoned_out_path, "w", encoding="utf-8") as f:
            for ln, raw in enumerate(raw_lines):
                if ln not in selected_line_nos:
                    f.write(raw + "\n")

    if scores_csv:
        id2ent = {v:k for k,v in ent2id.items()}
        with open(scores_csv, "w", encoding="utf-8") as f:
            f.write("head,tail,score,group_size\n")
            for (i, j), s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                gsz = len(undirected_pair_to_indices[(i, j)])
                f.write(f"{id2ent[i]},{id2ent[j]},{s:.10g},{gsz}\n")

    return {
        "selected_count": len(selected_lines),
        "budget": budget,
        "n_entities": n,
        "n_pairs": len(undirected_pairs),
        "unit": unit, "gate": gate, "k": k, "r": r
    }


def main():
    ap = argparse.ArgumentParser(description="Poison KG with filtered Kemeny + backup-aware selection (fixed).")
    ap.add_argument("--input", required=True, help="Triples .txt (head [space/tab] relation [space/tab] tail per line).")
    ap.add_argument("--budget", type=int, required=True, help="Number of triples to delete (total). For --unit pair, budget must cover the whole pair group size.")
    ap.add_argument("--k", type=int, default=2, help="Backup radius in hops (default: 2).")
    ap.add_argument("--r", type=float, default=1e-8, help="Regularization r for scoring (default: 1e-8).")
    ap.add_argument("--unit", choices=["pair","triple"], default="pair", help="Delete by pair or by single triple (default: pair).")
    ap.add_argument("--gate", choices=["hard","soft","none"], default="hard", help="Backup gate mode (default: hard).")
    ap.add_argument("--lam", type=float, default=0.0, help="Penalty lambda for --gate soft.")
    ap.add_argument("--sep", default=None, help="Field separator (default: any whitespace).")
    ap.add_argument("--out", required=True, help="Write selected deletion lines here.")
    ap.add_argument("--poisoned-out", required=True, help="Write the poisoned dataset (original minus deletions).")
    ap.add_argument("--scores-csv", default=None, help="Optional CSV of pair scores.")
    ap.add_argument(
    "--jitter",
    type=float,
    default=0.0,
    help="Diagonal jitter epsilon added to T to avoid singular factorization."
    )

    args = ap.parse_args()

    info = run_pipeline(args.input, args.budget, args.k, args.r, args.unit, args.gate, args.lam,
                        sep=args.sep, out_path=args.out, poisoned_out_path=args.poisoned_out, scores_csv=args.scores_csv, jitter=args.jitter)
    print(f"Selected {info['selected_count']} lines (budget {info['budget']}). Unit={info['unit']} Gate={info['gate']} k={info['k']} r={info['r']}")
    if info['selected_count'] == 0:
        print("Hint: on dense sets like KINSHIP, use --unit pair and/or lower --k, or set --gate soft/none.")

if __name__ == "__main__":
    main()
