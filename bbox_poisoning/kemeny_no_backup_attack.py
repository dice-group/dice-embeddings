

import argparse
import sys
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from collections import defaultdict, Counter  

# -------------------- I/O --------------------

def read_triples(path: str, sep: str = None, comment_prefix: str = "#"):
    """
    Read triples from a text file.

    Each non-empty, non-comment line is split by `sep` (default: any whitespace).
    We assume: head sep relation sep tail [ignored extra fields...]

    Returns:
        triples          : list[(h, r, t)] as strings
        raw_lines        : list[str] all original lines (for line-based output)
        triple_line_idx  : list[int] original line index for each triple
        entities         : set[str]
        relations        : set[str]
    """
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
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
            triple_line_idx.append(ln)
            entities.add(h); entities.add(t)
            relations.add(r)

    return triples, raw_lines, triple_line_idx, entities, relations


def save_lines(lines: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def save_triples(triples: List[Tuple[str, str, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


# -------------------- Indexing & graphs --------------------

def build_ids(entities: Set[str], relations: Set[str]):
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def index_triples(triples: List[Tuple[str, str, str]],
                  ent2id: Dict[str, int],
                  rel2id: Dict[str, int]):
    """
    Build:
      - triples_id: list[(hi, ri, ti)] in ID space
      - undirected_pairs: set[(i,j)] with i<j
      - undirected_pair_to_indices: (i,j) -> list[triple_idx]
      - directed out-neighbors: out_neighbors[u] = set(v)
    """
    triples_id = []
    undirected_pairs: Set[Tuple[int, int]] = set()
    undirected_pair_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    out_neighbors: Dict[int, Set[int]] = defaultdict(set)

    for idx, (h, r, t) in enumerate(triples):
        hi = ent2id[h]
        ti = ent2id[t]
        ri = rel2id[r]
        triples_id.append((hi, ri, ti))

        # directed backup graph (unlabeled)
        if hi != ti:
            out_neighbors[hi].add(ti)

        # undirected scoring graph (unlabeled)
        if hi != ti:
            a, b = (hi, ti) if hi < ti else (ti, hi)
            undirected_pairs.add((a, b))
            undirected_pair_to_indices[(a, b)].append(idx)

    n = len(ent2id)
    out_list = [set() for _ in range(n)]
    for u, nbrs in out_neighbors.items():
        out_list[u] = set(nbrs)

    return triples_id, undirected_pairs, undirected_pair_to_indices, out_list


def build_A_undirected(n: int, undirected_pairs: Set[Tuple[int, int]]) -> sp.csr_matrix:
    """
    Build symmetric 0/1 adjacency matrix A for the undirected scoring graph.
    """
    if not undirected_pairs:
        raise RuntimeError("No undirected pairs found (only self-loops or empty file).")

    rows = []
    cols = []
    data = []
    for i, j in undirected_pairs:
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([1.0, 1.0])
    A = sp.csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(n, n),
    )
    A.sum_duplicates()
    return A


# -------------------- Kemeny centrality (raw) --------------------

def factorize_T(A: sp.csr_matrix, r: float, jitter: float = 0.0):
    """
    Build T = (1 + r) D - A and return a solver T^{-1} *.

    jitter: small epsilon added to the diagonal to avoid exact singularity
            when nodes have degree 0, etc.
    """
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    T = (1.0 + r) * sp.diags(d) - A
    if jitter > 0.0:
        T = T + jitter * sp.eye(n, format="csr")
    T = T.tocsc()

    try:
        solve_lu = spla.factorized(T)
        def solve_T(rhs: np.ndarray) -> np.ndarray:
            return solve_lu(rhs)
        return solve_T, d
    except Exception as e:
        # Fallback: use CG with optional ILU preconditioner
        sys.stderr.write(f"[WARN] LU factorization failed ({e}); falling back to CG.\n")
        try:
            ilu = spla.spilu(T)
            M = spla.LinearOperator(T.shape, matvec=lambda x: ilu.solve(x))
        except Exception:
            M = None

        def solve_T(rhs: np.ndarray) -> np.ndarray:
            x, info = spla.cg(T, rhs, M=M, atol=1e-10, rtol=1e-10, maxiter=5000)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info={info})")
            return x

        return solve_T, d


def kemeny_edge_scores_raw(A: sp.csr_matrix,
                           r: float,
                           solve_T,
                           d: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Compute *raw* regularized Kemeny edge centrality c_r(e) for each undirected edge.

    This follows the structure of Algorithm 6.2:
      - z = T^{-1} d
      - gamma = d^T z + d^T 1
      - For each edge (i,j):
          v = e_i - e_j
          w = T^{-1} v
          delta = d^T w
          x = w - (delta/gamma) z
          alpha = x[i] - x[j]
          denom = 1 - alpha
          beta = sum_l d_l x_l^2
          c_r(e) = beta / denom

    We do NOT apply the paper's "filtered" transformation; this is the
    direct Kemeny-damage measure you want: large c_r(e) ~ large increase
    of regularized Kemeny if e is removed.
    """
    n = A.shape[0]
    z = solve_T(d)
    gamma = float(d @ z + d.sum())

    scores: Dict[Tuple[int, int], float] = {}

    indptr = A.indptr
    indices = A.indices
    v = np.zeros(n, dtype=np.float64)

    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for p in range(row_start, row_end):
            j = indices[p]
            if i < j:
                v.fill(0.0)
                v[i] = 1.0
                v[j] = -1.0
                w = solve_T(v)
                delta = float(d @ w)
                x = w - (delta / gamma) * z
                alpha = float(x[i] - x[j])
                denom = 1.0 - alpha

                # handle numerical edge-case
                if abs(denom) < 1e-14:
                    # treat as extremely large damage (bridge-like)
                    c_r = 1e12
                else:
                    beta = float(np.dot(x * x, d))
                    c_r = beta / denom

                scores[(i, j)] = c_r

    return scores


# -------------------- Backup test (no redundancy) --------------------

def khop_reachable(src: int,
                   dst: int,
                   out_neighbors: List[Set[int]],
                   forbidden_pairs: Set[Tuple[int, int]],
                   k: int) -> bool:
    """
    Directed k-hop reachability from src to dst.
    We ignore any arc (u,v) in forbidden_pairs.
    """
    if src == dst:
        return True
    visited = {src}
    frontier = [src]
    depth = 0

    while frontier and depth < k:
        next_frontier = []
        for u in frontier:
            for v in out_neighbors[u]:
                if (u, v) in forbidden_pairs:
                    continue
                if v == dst:
                    return True
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
        depth += 1

    return False


def has_khop_backup_pair(i: int,
                         j: int,
                         out_neighbors: List[Set[int]],
                         k: int) -> bool:
    """
    Check whether there exists any directed k-hop backup between i and j
    AFTER removing the undirected edge {i,j}.

    That is, remove arcs (i->j) and (j->i) (if any) and then check:
      i -> j reachable? OR j -> i reachable?

    If either direction is reachable, we say the edge HAS backup.
    """
    forbidden = {(i, j), (j, i)}
    if khop_reachable(i, j, out_neighbors, forbidden, k):
        return True
    if khop_reachable(j, i, out_neighbors, forbidden, k):
        return True
    return False


# -------------------- Selection --------------------

def select_edges_to_delete(
    triples_id: List[Tuple[int, int, int]],
    undirected_pairs: Set[Tuple[int, int]],
    undirected_pair_to_indices: Dict[Tuple[int, int], List[int]],
    out_neighbors: List[Set[int]],
    scores: Dict[Tuple[int, int], float],
    budget_triples: int,
    k_backup: int,
) -> List[int]:
    """
    Select triples to delete according to:
      - edges must have NO k-hop backup (in directed graph) once removed
      - rank by raw Kemeny damage scores (descending)
      - remove whole pair groups (all triples between head and tail)
      - stop when triple budget is reached or list exhausted

    Returns:
      triple_indices_to_delete: list[int] indices in triples_id
    """
    # sort edges by Kemeny damage
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    to_delete_triple_idxs: List[int] = []
    remaining = budget_triples

    for (i, j), s in ranked:
        if remaining <= 0:
            break

        pair = (i, j)
        if pair not in undirected_pairs:
            continue

        idxs = undirected_pair_to_indices[pair]
        if not idxs:
            continue

        cost = len(idxs)
        if cost > remaining:
            # cannot afford to remove this whole pair group
            continue

        # Backup test: skip edges that still have a k-hop alternative
        if has_khop_backup_pair(i, j, out_neighbors, k_backup):
            continue

        # Accept this pair: delete all its triples
        to_delete_triple_idxs.extend(idxs)
        remaining -= cost

    return to_delete_triple_idxs

########################################################

def precompute_kemeny_pair_scores(DB, train_path):
    """Run Kemeny script once (budget=0) to get per-pair scores for this DB."""
    scores_root = RUNS_ROOT / "kemeny_scores" / DB
    scores_root.mkdir(parents=True, exist_ok=True)
    scores_csv = scores_root / "pair_scores.csv"

    if not scores_csv.exists():
        tmp_removed = scores_root / "tmp_removed.txt"
        tmp_poisoned = scores_root / "tmp_poisoned.txt"

        cmd = [
            "python",
            KEMENY_SCRIPT,
            "--input", str(train_path),
            "--budget", "0",
            "--k", "1",  # k for backup here is irrelevant since budget=0
            "--r", str(KEMENY_R),
            "--jitter", str(KEMENY_JITTER),
            "--sep", "\t",
            "--out-removed", str(tmp_removed),
            "--out-poisoned", str(tmp_poisoned),
            "--scores-csv", str(scores_csv),
        ]
        print(f"[Kemeny] Precomputing pair scores for {DB}...")
        subprocess.run(cmd, check=True)

    pair_scores = {}
    with scores_csv.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            h, t, score_str = parts[0], parts[1], parts[2]
            try:
                score = float(score_str)
            except ValueError:
                continue
            pair_scores[(h, t)] = score

    return pair_scores


def run_kemeny_addition(train_triples, pair_scores, budget, seed, max_neighbors=50):
    """Use Kemeny scores to add `budget` triples in structurally central places."""

    if budget <= 0:
        return [], list(train_triples)

    # node scores from pair scores
    node_score = defaultdict(float)
    for (h, t), s in pair_scores.items():
        node_score[h] += s
        node_score[t] += s

    if not node_score:
        return [], list(train_triples)

    # existing undirected pairs (head, tail) ignoring relation
    existing_pairs = set()
    for h, r, t in train_triples:
        if h == t:
            continue
        a, b = (h, t) if h < t else (t, h)
        existing_pairs.add((a, b))

    nodes = list(node_score.keys())
    nodes_sorted = sorted(nodes, key=lambda u: node_score[u], reverse=True)

    # candidate non-edges scored by s(u)+s(v)
    cand_scores = {}
    for u in nodes_sorted:
        local = 0
        for v in nodes_sorted:
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing_pairs or (a, b) in cand_scores:
                continue
            cand_scores[(a, b)] = node_score[u] + node_score[v]
            local += 1
            if local >= max_neighbors:
                break

    if not cand_scores:
        return [], list(train_triples)

    sorted_pairs = sorted(cand_scores.items(), key=lambda kv: kv[1], reverse=True)

    # relation distribution
    rel_counts = Counter(r for (h, r, t) in train_triples)
    if not rel_counts:
        return [], list(train_triples)

    rels, weights = zip(*rel_counts.items())
    rng = random.Random(seed)

    original_set = set(train_triples)
    added = []

    for (a, b), _score in sorted_pairs:
        if len(added) >= budget:
            break
        r = rng.choices(rels, weights=weights, k=1)[0]
        if rng.random() < 0.5:
            triple = (a, r, b)
        else:
            triple = (b, r, a)
        if triple in original_set or triple in added:
            continue
        added.append(triple)

    poisoned = list(train_triples) + added
    return added, poisoned


def save_added_and_eval(
    original_triples,
    added_triples,
    feature_tag,
    DB,
    top_k,
    experiment_idx,
    MODEL,
    experiment_seed,
    test_path,
    valid_path,
):
    out_dir = (
        SAVED_DATASETS_ROOT
        / DB
        / "add"
        / feature_tag
        / MODEL
        / str(top_k)
        / str(experiment_seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    new_train = list(original_triples) + list(added_triples)
    save_triples(new_train, str(out_dir / "train.txt"))

    (out_dir / "added.txt").write_text(
        "\n".join("\t".join(x) for x in added_triples),
        encoding="utf-8",
    )

    shutil.copy2(test_path, str(out_dir / "test.txt"))
    shutil.copy2(valid_path, str(out_dir / "valid.txt"))

    res = run_dicee_eval(
        dataset_folder=str(out_dir),
        model=MODEL,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        embedding_dim=EMB_DIM,
        loss_function=LOSS_FN,
        seed=experiment_seed,
        scoring_technique=SCORING_TECH,
        optim=OPTIM,
        path_to_store_single_run=str(
            RUNS_ROOT
            / f"add/{top_k}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}"
        ),
    )
    return res["Test"]["MRR"]

#################################################################

# -------------------- Main pipeline --------------------

def run_attack(input_path: str,
               sep: str,
               budget: int,
               k_backup: int,
               r_reg: float,
               jitter: float,
               out_removed: str,
               out_poisoned: str,
               scores_csv: str = None):

    # 1. Read triples
    triples, raw_lines, triple_line_idx, entities, relations = read_triples(input_path, sep=sep)
    if not triples:
        raise RuntimeError("No triples parsed from input. Check file and separator.")

    ent2id, rel2id = build_ids(entities, relations)
    triples_id, undirected_pairs, undirected_pair_to_indices, out_neighbors = \
        index_triples(triples, ent2id, rel2id)

    n = len(ent2id)
    print(f"[INFO] Loaded {len(triples)} triples, {n} entities, "
          f"{len(undirected_pairs)} undirected pairs.")

    # 2. Build undirected adjacency for Kemeny
    A = build_A_undirected(n, undirected_pairs)

    # 3. Factorize T and compute raw Kemeny edge scores
    print("[INFO] Factorizing T = (1+r)D - A ...")
    solve_T, d = factorize_T(A, r_reg, jitter=jitter)
    print("[INFO] Computing raw Kemeny edge scores ...")
    scores = kemeny_edge_scores_raw(A, r_reg, solve_T, d)

    # Optional: write scores CSV
    if scores_csv is not None:
        id2ent = {v: k for k, v in ent2id.items()}
        with open(scores_csv, "w", encoding="utf-8") as f:
            f.write("head,tail,score,group_size\n")
            for (i, j), s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                gsz = len(undirected_pair_to_indices.get((i, j), []))
                f.write(f"{id2ent[i]},{id2ent[j]},{s:.10g},{gsz}\n")

    # 4. Select edges to delete with NO k-hop backup
    print("[INFO] Selecting edges to delete (no backup, top Kemeny damage) ...")
    triple_idxs_to_delete = select_edges_to_delete(
        triples_id,
        undirected_pairs,
        undirected_pair_to_indices,
        out_neighbors,
        scores,
        budget_triples=budget,
        k_backup=k_backup,
    )

    selected_lines = set(triple_line_idx[idx] for idx in triple_idxs_to_delete)
    print(f"[INFO] Selected {len(selected_lines)} triples to delete (budget={budget}).")

    # 5. Write removed and poisoned outputs
    # Removed triples as lines
    removed_lines = [raw_lines[ln] for ln in sorted(selected_lines)]
    save_lines(removed_lines, out_removed)

    # Poisoned dataset = original minus removed lines
    kept_lines = [raw_lines[ln] for ln in range(len(raw_lines)) if ln not in selected_lines]
    save_lines(kept_lines, out_poisoned)

    print(f"[DONE] Removed triples written to: {out_removed}")
    print(f"[DONE] Poisoned train written to: {out_poisoned}")


def main():
    ap = argparse.ArgumentParser(
        description="Kemeny-based deletion attack: remove no-backup bottleneck edges."
    )
    ap.add_argument("--input", required=True,
                    help="Path to train triples file: head sep relation sep tail per line.")
    ap.add_argument("--sep", default=None,
                    help="Field separator (default: any whitespace). For tab: --sep $'\\t'")
    ap.add_argument("--budget", type=int, required=True,
                    help="Triple budget: total number of triples to remove "
                         "(removal is by undirected pair groups).")
    ap.add_argument("--k", type=int, default=2,
                    help="Backup radius (k-hop). We require NO path of length <= k between endpoints "
                         "after removing the edge.")
    ap.add_argument("--r", type=float, default=1e-8,
                    help="Regularization parameter r in T = (1+r)D - A.")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="Diagonal jitter epsilon added to T to avoid singular factorization.")
    ap.add_argument("--out-removed", required=True,
                    help="File to write removed triples (as original lines).")
    ap.add_argument("--out-poisoned", required=True,
                    help="File to write poisoned train (original minus removed).")
    ap.add_argument("--scores-csv", default=None,
                    help="Optional CSV to write per-edge Kemeny scores.")
    args = ap.parse_args()

    run_attack(
        input_path=args.input,
        sep=args.sep,
        budget=args.budget,
        k_backup=args.k,
        r_reg=args.r,
        jitter=args.jitter,
        out_removed=args.out_removed,
        out_poisoned=args.out_poisoned,
        scores_csv=args.scores_csv,
    )


if __name__ == "__main__":
    main()
