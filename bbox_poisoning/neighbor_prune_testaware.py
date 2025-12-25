from __future__ import annotations

import argparse
import collections
import math
import os
from typing import Dict, List, Sequence, Tuple, Set, Optional, Deque
from collections import deque

Triple = Tuple[str, str, str]  # (head, relation, tail)


def read_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                bad += 1
                continue
            h, r, t = parts
            triples.append((h, r, t))
    if bad:
        print(f"[warn] Skipped {bad} malformed lines (expected 3 tab-separated columns).")
    return triples


def write_triples(path: str, triples: Sequence[Triple]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def build_undirected_adj(triples: Sequence[Triple]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = collections.defaultdict(set)
    for h, _, t in triples:
        if h and t:
            adj[h].add(t)
            adj[t].add(h)
    return adj


def bfs_backup_path_exists(
    adj: Dict[str, Set[str]],
    u: str,
    v: str,
    k: int,
) -> bool:
    """
    Is there a path of length <= k between u and v, ignoring the direct u-v edge?
    """
    if u not in adj or v not in adj:
        return False
    if u == v:
        return True

    visited: Set[str] = {u}
    q: Deque[Tuple[str, int]] = deque([(u, 0)])

    while q:
        node, dist = q.popleft()
        if dist == k:
            continue
        for nbr in adj[node]:
            # skip the direct edge u-v / v-u as the first hop
            if (node == u and nbr == v) or (node == v and nbr == u and dist == 0):
                continue
            if nbr == v:
                return True
            if nbr not in visited:
                visited.add(nbr)
                q.append((nbr, dist + 1))

    return False


def collect_eval_entities(test_triples: Sequence[Triple]) -> Set[str]:
    ents: Set[str] = set()
    for h, _, t in test_triples:
        ents.add(h)
        ents.add(t)
    return ents


def compute_testaware_destructive_scores(
    train_triples: Sequence[Triple],
    adj: Dict[str, Set[str]],
    eval_entities: Set[str],
    backup_k: int,
) -> List[Tuple[float, int, str]]:
    """
    Test-aware, destructive structural scoring.

    Candidates:
      - Any train triple (h, r, t) where h or t is in eval_entities.

    Score (higher = more destructive to remove):
      score = 1/(1 + |N(u) ∩ N(v)|)
            + (no backup path ? 1 : 0)
            + 1/(1 + min(deg(u), deg(v)))
    """
    # precompute degrees
    deg = {n: len(neighs) for n, neighs in adj.items()}

    scored: List[Tuple[float, int, str]] = []

    for idx, (h, r, t) in enumerate(train_triples):
        u, v = h, t

        # test-aware filter: only edges touching test entities
        if u not in eval_entities and v not in eval_entities:
            continue

        if u not in adj or v not in adj:
            continue

        du = deg.get(u, 0)
        dv = deg.get(v, 0)
        if du <= 0 or dv <= 0:
            continue

        # local structure
        common_neighbors = len(adj[u].intersection(adj[v]))

        # backup path
        has_backup = False
        if backup_k > 0:
            has_backup = bfs_backup_path_exists(adj, u, v, backup_k)

        min_deg = min(du, dv)

        # destructive score
        score = 0.0
        score += 1.0 / (1.0 + float(common_neighbors))
        if not has_backup:
            score += 1.0
        score += 1.0 / (1.0 + float(min_deg))

        reason = (
            f"testaware_destructive"
            f"-rel:{r}-deg_u:{du}-deg_v:{dv}-common:{common_neighbors}-backup:{has_backup}"
        )

        scored.append((score, idx, reason))

    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Train triple file (head<TAB>rel<TAB>tail).")
    ap.add_argument("--test", required=True, help="Test triple file (head<TAB>rel<TAB>tail).")

    ap.add_argument("--remove-frac", type=float, default=0.1,
                    help="Fraction of candidate (test-aware) edges to remove.")
    ap.add_argument("--remove-n", type=int, default=None,
                    help="Fixed number of candidate edges to remove (overrides --remove-frac).")

    ap.add_argument("--backup-k", type=int, default=2,
                    help="Max path length for backup detection (0 disables backup term).")

    ap.add_argument("--out-prefix", default=None,
                    help="Prefix for outputs. Defaults to <train_basename>.testaware_destructive.")

    args = ap.parse_args()

    train_triples = read_triples(args.train)
    if not train_triples:
        raise SystemExit("No train triples loaded. Check input formatting (3 tab-separated columns).")

    test_triples = read_triples(args.test)
    if not test_triples:
        raise SystemExit("No test triples loaded. Check --test path / formatting.")

    eval_entities = collect_eval_entities(test_triples)
    print(f"Loaded {len(test_triples)} test triples, {len(eval_entities)} distinct test entities.")

    # undirected adjacency from train
    adj = build_undirected_adj(train_triples)

    # score test-aware candidate edges
    scored = compute_testaware_destructive_scores(
        train_triples=train_triples,
        adj=adj,
        eval_entities=eval_entities,
        backup_k=args.backup_k,
    )

    if not scored:
        print("No candidate edges touching test entities; nothing to prune.")
        return

    # sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    num_candidates = len(scored)
    if args.remove_n is not None:
        remove_n = max(0, min(args.remove_n, num_candidates))
    else:
        if not (0.0 <= args.remove_frac <= 1.0):
            raise ValueError(f"--remove-frac must be in [0,1], got {args.remove_frac}")
        remove_n = int(math.floor(args.remove_frac * num_candidates))

    if remove_n <= 0:
        print("remove_n <= 0; nothing will be pruned.")
        return

    selected = scored[:remove_n]

    removed_indices: Set[int] = {idx for (_score, idx, _reason) in selected}
    kept: List[Triple] = []
    removed: List[Triple] = []

    for i, tr in enumerate(train_triples):
        if i in removed_indices:
            removed.append(tr)
        else:
            kept.append(tr)

    base = os.path.splitext(os.path.basename(args.train))[0]
    if args.out_prefix:
        prefix = args.out_prefix
    else:
        if args.remove_n is not None:
            samp = f"n{args.remove_n}"
        else:
            samp = f"frac{args.remove_frac}"
        prefix = f"{base}.testaware_destructive_{samp}_bk{args.backup_k}"

    out_kept = prefix + "train.txt"
    out_removed = prefix + ".removed.txt"
    out_removed_ann = prefix + ".removed_annotated.tsv"
    out_report = prefix + ".report.txt"

    write_triples(out_kept, kept)
    write_triples(out_removed, removed)

    with open(out_removed_ann, "w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\treason\n")
        for score, idx, reason in selected:
            h, r, t = train_triples[idx]
            f.write(f"{h}\t{r}\t{t}\t{reason};score={score:.6g}\n")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY ===\n")
        f.write(f"train_triples={len(train_triples)}\n")
        f.write(f"test_triples={len(test_triples)}\n")
        f.write(f"eval_entities={len(eval_entities)}\n")
        f.write(f"candidate_edges_touching_test={num_candidates}\n")
        f.write(f"removed_candidates={remove_n}\n")
        f.write(f"kept_triples={len(kept)}\n")
        f.write(f"removed_triples={len(removed)}\n")

    print(f"Candidate edges touching test entities: {num_candidates}")
    print(f"Removed candidate edges: {remove_n}")
    print(f"Total removed: {len(removed)}   Total kept: {len(kept)}")
    print("Wrote:")
    print("  ", out_kept)
    print("  ", out_removed)
    print("  ", out_removed_ann)
    print("  ", out_report)


if __name__ == "__main__":
    main()
