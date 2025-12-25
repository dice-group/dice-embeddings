from __future__ import annotations

import argparse
import collections
import math
import os
import random
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


def unordered_pair(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def compute_pair_counts(triples: Sequence[Triple]) -> collections.Counter:
    c = collections.Counter()
    for h, _, t in triples:
        c[unordered_pair(h, t)] += 1
    return c


def compute_nonredundant_mask(triples: Sequence[Triple], pair_counts: collections.Counter) -> List[bool]:
    return [pair_counts[unordered_pair(h, t)] == 1 for (h, _, t) in triples]


def build_incoming_index(triples: Sequence[Triple], mask: Optional[Sequence[bool]] = None) -> Dict[str, List[int]]:
    inc: Dict[str, List[int]] = collections.defaultdict(list)
    for i, (_, _, t) in enumerate(triples):
        if mask is None or mask[i]:
            inc[t].append(i)
    return inc


def pick_hubs_by_incoming_nonredundant(
    inc_nr: Dict[str, List[int]],
    top_k: int,
    min_incoming_nr: int,
) -> List[Tuple[str, int]]:
    scored = [(node, len(idxs)) for node, idxs in inc_nr.items() if len(idxs) >= min_incoming_nr]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hub_hub_edge_indices(triples: Sequence[Triple], hub_set: Set[str]) -> List[int]:
    return [i for i, (h, _, t) in enumerate(triples) if h in hub_set and t in hub_set]


def triple_str(tr: Triple) -> str:
    h, r, t = tr
    return f"{h}\t{r}\t{t}"


def greedy_pick_disconnected_hubs(
    triples: Sequence[Triple],
    candidate_hubs: List[str],
    final_k: int,
) -> List[str]:
    """
    Greedy heuristic:
      - start with the best-scoring hub (first in candidate_hubs)
      - repeatedly add the hub that adds the fewest hub-hub edges to the current set
        (ties broken by original candidate order)
    """
    if final_k >= len(candidate_hubs):
        return candidate_hubs[:]

    cand_order = {h: i for i, h in enumerate(candidate_hubs)}
    selected: List[str] = [candidate_hubs[0]]
    selected_set = {selected[0]}

    # Precompute hub-hub adjacency counts between candidates
    pair_count = collections.Counter()
    cand_set = set(candidate_hubs)
    for (h, _, t) in triples:
        if h in cand_set and t in cand_set and h != t:
            pair_count[unordered_pair(h, t)] += 1

    while len(selected) < final_k:
        best = None
        best_added = None
        for h in candidate_hubs:
            if h in selected_set:
                continue
            added = 0
            for s in selected:
                added += pair_count.get(unordered_pair(h, s), 0)
            if best is None or added < best_added or (added == best_added and cand_order[h] < cand_order[best]):
                best = h
                best_added = added
        selected.append(best)
        selected_set.add(best)

    return selected


# ---------- NEW STUFF FOR NEIGHBOR CONNECTIVITY ----------

def build_undirected_adj(triples: Sequence[Triple]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = collections.defaultdict(set)
    for h, _, t in triples:
        if h and t:
            adj[h].add(t)
            adj[t].add(h)
    return adj


def neighbors_of_hubs(adj: Dict[str, Set[str]], hubs: Set[str]) -> Set[str]:
    neigh: Set[str] = set()
    for h in hubs:
        neigh.update(adj.get(h, set()))
    # hubs themselves are separate; we only want true neighbors here
    return neigh


def bfs_backup_path_exists(
    adj: Dict[str, Set[str]],
    u: str,
    v: str,
    k: int,
) -> bool:
    """
    Check if there's a path of length <= k between u and v in the undirected
    graph adj, *excluding* the direct edge u-v as the first step.

    This approximates "is there a backup path besides the direct link?".
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


def compute_neighbor_deg_stats(
    adj: Dict[str, Set[str]],
    hubs: Set[str],
    neighbors: Set[str],
) -> Dict[str, Dict[str, float]]:
    """
    Simple degree stats for neighbors of each hub in a given undirected adj.
    """
    stats: Dict[str, Dict[str, float]] = {}
    # map hub -> its neighbor set (in this graph)
    hub_to_neigh: Dict[str, Set[str]] = {h: set() for h in hubs}
    for h in hubs:
        hub_to_neigh[h] = set(adj.get(h, set())) & neighbors

    for h in hubs:
        ns = sorted(n for n in hub_to_neigh[h] if n in adj)
        if not ns:
            stats[h] = {
                "num_neighbors": 0.0,
                "avg_deg": 0.0,
                "median_deg": 0.0,
                "num_deg1_or_less": 0.0,
            }
            continue
        degs = [len(adj[n]) for n in ns]
        degs_sorted = sorted(degs)
        m = len(degs_sorted)
        if m % 2 == 1:
            median = float(degs_sorted[m // 2])
        else:
            median = 0.5 * (degs_sorted[m // 2 - 1] + degs_sorted[m // 2])
        num_deg1 = sum(1 for d in degs if d <= 1)
        stats[h] = {
            "num_neighbors": float(len(ns)),
            "avg_deg": float(sum(degs) / len(degs)),
            "median_deg": float(median),
            "num_deg1_or_less": float(num_deg1),
        }
    return stats


def compute_edge_scores_for_neighbor_damage(
    triples: Sequence[Triple],
    adj: Dict[str, Set[str]],
    hubs: Set[str],      # kept for consistency, but not used to restrict candidates anymore
    backup_k: int,
    use_backup: bool,
) -> List[Tuple[float, int, str]]:
    """
    Score edges (triples) by how much they hurt connectivity,
    now applied *globally* (no restriction to hubs/neighbors).

    Candidates:
      - any triple (h, r, t) where both endpoints exist in adj.

    Score:
      base = 1 / min(deg(u), deg(v))
      if use_backup:
        multiply base by 2 if NO backup path of length <= backup_k exists (excluding direct u-v).
    """
    # Precompute degrees
    deg = {n: len(neighs) for n, neighs in adj.items()}

    scored: List[Tuple[float, int, str]] = []

    for idx, (h, r, t) in enumerate(triples):
        u, v = h, t

        # GLOBAL: do not restrict to hubs or neighbors
        if u not in adj or v not in adj:
            continue

        du = deg.get(u, 0)
        dv = deg.get(v, 0)
        if du <= 0 or dv <= 0:
            continue

        base = 1.0 / float(min(du, dv))

        if use_backup and backup_k > 0:
            has_backup = bfs_backup_path_exists(adj, u, v, backup_k)
            if has_backup:
                score = base
                reason = f"candidate_low_deg_with_backup-deg_u:{du}-deg_v:{dv}"
            else:
                score = 2.0 * base
                reason = f"candidate_low_deg_no_backup-deg_u:{du}-deg_v:{dv}"
        else:
            score = base
            reason = "candidate_low_deg"

        scored.append((score, idx, reason))

    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input triple file (head<TAB>rel<TAB>tail).")

    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of hub CANDIDATES (ranked by incoming non-redundant edges).")
    ap.add_argument("--min-incoming-nr", type=int, default=1,
                    help="Minimum incoming non-redundant edges for a node to be considered a hub candidate.")

    ap.add_argument("--final-hubs", type=int, default=None,
                    help="If set (< top-k), pick this many hubs from the candidates using a greedy 'min hub-hub edges' heuristic.")

    ap.add_argument("--remove-frac", type=float, default=0.1,
                    help="Fraction of candidate edges (now global) to remove.")
    ap.add_argument("--remove-n", type=int, default=None,
                    help="Fixed number of candidate edges to remove globally (overrides --remove-frac).")

    ap.add_argument("--backup-k", type=int, default=3,
                    help="Max path length for backup detection (0 disables backup scoring).")
    ap.add_argument("--no-backup-signal", action="store_true",
                    help="Ignore backup paths and only use degree-based scoring.")

    ap.add_argument("--seed", type=int, default=42, help="Random seed (only used if you later add tie-breaking randomness).")

    ap.add_argument("--out-prefix", default=None,
                    help="Prefix for outputs. Defaults to <input_basename> with parameters.")

    args = ap.parse_args()

    triples = read_triples(args.input)
    if not triples:
        raise SystemExit("No triples loaded. Check input formatting (3 tab-separated columns).")

    # Standard pipeline to detect hubs by incoming non-redundant edges
    pair_counts = compute_pair_counts(triples)
    nr_mask = compute_nonredundant_mask(triples, pair_counts)

    inc_all = build_incoming_index(triples, mask=None)
    inc_nr = build_incoming_index(triples, mask=nr_mask)

    hubs_scored = pick_hubs_by_incoming_nonredundant(
        inc_nr, top_k=args.top_k, min_incoming_nr=args.min_incoming_nr
    )
    hub_candidates = [h for h, _ in hubs_scored]

    final_k = args.final_hubs
    if final_k is None or final_k >= len(hub_candidates):
        hubs = hub_candidates
    else:
        if final_k <= 0:
            hubs = []
        else:
            hubs = greedy_pick_disconnected_hubs(triples, hub_candidates, final_k=final_k)

    hub_set = set(hubs)

    if not hub_set:
        print("No hubs selected; nothing to prune.")
        return

    # Undirected adjacency for structural connectivity
    adj = build_undirected_adj(triples)

    # Neighbor sets in the ORIGINAL graph (still for reporting / stats)
    neighbors = neighbors_of_hubs(adj, hub_set)

    # Degree stats BEFORE pruning (around hubs)
    before_stats = compute_neighbor_deg_stats(adj, hub_set, neighbors)

    # Score candidate edges for connectivity impact (now GLOBAL)
    use_backup = not args.no_backup_signal
    scored = compute_edge_scores_for_neighbor_damage(
        triples=triples,
        adj=adj,
        hubs=hub_set,
        backup_k=args.backup_k,
        use_backup=use_backup,
    )

    if not scored:
        print("No candidate edges found; nothing to prune.")
        return

    # Sort by damage score (descending)
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

    # Indices selected for removal
    selected = scored[:remove_n]

    removed_indices: Set[int] = {idx for (_score, idx, _reason) in selected}
    reason_for_idx: Dict[int, str] = {idx: reason for (_score, idx, reason) in selected}

    kept: List[Triple] = []
    removed: List[Triple] = []
    for i, tr in enumerate(triples):
        if i in removed_indices:
            removed.append(tr)
        else:
            kept.append(tr)

    # Build new adjacency for AFTER pruning stats (still around hubs)
    adj_after = build_undirected_adj(kept)
    after_stats = compute_neighbor_deg_stats(adj_after, hub_set, neighbors)

    base = os.path.splitext(os.path.basename(args.input))[0]
    if args.out_prefix:
        prefix = args.out_prefix
    else:
        if args.remove_n is not None:
            samp = f"n{args.remove_n}"
        else:
            samp = f"frac{args.remove_frac}"
        final_tag = f"final{len(hubs)}" if (args.final_hubs is not None and args.final_hubs < args.top_k) else f"top{args.top_k}"
        bk_tag = f"bk{args.backup_k}" if use_backup and args.backup_k > 0 else "bk0"
        # you had this commented; leaving it commented if you want custom prefix
        # prefix = f"{base}.neighborprune_{final_tag}_{samp}_{bk_tag}_seed{args.seed}"
        #prefix = base + "."  # minimal default so your filenames aren't broken

    out_kept = prefix + "rain.txt"
    out_removed = prefix + ".removed.txt"
    out_removed_ann = prefix + ".removed_annotated.tsv"
    out_report = prefix + ".report.txt"

    write_triples(out_kept, kept)
    write_triples(out_removed, removed)

    with open(out_removed_ann, "w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\treason\n")
        for score, idx, reason in selected:
            h, r, t = triples[idx]
            f.write(f"{h}\t{r}\t{t}\t{reason};score={score:.6g}\n")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("=== HUBS (ranked by incoming non-redundant edges) ===\n")
        f.write("hub\tincoming_all\tincoming_nonredundant\n")
        for hub, _score in hubs_scored:
            if hub not in hub_set:
                continue
            f.write(
                f"{hub}\t{len(inc_all.get(hub, []))}\t{len(inc_nr.get(hub, []))}\n"
            )

        f.write("\n=== SELECTED HUBS ===\n")
        for hub in hubs:
            f.write(f"{hub}\n")

        f.write("\n=== NEIGHBOR DEGREE STATS BEFORE / AFTER PRUNING ===\n")
        f.write("hub\tphase\tnum_neighbors\tavg_deg\tmedian_deg\tnum_deg1_or_less\n")
        for hub in hubs:
            b = before_stats.get(hub, {})
            a = after_stats.get(hub, {})
            f.write(
                f"{hub}\tbefore\t{b.get('num_neighbors', 0.0)}\t"
                f"{b.get('avg_deg', 0.0)}\t{b.get('median_deg', 0.0)}\t"
                f"{b.get('num_deg1_or_less', 0.0)}\n"
            )
            f.write(
                f"{hub}\tafter\t{a.get('num_neighbors', 0.0)}\t"
                f"{a.get('avg_deg', 0.0)}\t{a.get('median_deg', 0.0)}\t"
                f"{a.get('num_deg1_or_less', 0.0)}\n"
            )

        f.write("\n=== REMOVAL SUMMARY ===\n")
        f.write(f"input_triples={len(triples)}\n")
        f.write(f"kept_triples={len(kept)}\n")
        f.write(f"removed_triples={len(removed)}\n")
        f.write(f"candidate_edges={num_candidates}\n")
        f.write(f"removed_candidates={remove_n}\n")

    # Console summary
    print("Selected hubs:")
    for hub in hubs:
        print(
            f"  {hub}  in_all={len(inc_all.get(hub, []))}  "
            f"in_nonredundant={len(inc_nr.get(hub, []))}"
        )
    print(f"Candidate edges (global): {num_candidates}")
    print(f"Removed candidate edges: {remove_n}")
    print(f"Total removed: {len(removed)}   Total kept: {len(kept)}")
    print("Wrote:")
    print("  ", out_kept)
    print("  ", out_removed)
    print("  ", out_removed_ann)
    print("  ", out_report)


if __name__ == "__main__":
    main()
