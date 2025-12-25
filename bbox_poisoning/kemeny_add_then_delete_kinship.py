#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

import networkx as nx

# ----------------------------- I/O ---------------------------------

def load_triples(path):
    with open(path, "r", encoding="utf-8") as f:
        return [tuple(line.strip().split()[:3]) for line in f if line.strip()]

def save_triples(triples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

# -------------------------- helpers --------------------------------

def pair_key(u, v):
    return (u, v) if u <= v else (v, u)

def build_digraph_from_triples(triples):
    G = nx.DiGraph()
    for h, _, t in triples:
        if h != t and not G.has_edge(h, t):
            G.add_edge(h, t, origin="orig")
    return G

def build_undirected_with_origin(Gd):
    Gud = nx.Graph()
    for u, v, data in Gd.edges(data=True):
        a, b = pair_key(u, v)
        if Gud.has_edge(a, b):
            if data.get("origin", "orig") == "orig":
                Gud[a][b]["origin"] = "orig"
        else:
            Gud.add_edge(a, b, origin=data.get("origin", "orig"))
    return Gud

def triples_by_pair(triples):
    m = defaultdict(list)
    for h, r, t in triples:
        if h == t:
            continue
        m[pair_key(h, t)].append((h, r, t))
    return m

def find_communities(Gud):
    comms = list(nx.algorithms.community.greedy_modularity_communities(Gud)) if Gud.number_of_edges() else [set(Gud.nodes())]
    node2comm = {}
    for cid, nodes in enumerate(comms):
        for u in nodes:
            node2comm[u] = cid
    return comms, node2comm

def kemeny_constant_lcc(Gud):
    if Gud.number_of_nodes() == 0:
        return float("nan")
    if nx.is_connected(Gud):
        return nx.kemeny_constant(Gud)
    cc = max(nx.connected_components(Gud), key=len)
    H = Gud.subgraph(cc).copy()
    return nx.kemeny_constant(H)

def rank_edges_by_kemeny(Gud_aug, exclude_pairs=None, candidate_filter=None, verbose=True):
    exclude_pairs = set() if exclude_pairs is None else set(exclude_pairs)
    H = Gud_aug.copy()
    baseK = kemeny_constant_lcc(H)

    bridges = set(nx.bridges(H))
    ranked = []
    for u, v in list(H.edges()):
        a, b = pair_key(u, v)
        if (a, b) in exclude_pairs:
            continue
        if candidate_filter is not None and not candidate_filter(u, v):
            continue
        if (a, b) in bridges:
            ranked.append(((a, b), float("inf"), True))
            continue
        H.remove_edge(a, b)
        if not nx.is_connected(H):
            dK = float("inf")
            ranked.append(((a, b), dK, True))
        else:
            K1 = kemeny_constant_lcc(H)
            dK = float(K1 - baseK)
            ranked.append(((a, b), dK, False))
        H.add_edge(a, b, **Gud_aug[a][b])

    ranked.sort(key=lambda kv: (math.isinf(kv[1]), kv[1]), reverse=True)
    if verbose:
        finite = [d for (_, d, _) in ranked if math.isfinite(d)]
        print(f"[Kemeny] base={baseK:.6g}, candidates={len(ranked)}, finite={len(finite)}")
    return baseK, ranked

def add_intra_community_edges(
    train_triples, Gud_mutable, node2comm, add_budget_triples, rng,
    per_pair_cap=3  # allow up to 3 new triples per intra pair
):
    if add_budget_triples <= 0:
        return [], set()

    rel_counts = Counter(r for (_, r, _) in train_triples)
    if not rel_counts:
        return [], set()

    triple_seen = set(train_triples)             # forbid exact duplicates only
    pair_added_count = defaultdict(int)          # throttle per pair
    added_triples, added_pairs = [], set()

    # group nodes by community
    comm2nodes = defaultdict(list)
    for u, cid in node2comm.items():
        comm2nodes[cid].append(u)

    rels, w = zip(*rel_counts.items())
    for cid, nodes in comm2nodes.items():
        nodes = list(nodes); rng.shuffle(nodes)
        for u in nodes:
            vs = nodes[:] ; rng.shuffle(vs)
            for v in vs:
                if u == v: continue
                pk = pair_key(u, v)
                # only intra-community
                if node2comm.get(u, -1) != node2comm.get(v, -1): 
                    continue
                # cap per pair
                if pair_added_count[pk] >= per_pair_cap:
                    continue
                # sample a relation and direction
                r = rng.choices(rels, weights=w, k=1)[0]
                triple = (u, r, v) if rng.random() < 0.5 else (v, r, u)
                if triple in triple_seen:
                    continue  # exact duplicate
                # accept
                added_triples.append(triple)
                triple_seen.add(triple)
                pair_added_count[pk] += 1
                added_pairs.add(pk)
                if Gud_mutable.has_edge(*pk):
                    # keep existing attrs; this is an extra triple on the same pair
                    pass
                else:
                    Gud_mutable.add_edge(pk[0], pk[1], origin="added")
                if len(added_triples) >= add_budget_triples:
                    return added_triples, added_pairs

    return added_triples, added_pairs


def select_pairs_by_triple_budget(ranked_edges, pair2orig_triples, triple_budget, allow_partial=False):
    remaining = int(triple_budget)
    selected, removed = [], []
    for (u, v), _delta, _is_bridge in ranked_edges:
        if remaining <= 0:
            break
        group = pair2orig_triples.get(pair_key(u, v), [])
        if not group:
            continue
        if len(group) <= remaining:
            selected.append((u, v))
            removed.extend(group)
            remaining -= len(group)
        elif allow_partial:
            removed.extend(group[:remaining])
            remaining = 0
            break
        else:
            continue
    return selected, removed

def undirected_stats(triples):
    nodes = set()
    pairs = set()
    for h, _, t in triples:
        if h == t:
            continue
        nodes.add(h); nodes.add(t)
        pairs.add(pair_key(h, t))
    G = nx.Graph()
    G.add_edges_from(pairs)
    comps = list(nx.connected_components(G)) if G.number_of_nodes() else []
    cc_sizes = sorted((len(c) for c in comps), reverse=True)
    return {
        "nodes": len(nodes),
        "pairs": len(pairs),
        "components": len(comps),
        "largest_cc": (cc_sizes[0] if cc_sizes else 0),
    }

def relation_counts(triples):
    return Counter(r for _, r, _ in triples)

def resolve_budget(n_total, pct, count, name):
    """
    If pct is not None, use it; supports 0.02 or 2 (interpreted as 2%).
    Otherwise use count. Clamp to [0, n_total].
    """
    if pct is not None:
        p = float(pct)
        if p > 1.0:  # interpret e.g. 2 as 2%
            p = p / 100.0
        b = int(round(n_total * p))
        b = max(0, min(b, n_total))
        return b, p
    if count is None:
        return 0, None
    b = max(0, min(int(count), n_total))
    return b, None

# ------------------------------ MAIN --------------------------------

def main():
    ap = argparse.ArgumentParser("KINSHIP: Add intra-community edges, then Kemeny-delete original edges; budgets as % or counts.")
    ap.add_argument("--kgdir", default="./KGs/KINSHIP", help="Folder with train.txt, valid.txt, test.txt")
    ap.add_argument("--out_root", default="./kemeny_add_then_delete_out", help="Where to write datasets")
    # Percentage budgets (use 0.02 for 2%, or 2 for 2%)
    ap.add_argument("--add_pct", type=float, default=0.02, help="Additions as fraction/percent of original train (e.g., 0.02 or 2)")
    ap.add_argument("--del_pct", type=float, default=0.02, help="Deletions as fraction/percent of original train (e.g., 0.02 or 2)")
    # Fallback counts (used only if pct not given)
    ap.add_argument("--add_budget", type=int, default=None, help="Triples to ADD if --add_pct not given")
    ap.add_argument("--del_budget", type=int, default=None, help="Triples to DELETE if --del_pct not given")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--intercommunity_only", action="store_true", help="Delete only cross-community original edges")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    kgdir = Path(args.kgdir)
    train_path = kgdir / "train.txt"
    valid_path = kgdir / "valid.txt"
    test_path  = kgdir / "test.txt"

    # ---------- Load ----------
    train_triples = load_triples(str(train_path))
    n_train = len(train_triples)
    print(f"[load] train={n_train} triples")

    # Resolve budgets from pct or counts
    add_budget, add_frac = resolve_budget(n_train, args.add_pct, args.add_budget, "add")
    del_budget, del_frac = resolve_budget(n_train, args.del_pct, args.del_budget, "del")
    print(f"[budgets] add: requested={'%'+str(add_frac*100) if add_frac is not None else args.add_budget} -> {add_budget} triples")
    print(f"[budgets] del: requested={'%'+str(del_frac*100) if del_frac is not None else args.del_budget} -> {del_budget} triples")

    # ---------- Original graphs & communities ----------
    Gd_orig = build_digraph_from_triples(train_triples)
    Gud_orig = build_undirected_with_origin(Gd_orig)
    comms, node2comm = find_communities(Gud_orig)
    print(f"[comm] communities={len(comms)} (greedy modularity)")

    # ---------- ADDITIONS (intra only) ----------
    added_triples, added_pairs = add_intra_community_edges(
        train_triples, Gud_orig, node2comm, add_budget_triples=add_budget, rng=rng
    )
    print(f"[add] added_triples={len(added_triples)} added_pairs={len(added_pairs)}")

    # Augmented train
    train_after_add = list(train_triples) + list(added_triples)

    # Build augmented undirected graph (mark added vs orig)
    Gd_aug = nx.DiGraph()
    for h, _, t in train_triples:
        if h != t and not Gd_aug.has_edge(h, t):
            Gd_aug.add_edge(h, t, origin="orig")
    for h, r, t in added_triples:
        if h != t and not Gd_aug.has_edge(h, t):
            Gd_aug.add_edge(h, t, origin="added")
    Gud_aug = build_undirected_with_origin(Gd_aug)

    # ---------- Deletion candidates ----------
    pair2orig = triples_by_pair(train_triples)  # delete only original
    exclude_pairs = set(added_pairs)            # never delete additions

    if args.intercommunity_only:
        def cand_filter(u, v):
            return node2comm.get(u, -1) != node2comm.get(v, -1) and pair_key(u, v) in pair2orig
    else:
        def cand_filter(u, v):
            return pair_key(u, v) in pair2orig

    # ---------- Rank by Kemeny on augmented graph ----------
    baseK_aug, ranked = rank_edges_by_kemeny(Gud_aug, exclude_pairs=exclude_pairs, candidate_filter=cand_filter, verbose=True)
    pair2dK = {pair_key(u, v): dK for ((u, v), dK, _br) in ranked}

    # ---------- Select deletions under triple budget ----------
    selected_pairs, removed_triples = select_pairs_by_triple_budget(
        ranked, pair2orig, triple_budget=del_budget, allow_partial=False
    )
    print(f"[delete] selected_pairs={len(selected_pairs)} removed_triples={len(removed_triples)}")

    # ---------- Final kept train ----------
    removed_set = set(removed_triples)
    added_set   = set(added_triples)
    original_set = set(train_triples)

    # keep all added; remove selected ORIGINAL triples
    final_train = [t for t in train_after_add if not (t in removed_set and t in original_set)]

    # ---------- Save datasets ----------
    add_tag = f"{int(round(add_frac*100))}pct" if add_frac is not None else f"{add_budget}"
    del_tag = f"{int(round(del_frac*100))}pct" if del_frac is not None else f"{del_budget}"

    out_base = Path(args.out_root) / "KINSHIP" / f"add_{add_tag}" / f"del_{del_tag}"
    add_dir = out_base / "addition"
    del_dir = out_base / "deletion"
    add_dir.mkdir(parents=True, exist_ok=True)
    del_dir.mkdir(parents=True, exist_ok=True)

    save_triples(added_triples,           str(add_dir / "added.txt"))
    save_triples(train_after_add,         str(add_dir / "train_after_addition.txt"))
    save_triples(removed_triples,         str(del_dir / "removed.txt"))       # removed (original only)
    save_triples(final_train,             str(del_dir / "train.txt"))         # evaluate this

    # copy val/test for convenience
    valid_path = Path(args.kgdir) / "valid.txt"
    test_path  = Path(args.kgdir) / "test.txt"
    if valid_path.exists():
        with open(valid_path, "r", encoding="utf-8") as fin, open(del_dir / "valid.txt", "w", encoding="utf-8") as fout:
            fout.write(fin.read())
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as fin, open(del_dir / "test.txt", "w", encoding="utf-8") as fout:
            fout.write(fin.read())

    # ---------- META: rich diagnostics ----------
    def undirected_stats(trs):
        nodes = set()
        pairs = set()
        for h, _, t in trs:
            if h == t: continue
            nodes.add(h); nodes.add(t)
            pairs.add(pair_key(h, t))
        G = nx.Graph(); G.add_edges_from(pairs)
        comps = list(nx.connected_components(G)) if G.number_of_nodes() else []
        cc_sizes = sorted((len(c) for c in comps), reverse=True)
        return {"nodes": len(nodes), "pairs": len(pairs), "components": len(comps), "largest_cc": (cc_sizes[0] if cc_sizes else 0)}

    def relation_counts(trs):
        return Counter(r for _, r, _ in trs)

    final_set = set(final_train)
    orig_minus_final = original_set - final_set      # removed originals
    final_minus_orig = final_set - original_set      # additions present in final
    sym_diff = len(orig_minus_final) + len(final_minus_orig)

    stats_orig = undirected_stats(train_triples)
    stats_add  = undirected_stats(train_after_add)
    stats_fin  = undirected_stats(final_train)

    rel_orig  = relation_counts(train_triples)
    rel_final = relation_counts(final_train)
    all_rel = set(rel_orig) | set(rel_final)
    rel_deltas = sorted(((r, rel_final.get(r,0)-rel_orig.get(r,0)) for r in all_rel),
                        key=lambda x: abs(x[1]), reverse=True)[:10]

    dK_vals = [pair2dK.get(pair_key(u, v), float("nan")) for (u, v) in selected_pairs]
    finite_dK = [x for x in dK_vals if math.isfinite(x)]
    dK_summary = {
        "count": len(dK_vals),
        "finite_count": len(finite_dK),
        "min": (min(finite_dK) if finite_dK else "NA"),
        "max": (max(finite_dK) if finite_dK else "NA"),
        "mean": (sum(finite_dK)/len(finite_dK) if finite_dK else "NA"),
    }

    meta_path = out_base / "meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("=== KINSHIP add-then-delete (Kemeny on augmented, delete ORIGINAL only) ===\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"intercommunity_only={args.intercommunity_only}\n\n")

        f.write("## Budgets (requested vs resolved, based on ORIGINAL train size)\n")
        f.write(f"n_original_triples={n_train}\n")
        f.write(f"requested_add_pct={add_frac if add_frac is not None else 'NA'}\n")
        f.write(f"requested_del_pct={del_frac if del_frac is not None else 'NA'}\n")
        f.write(f"resolved_add_budget={add_budget}\n")
        f.write(f"resolved_del_budget={del_budget}\n")
        f.write(f"actual_added_triples={len(added_set)}\n")
        f.write(f"actual_removed_triples={len(set(removed_triples))}\n\n")

        f.write("## Dataset sizes (triples)\n")
        f.write(f"original_train={len(original_set)}\n")
        f.write(f"train_after_addition={len(train_after_add)}\n")
        f.write(f"final_train_after_deletion={len(final_set)}\n\n")

        f.write("## Differences between ORIGINAL and FINAL (triples)\n")
        f.write(f"original_minus_final={len(orig_minus_final)}  # present in original, absent in final (removed)\n")
        f.write(f"final_minus_original={len(final_minus_orig)}  # new in final (added retained)\n")
        f.write(f"symmetric_difference={sym_diff}\n\n")

        f.write("## Undirected backbone stats (nodes/pairs/components/largest_cc)\n")
        f.write(f"original: nodes={stats_orig['nodes']}, pairs={stats_orig['pairs']}, components={stats_orig['components']}, largest_cc={stats_orig['largest_cc']}\n")
        f.write(f"after_add: nodes={stats_add['nodes']}, pairs={stats_add['pairs']}, components={stats_add['components']}, largest_cc={stats_add['largest_cc']}\n")
        f.write(f"final:     nodes={stats_fin['nodes']}, pairs={stats_fin['pairs']}, components={stats_fin['components']}, largest_cc={stats_fin['largest_cc']}\n\n")

        f.write("## Kemeny\n")
        f.write(f"baseK_on_augmented_graph={baseK_aug}\n")
        f.write(f"selected_pairs={len(selected_pairs)}\n")
        f.write(f"deltaK_summary(count={dK_summary['count']}, finite={dK_summary['finite_count']}, "
                f"min={dK_summary['min']}, mean={dK_summary['mean']}, max={dK_summary['max']})\n\n")

        f.write("## Top 10 relation count deltas (final - original)\n")
        for r, d in rel_deltas:
            f.write(f"{r}\t{d}\n")

    print("\n[done]")
    print(f"  Added triples:            {add_dir / 'added.txt'}")
    print(f"  Train after addition:     {add_dir / 'train_after_addition.txt'}")
    print(f"  Removed (original only):  {del_dir / 'removed.txt'}")
    print(f"  Final train to evaluate:  {del_dir / 'train.txt'}")
    print(f"  Meta with diffs:          {meta_path}")
    print(f"  Valid/Test copied into:   {del_dir}")

if __name__ == "__main__":
    main()
