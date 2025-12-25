import os
import math
import shutil
import networkx as nx
from pathlib import Path
from collections import defaultdict

from config import (DBS, MODELS, RECIPRIOCAL, PERCENTAGES)

# ----------------- I/O -----------------
def load_triples(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [tuple(line.strip().split()[:3]) for line in f if line.strip()]

def save_triples(triples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

# ----------------- graph build -----------------
def build_digraph(triples):
    G = nx.DiGraph()
    for h, _, t in triples:
        if h != t and not G.has_edge(h, t):
            G.add_edge(h, t)
    return G

def pair_key(u, v):
    return (u, v) if u <= v else (v, u)

def triples_by_pair(triples):
    m = defaultdict(list)
    for h, r, t in triples:
        if h == t: 
            continue
        m[pair_key(h, t)].append((h, r, t))
    return m

# ----------------- Kemeny ranking -----------------
def kemeny_constant_lcc(Gud):
    if Gud.number_of_nodes() == 0:
        return float("nan")
    if nx.is_connected(Gud):
        return nx.kemeny_constant(Gud)
    cc = max(nx.connected_components(Gud), key=len)
    H = Gud.subgraph(cc).copy()
    return nx.kemeny_constant(H)

def rank_edges_by_kemeny(Gud, verbose=True):
    """Return (baseK, [((u,v), deltaK, is_bridge)]) sorted by descending damage."""
    H = Gud.copy()
    baseK = kemeny_constant_lcc(H)
    bridges = set(nx.bridges(H))

    ranked = []
    for u, v in list(H.edges()):
        a, b = pair_key(u, v)
        if (a, b) in bridges:
            ranked.append(((a, b), float('inf'), True))
            continue
        H.remove_edge(a, b)
        if not nx.is_connected(H):
            ranked.append(((a, b), float('inf'), True))
        else:
            K1 = kemeny_constant_lcc(H)
            ranked.append(((a, b), float(K1 - baseK), False))
        H.add_edge(a, b)

    ranked.sort(key=lambda kv: (math.inf if math.isinf(kv[1]) else kv[1]), reverse=True)
    if verbose:
        finite = [d for (_, d, _) in ranked if math.isfinite(d)]
        print(f"[Kemeny] base={baseK:.6g}, edges={Gud.number_of_edges()}, bridges={len(bridges)}, finite={len(finite)}")
    return baseK, ranked

# ----------------- selection under triple budget -----------------
def select_pairs_by_triple_budget(ranked_edges, pair2triples, triple_budget, allow_partial=False):
    """Greedily take whole pair-groups until the total #triples removed <= triple_budget."""
    remaining = int(triple_budget)
    selected_pairs, removed = [], []
    for (u, v), _delta, _is_bridge in ranked_edges:
        if remaining <= 0:
            break
        group = pair2triples.get(pair_key(u, v), [])
        if not group:
            continue
        if len(group) <= remaining:
            selected_pairs.append((u, v))
            removed.extend(group)
            remaining -= len(group)
        elif allow_partial:
            removed.extend(group[:remaining])
            remaining = 0
            break
        else:
            continue
    return selected_pairs, removed

# ----------------- build datasets -----------------
def build_kemeny_nx_datasets(db, recip, train_path, budgets_triples, out_root):
    """
    Creates folders:
      out_root/<recip>/<db>/tri_<B>/{train.txt, removed.txt, meta.txt}
    """
    triples = load_triples(train_path)
    Gd = build_digraph(triples)
    Gud = Gd.to_undirected(as_view=False)
    _baseK, ranked = rank_edges_by_kemeny(Gud, verbose=True)
    pair2trip = triples_by_pair(triples)

    base_out = Path(out_root) / recip / db
    built = {}

    for B in budgets_triples:
        sel_pairs, removed = select_pairs_by_triple_budget(ranked, pair2trip, B, allow_partial=False)
        removed_set = set(removed)
        kept = [t for t in triples if t not in removed_set]
        out_dir = base_out / f"tri_{B}"
        out_dir.mkdir(parents=True, exist_ok=True)

        save_triples(kept, str(out_dir / "train.txt"))
        save_triples(list(removed_set), str(out_dir / "removed.txt"))
        with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
            f.write(f"db={db}\nrequested_triple_budget={B}\n"
                    f"pairs_removed={len(sel_pairs)}\ntriples_removed={len(removed_set)}\n"
                    f"triples_kept={len(kept)}\noriginal_triples={len(triples)}\n")
        built[B] = str(out_dir)

        print(f"[{db}] budget={B}: pairs={len(sel_pairs)}, removed_triples={len(removed_set)}, kept={len(kept)} -> {out_dir}")

    return built

# ----------------- example CLI -----------------
if __name__ == "__main__":
    db = ["UMLS", "KINSHIP"]

    for db in DBS:
        TRIPLES_PATH = f"./KGs/{db}/train.txt"
        n_train = len(load_triples(TRIPLES_PATH))
        BUDGETS = [max(1, int(n_train * p)) for p in PERCENTAGES]

        OUT_ROOT = "./kemeny_nx_datasets"
        build_kemeny_nx_datasets(db, RECIPRIOCAL, TRIPLES_PATH, BUDGETS, OUT_ROOT)
