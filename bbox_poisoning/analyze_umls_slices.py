#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, random, argparse
from collections import defaultdict, Counter
import numpy as np
import networkx as nx

# ---------- I/O ----------
def load_triples(path, sep=None):
    tri = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = s.split(sep) if sep is not None else s.split()
            if len(parts) < 3: continue
            h, r, t = parts[0], parts[1], parts[2]
            tri.append((h, r, t))
    return tri

def pair_key(u, v): return (u, v) if u <= v else (v, u)

# ---------- Graphs ----------
def G_dir(triples):
    G = nx.DiGraph()
    for h, _, t in triples:
        if h != t and not G.has_edge(h, t):
            G.add_edge(h, t)
    return G

def G_und(triples):
    G = nx.Graph()
    for h, _, t in triples:
        if h == t: continue
        a, b = pair_key(h, t)
        if not G.has_edge(a, b): G.add_edge(a, b)
    return G

def triples_by_pair(triples):
    m = defaultdict(list)
    for h, r, t in triples:
        if h == t: continue
        m[pair_key(h,t)].append((h, r, t))
    return m

def relsets_by_pair(triples):
    m = defaultdict(set)
    for h, r, t in triples:
        if h == t: continue
        m[pair_key(h,t)].add(r)
    return m

# ---------- Metrics ----------
def density_und(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n <= 1: return 0.0
    return m / (n * (n - 1) / 2)

def backup_fraction_und(G, k=2, sample_edges=4000, seed=0):
    if G.number_of_edges() == 0: return float("nan")
    rng = random.Random(seed)
    edges = list(G.edges())
    if len(edges) > sample_edges: edges = rng.sample(edges, sample_edges)

    def khop(u, v, forb, kmax):
        if u == v: return True
        visited = {u}; frontier = [u]; d = 0
        while frontier and d < kmax:
            nxt = []
            for x in frontier:
                for y in G.neighbors(x):
                    if (x == forb[0] and y == forb[1]) or (x == forb[1] and y == forb[0]): continue
                    if y == v: return True
                    if y not in visited:
                        visited.add(y); nxt.append(y)
            frontier = nxt; d += 1
        return False

    backed = 0
    for u, v in edges:
        if not G.has_edge(u, v): continue
        G.remove_edge(u, v)
        ok = khop(u, v, (u, v), 2)
        G.add_edge(u, v)
        if ok: backed += 1
    return backed / len(edges) if edges else float("nan")

def kemeny_LCC(G):
    if G.number_of_nodes() == 0: return float("nan")
    if nx.is_connected(G): return float(nx.kemeny_constant(G))
    cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(cc).copy()
    return float(nx.kemeny_constant(H))

def comm_stats(G):
    if G.number_of_edges() == 0: 
        return {"n":0,"Q":float("nan"),"sizes":[],"node2comm":{}}
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    Q = nx.algorithms.community.modularity(G, comms)
    node2comm = {}
    for i, c in enumerate(comms):
        for u in c: node2comm[u] = i
    return {"n":len(comms), "Q":float(Q), "sizes":sorted([len(c) for c in comms], reverse=True), "node2comm":node2comm}

def summarize(name, triples, kemeny=True, backup_sample=4000):
    Gd, Gud = G_dir(triples), G_und(triples)
    nV, nEund = Gud.number_of_nodes(), Gud.number_of_edges()
    p2t = triples_by_pair(triples)
    p2rels = relsets_by_pair(triples)
    mult = [len(v) for v in p2t.values()]
    relmult = [len(v) for v in p2rels.values()]
    rel_counts = Counter(r for _, r, _ in triples)
    Q = comm_stats(Gud)
    s = {
        "name": name,
        "triples": len(triples),
        "entities": nV,
        "relations": len(rel_counts),
        "und_pairs": nEund,
        "density": density_und(Gud),
        "avg_degree": (2*nEund/nV) if nV>0 else 0.0,
        "avg_clustering": float(nx.average_clustering(Gud)) if nV>0 else float("nan"),
        "bridges_fraction": (len(list(nx.bridges(Gud)))/nEund) if nEund>0 else 0.0,
        "backup_frac_k2": backup_fraction_und(Gud, k=2, sample_edges=backup_sample),
        "communities": Q["n"],
        "modularity_Q": Q["Q"],
        "largest_comm": Q["sizes"][0] if Q["sizes"] else 0,
        "pair_mult_mean": float(np.mean(mult)) if mult else 0.0,
        "pair_mult_med": float(np.median(mult)) if mult else 0.0,
        "rel_per_pair_mean": float(np.mean(relmult)) if relmult else 0.0,
        "rel_per_pair_med": float(np.median(relmult)) if relmult else 0.0,
        "kemeny_LCC": kemeny_LCC(Gud) if kemeny else float("nan"),
        "node2comm": Q["node2comm"],
        "Gud": Gud,  # for downstream checks
        "rel_counts": rel_counts,
    }
    return s

def KL(p, q):
    # KL(p||q) for discrete distributions with shared support
    eps = 1e-12
    keys = set(p.keys()) | set(q.keys())
    P = np.array([p.get(k,0) for k in keys], dtype=float); P = P / (P.sum() + eps)
    Q = np.array([q.get(k,0) for k in keys], dtype=float); Q = Q / (Q.sum() + eps)
    mask = (P>0) & (Q>0)
    if not mask.any(): return float("inf")
    return float(np.sum(P[mask] * np.log((P[mask]+eps)/(Q[mask]+eps))))

# ---------- Removal diagnostics ----------
def removal_analysis(orig, remained, removed):
    # sanity: sets should match
    set_orig = set(orig)
    set_rem = set(remained)
    set_remvd = set(removed)
    if set_orig != (set_rem | set_remvd):
        print("[WARN] remained ∪ removed != original (duplicates or mismatches).")

    ents_orig = set([h for h,_,t in orig] + [t for _,_,t in orig])
    ents_rem = set([h for h,_,t in remained] + [t for _,_,t in remained])
    ents_remvd = set([h for h,_,t in removed] + [t for _,_,t in removed])

    # entities that lost all incident edges
    Gud_orig = G_und(orig)
    Gud_rem = G_und(remained)
    deg0 = [u for u in ents_orig if Gud_rem.degree(u)==0]
    # inter- vs intra-community removals in the original partition
    S_orig = summarize("original", orig, kemeny=False)
    node2comm = S_orig["node2comm"]
    Gud_o = S_orig["Gud"]
    def is_inter(u,v): return node2comm.get(u,-1) != node2comm.get(v,-1)

    removed_pairs = set(pair_key(h,t) for h,_,t in removed if h!=t)
    inter = sum(1 for (u,v) in removed_pairs if is_inter(u,v))
    intra = len(removed_pairs) - inter
    inter_ratio = inter / len(removed_pairs) if removed_pairs else 0.0

    # degree endpoints of removed pairs (in original)
    deg_o = dict(Gud_o.degree())
    deg_endpoints = []
    for (u,v) in removed_pairs:
        du, dv = deg_o.get(u,0), deg_o.get(v,0)
        deg_endpoints.append((min(du,dv), max(du,dv)))
    lowlow = sum(1 for a,b in deg_endpoints if a<=2 and b<=2) / (len(deg_endpoints) or 1)
    hubcut = sum(1 for a,b in deg_endpoints if b>=10) / (len(deg_endpoints) or 1)

    return {
        "entities_lost_all_degree": len(deg0),
        "lost_entities_list_sample": deg0[:10],
        "removed_unique_pairs": len(removed_pairs),
        "removed_inter_comm_pairs": inter,
        "removed_intra_comm_pairs": intra,
        "removed_inter_ratio": inter_ratio,
        "removed_low_low_deg_frac": lowlow,
        "removed_hub_endpoint_frac": hubcut,
    }

# ---------- Pretty ----------
def fnum(x, nd=6):
    if isinstance(x,float):
        if math.isnan(x): return "nan"
        return f"{x:.{nd}g}"
    return str(x)

def report_side_by_side(A, B, labelA="original", labelB="remained"):
    keys = [
        ("triples","int"),("entities","int"),("relations","int"),
        ("und_pairs","int"),("density","float"),("avg_degree","float"),
        ("avg_clustering","float"),("bridges_fraction","float"),
        ("backup_frac_k2","float"),("communities","int"),
        ("modularity_Q","float"),("largest_comm","int"),
        ("kemeny_LCC","float"),
        ("pair_mult_mean","float"),("pair_mult_med","float"),
        ("rel_per_pair_mean","float"),("rel_per_pair_med","float"),
    ]
    w = max(len(k) for k,_ in keys)
    colw = max(len(labelA),len(labelB),12)
    print("\n=== Structure comparison ===")
    print(f"{'metric':<{w}}  {labelA:>{colw}}  {labelB:>{colw}}  {'Δ(B−A)':>{colw}}")
    print("-"*(w+2+3*colw))
    for k, kind in keys:
        a, b = A[k], B[k]
        if kind == "int":
            sa, sb = f"{int(a)}", f"{int(b)}"
        else:
            sa, sb = fnum(a,4), fnum(b,4)
        try:
            delta = (b - a) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else float("nan")
        except Exception:
            delta = float("nan")
        print(f"{k:<{w}}  {sa:>{colw}}  {sb:>{colw}}  {fnum(delta,4):>{colw}}")

def top_k_shift(A_rel, B_rel, k=10):
    keys = set(A_rel.keys()) | set(B_rel.keys())
    table = []
    for r in keys:
        a = A_rel.get(r,0); b = B_rel.get(r,0)
        table.append((r, a, b, b-a))
    table.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"\nTop-{k} relation count changes (remained − original):")
    for r,a,b,dc in table[:k]:
        print(f"  {r:<25}  {a:>6} -> {b:>6}  Δ={dc:+d}")

def main():
    ap = argparse.ArgumentParser("Analyze UMLS slices: original vs remained and removed")
    ap.add_argument("--original", default="original.txt")
    ap.add_argument("--remained", default="umls_2_remained.txt")
    ap.add_argument("--removed",  default="umls_2_removed.txt")
    ap.add_argument("--kemeny", action="store_true", help="Compute Kemeny on LCC (can be slow)")
    ap.add_argument("--backup_sample", type=int, default=4000)
    args = ap.parse_args()

    orig = load_triples(args.original)
    rem  = load_triples(args.remained)
    rmv  = load_triples(args.removed)

    print(f"[load] original={len(orig)}  remained={len(rem)}  removed={len(rmv)}")
    if set(orig) != (set(rem) | set(rmv)):
        print("[WARN] Set mismatch: remained ∪ removed != original (duplicates or parse issues).")

    S_o = summarize("original", orig, kemeny=args.kemeny, backup_sample=args.backup_sample)
    S_r = summarize("remained", rem,  kemeny=args.kemeny, backup_sample=args.backup_sample)

    report_side_by_side(S_o, S_r, "original", "remained")

    # relation distribution shift
    kl_ro = KL(S_r["rel_counts"], S_o["rel_counts"])
    kl_or = KL(S_o["rel_counts"], S_r["rel_counts"])
    print(f"\nRelation distribution KL(remained || original) = {fnum(kl_ro,4)}")
    print(f"Relation distribution KL(original || remained) = {fnum(kl_or,4)}")
    top_k_shift(S_o["rel_counts"], S_r["rel_counts"], k=15)

    # removal diagnostics
    R = removal_analysis(orig, rem, rmv)
    print("\n=== Removal diagnostics (measured in ORIGINAL partition) ===")
    for k, v in R.items():
        print(f"{k:>30}: {v}")

    # Heuristic interpretation
    print("\n=== Why did performance drop? ===")
    notes = []
    if S_r["modularity_Q"] > S_o["modularity_Q"] + 1e-3:
        notes.append("Graph is more modular (Q up) → fewer cross-community shortcuts.")
    if S_r["density"] < S_o["density"] - 1e-3:
        notes.append("Density down → longer paths, less multi-hop signal.")
    if S_r["avg_clustering"] < S_o["avg_clustering"] - 1e-3:
        notes.append("Clustering down → local triangle patterns broken.")
    if S_r["kemeny_LCC"] > S_o["kemeny_LCC"] + 1e-6:
        notes.append("Kemeny up → slower mixing; random-walk connectivity worsened.")
    if R["removed_inter_ratio"] > 0.5:
        notes.append("Removed edges are mostly inter-community → global connectors lost.")
    if R["entities_lost_all_degree"] > 0:
        notes.append("Some entities lost all incident edges → no training signal for them.")
    if KL(S_r["rel_counts"], S_o["rel_counts"]) > 0.1:
        notes.append("Relation distribution shifted → model sees fewer examples of some relations.")
    if not notes:
        notes.append("No single smoking gun; removal was diffuse or effect is mainly data volume.")
    for line in notes:
        print(" - " + line)

if __name__ == "__main__":
    main()
