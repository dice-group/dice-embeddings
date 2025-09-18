import networkx as nx

def build_digraph(triples):
    G = nx.DiGraph()
    for h, _, t in triples:
        if not G.has_edge(h, t):
            G.add_edge(h, t)
    return G

def _existing_sets(triples):
    triple_set = set(triples)
    ht_set = set((h, t) for h, _, t in triples)
    return triple_set, ht_set

def _rank_nodes(centrality_dict, top_k_nodes):
    ranked = sorted(centrality_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if top_k_nodes is not None:
        ranked = ranked[:top_k_nodes]
    return [n for n, _ in ranked]

def add_corrupted_by_pagerank(triples, budget, mode="both", top_k_nodes=100, avoid_existing_edge=True,
                              alpha=0.85, max_iter=100, tol=1e-06):

    G = build_digraph(triples)
    pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
    top_nodes = _rank_nodes(pr, top_k_nodes)

    triple_set, ht_set = _existing_sets(triples)
    candidates, seen = [], set()
    for h, r, t in triples:
        if mode in ("head", "both"):
            for n in top_nodes:
                if n == h or n == t: continue
                cand = (n, r, t)
                if cand in triple_set or cand in seen: continue
                if avoid_existing_edge and (n, t) in ht_set: continue
                score = 0.5 * (pr.get(n, 0.0) + pr.get(t, 0.0))
                candidates.append((score, cand)); seen.add(cand)
        if mode in ("tail", "both"):
            for n in top_nodes:
                if n == h or n == t: continue
                cand = (h, r, n)
                if cand in triple_set or cand in seen: continue
                if avoid_existing_edge and (h, n) in ht_set: continue
                score = 0.5 * (pr.get(h, 0.0) + pr.get(n, 0.0))
                candidates.append((score, cand)); seen.add(cand)
    candidates.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]), reverse=True)
    return [trip for _, trip in candidates[:min(budget, len(candidates))]]

def add_corrupted_by_harmonic_closeness(triples, budget, mode="both", top_k_nodes=100,
                                        undirected=True, avoid_existing_edge=True):

    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    hc = nx.harmonic_centrality(G)
    top_nodes = _rank_nodes(hc, top_k_nodes)

    triple_set, ht_set = _existing_sets(triples)
    candidates, seen = [], set()
    for h, r, t in triples:
        if mode in ("head", "both"):
            for n in top_nodes:
                if n == h or n == t: continue
                cand = (n, r, t)
                if cand in triple_set or cand in seen: continue
                if avoid_existing_edge and (n, t) in ht_set: continue
                score = 0.5 * (hc.get(n, 0.0) + hc.get(t, 0.0))
                candidates.append((score, cand)); seen.add(cand)
        if mode in ("tail", "both"):
            for n in top_nodes:
                if n == h or n == t: continue
                cand = (h, r, n)
                if cand in triple_set or cand in seen: continue
                if avoid_existing_edge and (h, n) in ht_set: continue
                score = 0.5 * (hc.get(h, 0.0) + hc.get(n, 0.0))
                candidates.append((score, cand)); seen.add(cand)
    candidates.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]), reverse=True)
    return [trip for _, trip in candidates[:min(budget, len(candidates))]]




def add_corrupted_by_edge_betweenness(triples, budget, mode="both", top_k_nodes=100,
                                      undirected=True, avoid_existing_edge=True,
                                      approx_k=None, agg="max", normalized=True, weight=None, seed=None):
    

    if agg not in {"max", "sum", "mean"}:
        raise ValueError("agg must be one of {'max','sum','mean'}")

    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    edge_bw = nx.edge_betweenness_centrality(G, k=approx_k, normalized=normalized, weight=weight, seed=seed)

    node_score = {n: 0.0 for n in G.nodes()}
    deg_count = {n: 0 for n in G.nodes()} 

    for (u, v), s in edge_bw.items():
        if agg == "max":
            if s > node_score[u]: node_score[u] = s
            if s > node_score[v]: node_score[v] = s
        else:  # sum or mean
            node_score[u] += s
            node_score[v] += s
        deg_count[u] += 1
        deg_count[v] += 1

    if agg == "mean":
        for n in node_score:
            c = deg_count[n]
            node_score[n] = (node_score[n] / c) if c > 0 else 0.0

    top_nodes = _rank_nodes(node_score, top_k_nodes)   
    triple_set, ht_set = _existing_sets(triples)        

    candidates, seen = [], set()
    for h, r, t in triples:
        if mode in ("head", "both"):
            for n in top_nodes:
                if n == h or n == t:
                    continue
                cand = (n, r, t)
                if cand in triple_set or cand in seen:
                    continue
                if avoid_existing_edge and (n, t) in ht_set:
                    continue
                score = 0.5 * (node_score.get(n, 0.0) + node_score.get(t, 0.0))
                candidates.append((score, cand)); seen.add(cand)

        if mode in ("tail", "both"):
            for n in top_nodes:
                if n == h or n == t:
                    continue
                cand = (h, r, n)
                if cand in triple_set or cand in seen:
                    continue
                if avoid_existing_edge and (h, n) in ht_set:
                    continue
                score = 0.5 * (node_score.get(h, 0.0) + node_score.get(n, 0.0))
                candidates.append((score, cand)); seen.add(cand)

    candidates.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]), reverse=True)
    return [trip for _, trip in candidates[:min(budget, len(candidates))]]
