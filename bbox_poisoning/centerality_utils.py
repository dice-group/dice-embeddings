import networkx as nx
import heapq

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

def remove_by_edge_betweenness(triples, budget, approx_k=None):
    G = build_digraph(triples)
    edge_bw = nx.edge_betweenness_centrality(G, k=approx_k, normalized=True)
    scored = []
    for i, (h, r, t) in enumerate(triples):
        s = edge_bw.get((h, t), 0.0)
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for _, _, _, _, i in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]

def remove_by_endpoint_pagerank(triples, budget, alpha=0.85, max_iter=100, tol=1e-06):
    G = build_digraph(triples)
    pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
    scored = []
    for i, (h, r, t) in enumerate(triples):
        s = 0.5 * (pr.get(h, 0.0) + pr.get(t, 0.0))
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for _, _, _, _, i in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]

def remove_by_endpoint_harmonic_closeness(triples, budget, undirected=True):
    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    hc = nx.harmonic_centrality(G)
    scored = []
    for i, (h, r, t) in enumerate(triples):
        s = 0.5 * (hc.get(h, 0.0) + hc.get(t, 0.0))
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for _, _, _, _, i in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]

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

def _push_candidate_heap(heap, budget, score, cand):
    if budget <= 0:
        return
    key = (score, str(cand[0]), str(cand[1]), str(cand[2]))  # deterministic tie-break
    item = (key, cand)
    if len(heap) < budget:
        heapq.heappush(heap, item)
    else:
        if item > heap[0]:
            heapq.heapreplace(heap, item)

def _rank_nodes_stable(centrality_dict, top_k_nodes):
    ranked = sorted(
        centrality_dict.items(),
        key=lambda kv: (kv[1], str(kv[0])),
        reverse=True
    )
    if top_k_nodes is not None:
        ranked = ranked[:top_k_nodes]
    return [n for n, _ in ranked]

def add_corrupted_by_pagerank_topk(triples, budget, mode="both", top_k_nodes=100,
                                   avoid_existing_edge=True, alpha=0.85, max_iter=100, tol=1e-06):
    if budget <= 0:
        return []

    G = build_digraph(triples)
    pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
    top_nodes = _rank_nodes_stable(pr, top_k_nodes)

    triple_set, ht_set = _existing_sets(triples)
    heap, seen = [], set()

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
                score = 0.5 * (pr.get(n, 0.0) + pr.get(t, 0.0))
                _push_candidate_heap(heap, budget, score, cand)
                seen.add(cand)

        if mode in ("tail", "both"):
            for n in top_nodes:
                if n == h or n == t:
                    continue
                cand = (h, r, n)
                if cand in triple_set or cand in seen:
                    continue
                if avoid_existing_edge and (h, n) in ht_set:
                    continue
                score = 0.5 * (pr.get(h, 0.0) + pr.get(n, 0.0))
                _push_candidate_heap(heap, budget, score, cand)
                seen.add(cand)

    heap.sort(reverse=True)
    return [cand for (_, cand) in heap]

def add_corrupted_by_harmonic_closeness_topk(triples, budget, mode="both", top_k_nodes=100,
                                             undirected=True, avoid_existing_edge=True):
    if budget <= 0:
        return []

    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    hc = nx.harmonic_centrality(G)
    top_nodes = _rank_nodes_stable(hc, top_k_nodes)

    triple_set, ht_set = _existing_sets(triples)
    heap, seen = [], set()

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
                score = 0.5 * (hc.get(n, 0.0) + hc.get(t, 0.0))
                _push_candidate_heap(heap, budget, score, cand)
                seen.add(cand)

        if mode in ("tail", "both"):
            for n in top_nodes:
                if n == h or n == t:
                    continue
                cand = (h, r, n)
                if cand in triple_set or cand in seen:
                    continue
                if avoid_existing_edge and (h, n) in ht_set:
                    continue
                score = 0.5 * (hc.get(h, 0.0) + hc.get(n, 0.0))
                _push_candidate_heap(heap, budget, score, cand)
                seen.add(cand)

    heap.sort(reverse=True)
    return [cand for (_, cand) in heap]

