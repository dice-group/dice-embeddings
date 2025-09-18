import torch, numpy as np, networkx as nx

def triples_to_idx(triples, E2I, R2I):
    return np.array([(E2I[h], R2I[r], E2I[t]) for h, r, t in triples], dtype=np.int64)

def logits_for_indices(model, h_i, r_i, t_i, device):
    idxs = torch.tensor([[h_i, r_i, t_i]], dtype=torch.long, device=device)
    return model.forward_triples(idxs).reshape(-1)  # logits

def pos_only_loss(logits):
    return -torch.nn.functional.logsigmoid(logits).mean()

def zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def per_triple_grad_rows(oracle, triple_idx, device):
    model = oracle.model
    model.train()
    ent = model.entity_embeddings.weight
    rel = model.relation_embeddings.weight
    ent.requires_grad_(True); rel.requires_grad_(True)

    zero_grads(model)
    h, r, t = map(int, triple_idx)
    logits = logits_for_indices(model, h, r, t, device)
    loss = pos_only_loss(logits)
    loss.backward()

    g_h = ent.grad[h].detach().clone()
    g_r = rel.grad[r].detach().clone()
    g_t = ent.grad[t].detach().clone()
    return g_h, g_r, g_t, h, r, t

def grad_change_score(g_h, g_r, g_t, p=2):
    if p == 2:
        return torch.linalg.vector_norm(torch.cat([g_h, g_r, g_t])).item()
    # L1 fallback
    return (g_h.abs().sum() + g_r.abs().sum() + g_t.abs().sum()).item()

def pick_top_influential(oracle, triples, top_m, device=None, max_candidates=None, seed=0):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)

    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    triples_idx = triples_to_idx(triples, E2I, R2I)

    N = len(triples_idx)
    if max_candidates is not None and max_candidates < N:
        cand_ids = np.random.choice(N, size=max_candidates, replace=False)
    else:
        cand_ids = np.arange(N)

    scores = np.zeros(len(cand_ids), dtype=np.float64)
    for j, i in enumerate(cand_ids):
        g_h, g_r, g_t, *_ = per_triple_grad_rows(oracle, triples_idx[i], device)
        scores[j] = grad_change_score(g_h, g_r, g_t, p=2)

    k = min(top_m, len(cand_ids))
    top_local = np.argpartition(scores, -k)[-k:]
    top_local = top_local[np.argsort(scores[top_local])[::-1]]
    chosen_ids = cand_ids[top_local]
    return chosen_ids.tolist()

def build_digraph(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        if not G.has_node(h): G.add_node(h)
        if not G.has_node(t): G.add_node(t)
        if not G.has_edge(h, t): G.add_edge(h, t)
    return G

def rank_candidates_by_betweenness(triples, candidate_ids, approx_k=None, budget=100):
    G = build_digraph(triples)
    edge_bw = nx.edge_betweenness_centrality(G, k=approx_k, normalized=True)
    scored = []
    for i in candidate_ids:
        h, r, t = triples[i]
        s = edge_bw.get((h, t), 0.0)
        scored.append((s, i))
    scored.sort(reverse=True)
    keep = [i for _, i in scored[:min(budget, len(scored))]]
    return [triples[i] for i in keep]

def rank_candidates_by_closeness(triples, candidate_ids, budget=100):
    G = build_digraph(triples)
    node_cl = nx.closeness_centrality(G)
    scored = []
    for i in candidate_ids:
        h, r, t = triples[i]
        s = 0.5 * (node_cl.get(h, 0.0) + node_cl.get(t, 0.0))
        scored.append((s, i))
    scored.sort(reverse=True)
    keep = [i for _, i in scored[:min(budget, len(scored))]]
    return [triples[i] for i in keep]

def removal_lists_grad_then_centrality(oracle, triples, budget,
                                       top_m_grad=None, approx_k=None,
                                       device=None, max_candidates=None, seed=0):
    if top_m_grad is None:
        top_m_grad = min(len(triples), max(budget * 10, budget))  # shallow filter
    cand_ids = pick_top_influential(
        oracle, triples, top_m=top_m_grad, device=device,
        max_candidates=max_candidates, seed=seed
    )
    betw_list = rank_candidates_by_betweenness(triples, cand_ids, approx_k=approx_k, budget=budget)
    clos_list = rank_candidates_by_closeness(triples, cand_ids, budget=budget)
    return betw_list, clos_list


def remove_by_edge_betweenness(triples, budget, approx_k=None):
    G = build_digraph(triples)
    edge_bw = nx.edge_betweenness_centrality(G, k=approx_k, normalized=True)
    scored = []
    for i, (h, r, t) in enumerate(triples):
        s = edge_bw.get((h, t), 0.0)
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for (_, _, _, _, i) in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]


def remove_by_endpoint_closeness(triples, budget, undirected=False):
    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    node_cl = nx.closeness_centrality(G)
    scored = []
    for i, (h, r, t) in enumerate(triples):
        s = 0.5 * (node_cl.get(h, 0.0) + node_cl.get(t, 0.0))
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for (_, _, _, _, i) in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]

