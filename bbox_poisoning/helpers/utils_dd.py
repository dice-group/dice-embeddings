# deterministic_grad_then_centrality.py
import torch, numpy as np, networkx as nx

# --- determinism switches (safe on CPU; for GPU see notes below) ---
def set_deterministic():
    torch.use_deterministic_algorithms(True)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True
    except Exception:
        pass

# --- mappings & helpers ---
def triples_to_idx(triples, E2I, R2I):
    # Deterministic mapping using your oracle dicts
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

# --- Stage 1: per-triple gradient norm (influence proxy) ---
def per_triple_grad_norms(oracle, triples_idx, device="cpu"):
    model = oracle.model.to(device)
    model.train()

    ent = model.entity_embeddings.weight
    rel = model.relation_embeddings.weight
    ent.requires_grad_(True); rel.requires_grad_(True)

    N = len(triples_idx)
    norms = np.zeros(N, dtype=np.float64)

    for i in range(N):
        zero_grads(model)
        h, r, t = map(int, triples_idx[i])
        logits = logits_for_indices(model, h, r, t, device)
        loss = pos_only_loss(logits)
        loss.backward()
        g_h = ent.grad[h]
        g_r = rel.grad[r]
        g_t = ent.grad[t]
        # L2 norm of concatenated rows
        norms[i] = torch.linalg.vector_norm(torch.cat([g_h, g_r, g_t])).item()

    return norms  # shape [N]

# --- Graph centralities (exact; deterministic) ---
def build_digraph(triples):
    G = nx.DiGraph()
    for h, _, t in triples:
        if not G.has_edge(h, t):
            G.add_edge(h, t)
    return G

def rank_candidates_by_betweenness(triples, candidate_ids, budget):
    G = build_digraph(triples)
    edge_bw = nx.edge_betweenness_centrality(G, k=None, normalized=True)  # exact, deterministic
    scored = []
    for i in candidate_ids:
        h, _, t = triples[i]
        s = edge_bw.get((h, t), 0.0)
        # stable tie-breaker on (score, head, rel, tail, index)
        scored.append((s, h, triples[i][1], t, i))
    scored.sort(reverse=True)
    top_idx = [i for *_, i in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]

def rank_candidates_by_closeness(triples, candidate_ids, budget):
    G = build_digraph(triples)
    node_cl = nx.closeness_centrality(G)
    scored = []
    for i in candidate_ids:
        h, r, t = triples[i]
        s = 0.5 * (node_cl.get(h, 0.0) + node_cl.get(t, 0.0))
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for *_, i in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]

# --- Orchestrator: deterministic two lists ---
def removal_lists_grad_then_centrality_deterministic(oracle, triples, budget,
                                                     top_m_grad=None, device="cpu"):
    set_deterministic()

    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    triples_idx = triples_to_idx(triples, E2I, R2I)

    # Stage 1: influence proxy (no randomness)
    norms = per_triple_grad_norms(oracle, triples_idx, device=device)

    # pick top-M deterministically; tie-break with index
    N = len(norms)
    if top_m_grad is None:
        top_m_grad = min(N, max(budget * 10, budget))
    order = np.lexsort((-np.arange(N), -norms))  # sort by norms desc, then index desc
    cand_ids = order[:top_m_grad]

    # Stage 2a/2b: centrality ranks (exact)
    betw_list = rank_candidates_by_betweenness(triples, cand_ids, budget)
    clos_list = rank_candidates_by_closeness(triples, cand_ids, budget)
    return betw_list, clos_list
