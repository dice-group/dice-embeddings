import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import matplotlib
import torch.nn.functional as F
from operator import itemgetter

matplotlib.use("Agg")

import os, random, numpy as np, torch

def set_seeds(seed):
    try:
        s = int(np.uint32(seed))
    except Exception:
        s = int(np.uint32(abs(hash(str(seed)))))

    os.environ["PYTHONHASHSEED"] = str(s)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def compute_triple_centrality(graph_triples, adversarial_triples, type):
    corrupted_triples = [item[0] for item in adversarial_triples]

    G = nx.Graph()

    if type == "global":
        for h, r, t in  graph_triples + corrupted_triples:
            G.add_edge(h, t, relation=r)
    if type == "local":
        for h, r, t in  corrupted_triples:
            G.add_edge(h, t, relation=r)

    # Centrality measures
    deg = nx.degree_centrality(G)
    clo = nx.closeness_centrality(G)
    bet = nx.betweenness_centrality(G, normalized=True)

    results = {}
    for h, r, t in corrupted_triples:
        results[(h, r, t)] = {
            "degree":     (deg[h] + deg[t]) / 2,
            "closeness":  (clo[h] + clo[t]) / 2,
            "betweenness": (bet[h] + bet[t]) / 2,
        }

    return results

def visualize_results(csv_path, save_path, title=""):
    df = pd.read_csv(csv_path, header=None)

    raw_x   = df.iloc[0, 1:].astype(float)
    x_pos   = np.arange(len(raw_x))

    rows = df.iloc[1:]

    markers = ["o", "s", "^", "v", "<", ">", "d", "D", "p", "P", "X", "*", "+"]
    styles  = ["-", "--", "-.", ":"]
    combos  = [dict(marker=m, linestyle=ls) for m, ls in product(markers, styles)]

    plt.figure(figsize=(14, 14))
    for i, (_, row) in enumerate(rows.iterrows()):
        label = row.iloc[0]
        y     = row.iloc[1:].astype(float)

        if "random" in label.lower():
            plt.plot(x_pos, y, linewidth=4, marker="X", linestyle="-", label=label, zorder=5)
        else:
            plt.plot(x_pos, y, linewidth=2, label=label, **combos[i % len(combos)])

    plt.xlabel("Triple injection ratios")
    plt.ylabel("MRR")
    plt.title(f"MRR vs Perturbation Ratios {title}")

    plt.xticks(x_pos, raw_x)
    plt.grid(alpha=0.3)

    leg = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    for text in leg.get_texts():
        if "random" in text.get_text().lower():
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt.close('all')
    print(f"Plot saved → {save_path}")


def load_embeddings(entity_csv, relation_csv):
    ent_df = pd.read_csv(entity_csv, index_col=0)
    rel_df = pd.read_csv(relation_csv, index_col=0)

    entity_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in ent_df.iterrows()}
    relation_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in rel_df.iterrows()}

    return entity_emb, relation_emb

def load_triples(path):
    f = open(path, 'r')
    try:
        triples = [tuple(line.strip().split()[:3]) for line in f]
    finally:
        f.close()
    return triples

def get_local_grad_norm(model, h_idx, r_idx, t_idx):
    entity_grad = model.entity_embeddings.weight.grad
    relation_grad = model.relation_embeddings.weight.grad

    norm_parts = []
    if entity_grad is not None:
        norm_parts.append(entity_grad[h_idx])
        norm_parts.append(entity_grad[t_idx])
    if relation_grad is not None:
        norm_parts.append(relation_grad[r_idx])

    if norm_parts:
        return torch.sqrt(torch.stack(norm_parts).sum()).item()

    return 0.0

def compute_embedding_change(model, h_idx, r_idx, t_idx):
    entity_grad = model.entity_embeddings.weight
    relation_grad = model.relation_embeddings.weight

    norm_parts = []
    if entity_grad is not None:
        norm_parts.append(entity_grad[h_idx])
        norm_parts.append(entity_grad[t_idx])
    if relation_grad is not None:
        norm_parts.append(relation_grad[r_idx])

    if norm_parts:
        return torch.sqrt(torch.stack(norm_parts).sum()).item()
    return 0.0

def get_low_score_high_gradient_triples(triples_list):
    scores = [float(item[1]) for item in triples_list]
    avg_score = sum(scores) / len(scores)

    filtered = []
    for item in triples_list:
        score = float(item[1])
        if score < avg_score:
            filtered.append((item[0], item[1], item[2]))

    filtered.sort(key=lambda x: x[2], reverse=True)

    return filtered


def select_score_range_high_gradient(triples_data, k, score_min=0.35, score_max=0.50):
    candidates = [triple for triple in triples_data if score_min <= triple[1] <= score_max]

    sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    return sorted_candidates[:k]

def select_low_score_high_gradient(triples_data, score_percentile):
    triples_data_sorted = sorted(triples_data, key=lambda x: x[1])

    cutoff_index = max(1, int(len(triples_data) * (score_percentile / 100)))
    low_score_candidates = triples_data_sorted[:cutoff_index]

    low_score_high_grad = sorted(low_score_candidates, key=lambda x: x[2], reverse=True)

    return low_score_high_grad

def select_high_score_high_gradient(triples_data, score_percentile):
    triples_by_score = sorted(triples_data, key=lambda x: x[1], reverse=True)

    cutoff = max(1, int(len(triples_by_score) * (score_percentile / 100)))
    high_score_candidates = triples_by_score[:cutoff]

    candidates_by_grad = sorted(high_score_candidates, key=lambda x: x[2], reverse=True)

    return candidates_by_grad

def select_low_score_low_gradient(triples_data, score_percentile):
    triples_by_score = sorted(triples_data, key=lambda x: x[1])
    cutoff = max(1, int(len(triples_by_score) * (score_percentile / 100)))
    low_score_candidates = triples_by_score[:cutoff]

    low_grad = sorted(low_score_candidates, key=lambda x: x[2])
    return low_grad

def select_high_score_low_gradient(triples_data, score_percentile):
    triples_by_score = sorted(triples_data, key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(triples_by_score) * (score_percentile / 100)))
    high_score_candidates = triples_by_score[:cutoff]

    low_grad = sorted(high_score_candidates, key=lambda x: x[2])
    return low_grad


def low_degree(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["degree"])

def high_degree(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["degree"], reverse=True)

def low_closeness(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["closeness"])

def high_closeness(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["closeness"], reverse=True)

def high_betweenness(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["betweenness"], reverse=True)


def fgsm_delta(grad, eps, norm):
    if norm == "linf":
        return eps * grad.sign()
    if norm == "l2":
        gnorm = grad.norm() + 1e-12
        return eps * (grad / gnorm)
    raise ValueError("norm must be 'linf' or 'l2'")


def nearest_idx_excluding(vec, table, exclude_idx):
    v = vec.unsqueeze(0)
    d = torch.cdist(v, table)[0]
    d[exclude_idx] = float("inf")
    return int(torch.argmin(d).item())


def logits_for_indices(model, h_i, r_i, t_i, device):
    idxs = torch.tensor([[h_i, r_i, t_i]], dtype=torch.long, device=device)
    return model.forward_triples(idxs)


def select_adversarial_triples_fgsm(
    triples,
    corruption_type,
    oracle,
    seed,
    eps,
    norm,
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = next(oracle.model.parameters()).device

    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}
    I2R = {i: r for r, i in R2I.items()}

    base_seen = {(E2I[h], R2I[r], E2I[t]) for (h, r, t) in triples}

    seen = set(base_seen)

    ent_emb = oracle.model.entity_embeddings
    rel_emb = oracle.model.relation_embeddings

    # frozen
    E = ent_emb.weight.detach()
    R = rel_emb.weight.detach()

    adverserial_triples = []

    for (h, r, t) in triples:
        h_i, r_i, t_i = E2I[h], R2I[r], E2I[t]

        for p in oracle.model.parameters():
            p.requires_grad_(True)
        oracle.model.zero_grad(set_to_none=True)
        oracle.model.train(False)

        logits_clean = logits_for_indices(oracle.model, h_i, r_i, t_i, device)
        target_pos = torch.ones_like(logits_clean)
        loss = F.binary_cross_entropy_with_logits(logits_clean, target_pos)
        loss.backward()

        GE = ent_emb.weight.grad
        GR = rel_emb.weight.grad

        if GE is None or GR is None:
            raise RuntimeError("Embedding grads are None.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h, g_r, g_t = GE[h_i], GR[r_i], GE[t_i]
        triple_grad_norm = torch.linalg.vector_norm(torch.cat([g_h.flatten(), g_r.flatten(), g_t.flatten()]))

        # FGSM step
        dh = fgsm_delta(g_h, eps, norm)
        dr = fgsm_delta(g_r, eps, norm)
        dt = fgsm_delta(g_t, eps, norm)

        Eh = E[h_i] + dh
        Er = R[r_i] + dr
        Et = E[t_i] + dt

        # nearest (exclude originals)
        h_adv_i = nearest_idx_excluding(Eh, E, h_i)
        r_adv_i = nearest_idx_excluding(Er, R, r_i)
        t_adv_i = nearest_idx_excluding(Et, E, t_i)

        # choose corruption pattern
        if corruption_type == "all":
            cand = (h_adv_i, r_adv_i, t_adv_i)
        elif corruption_type == "head":
            cand = (h_adv_i, r_i, t_i)
        elif corruption_type == "rel":
            cand = (h_i, r_adv_i, t_i)
        elif corruption_type == "tail":
            cand = (h_i, r_i, t_adv_i)
        elif corruption_type == "head-tail":
            cand = (h_adv_i, r_i, t_adv_i)
        elif corruption_type == "head-rel":
            cand = (h_adv_i, r_adv_i, t_i)
        elif corruption_type == "random-one":
            with torch.no_grad():
                lh = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, h_adv_i, r_i, t_i, device), target_pos).item()
                lr = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, h_i, r_adv_i, t_i, device), target_pos).item()
                lt = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, h_i, r_i, t_adv_i, device), target_pos).item()
            if lh >= lr and lh >= lt:
                cand = (h_adv_i, r_i, t_i)
            elif lr >= lt:
                cand = (h_i, r_adv_i, t_i)
            else:
                cand = (h_i, r_i, t_adv_i)
        else:
            raise ValueError("Invalid corruption_type")

        attempts = 0
        max_attempts = 5
        while (cand == (h_i, r_i, t_i)) or (cand in seen):
            attempts += 1
            if attempts > max_attempts:
                break
            # adjust one changed field (kept simple)
            if cand[0] != h_i:
                d = torch.cdist(Eh.unsqueeze(0), E)[0]
                d[h_i] = float("inf")
                d[cand[0]] = float("inf")
                cand = (int(torch.argmin(d).item()), cand[1], cand[2])
            elif cand[1] != r_i:
                d = torch.cdist(Er.unsqueeze(0), R)[0]
                d[r_i] = float("inf")
                d[cand[1]] = float("inf")
                cand = (cand[0], int(torch.argmin(d).item()), cand[2])
            elif cand[2] != t_i:
                d = torch.cdist(Et.unsqueeze(0), E)[0]
                d[t_i] = float("inf")
                d[cand[2]] = float("inf")
                cand = (cand[0], cand[1], int(torch.argmin(d).item()))

        if (cand == (h_i, r_i, t_i)) or (cand in seen):
            continue

        seen.add(cand)

        h_c, r_c, t_c = cand
        logits_adv = logits_for_indices(oracle.model, h_c, r_c, t_c, device)
        pred_prob = torch.sigmoid(logits_adv).item()

        corrupted_triple = (I2E[h_c], I2R[r_c], I2E[t_c])
        clean_triple = (h, r, t)
        adverserial_triples.append((corrupted_triple, clean_triple, pred_prob, float(triple_grad_norm)))

    # build outputs (unchanged from your version)
    fgsm_adverserial_triples = adverserial_triples.copy()
    low_scores = sorted(adverserial_triples.copy(), key=itemgetter(2))

    corrupted_centerality_global = compute_triple_centrality(triples, adverserial_triples.copy(), type="global")
    high_close_global = high_closeness(corrupted_centerality_global)
    high_betw_global = high_betweenness(corrupted_centerality_global)

    corrupted_centerality_local = compute_triple_centrality(triples, adverserial_triples.copy(), type="local")
    high_close_local = high_closeness(corrupted_centerality_local)
    high_betw_local = high_betweenness(corrupted_centerality_local)

    high_gradients = sorted(adverserial_triples.copy(), key=itemgetter(3), reverse=True)

    return low_scores, high_close_global, high_betw_global, high_close_local,  high_betw_local,  high_gradients, fgsm_adverserial_triples

#########################

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

def _rank_nodes(centrality_dict):
    return sorted(centrality_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

def _propose_corruptions(triples, node_centrality, budget, mode="both", top_k_nodes=100, avoid_existing_edge=False):
    G_nodes_ranked = _rank_nodes(node_centrality)
    if top_k_nodes is not None:
        G_nodes_ranked = G_nodes_ranked[:top_k_nodes]
    top_nodes = [n for n, _ in G_nodes_ranked]

    triple_set, ht_set = _existing_sets(triples)
    candidates = []
    seen = set()

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
                score = 0.5 * (node_centrality.get(n, 0.0) + node_centrality.get(t, 0.0))
                candidates.append((score, cand))
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
                score = 0.5 * (node_centrality.get(h, 0.0) + node_centrality.get(n, 0.0))
                candidates.append((score, cand))
                seen.add(cand)

    candidates.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]), reverse=True)
    top = [trip for _, trip in candidates[:budget]]
    return top

def add_corrupted_by_betweenness(triples, budget, mode="both", top_k_nodes=100, avoid_existing_edge=False):
    G = build_digraph(triples)
    node_cent = nx.betweenness_centrality(G, normalized=True)
    return _propose_corruptions(
        triples, node_cent, budget, mode=mode, top_k_nodes=top_k_nodes, avoid_existing_edge=avoid_existing_edge
    )

def add_corrupted_by_closeness(triples, budget, mode="both", top_k_nodes=100, undirected=False, avoid_existing_edge=False):
    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    node_cent = nx.closeness_centrality(G)
    return _propose_corruptions(
        triples, node_cent, budget, mode=mode, top_k_nodes=top_k_nodes, avoid_existing_edge=avoid_existing_edge
    )

def add_corrupted_by_degree(triples, budget, mode="both", top_k_nodes=100, undirected=True, avoid_existing_edge=True):
    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    deg = dict(G.degree())                     # raw degree
    n = max(len(G), 1)
    node_cent = {u: d / max(n-1, 1) for u, d in deg.items()}
    return _propose_corruptions(triples, node_cent, budget, mode, top_k_nodes, avoid_existing_edge)

def add_corrupted_by_hits(triples, budget, mode="both", top_k_nodes=100, avoid_existing_edge=True, max_iter=100):
    G = build_digraph(triples)
    hubs, auth = nx.hits(G, max_iter=max_iter, normalized=True)
    node_cent = {u: 0.5*(hubs.get(u,0.0) + auth.get(u,0.0)) for u in G.nodes()}
    return _propose_corruptions(triples, node_cent, budget, mode, top_k_nodes, avoid_existing_edge)

def save_triples(triple_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triple_list:
            f.write(f"{h}\t{r}\t{t}\n")