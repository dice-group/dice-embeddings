import pandas as pd
import os
import torch, random, heapq, statistics
import math
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import statistics
from itertools import product
import matplotlib
import torch.nn.functional as F
import torch.nn as nn
from operator import itemgetter

matplotlib.use("Agg")

def get_embedding_modules_by_name(model):
    """Find embeddings by common attribute names. Edit names here to match your Keci class."""
    m = model.module if hasattr(model, "module") else model
    ent = None
    rel = None
    for name in ("entity_emb", "entities", "entity_embeddings", "ent_emb"):
        if hasattr(m, name) and isinstance(getattr(m, name), nn.Embedding):
            ent = getattr(m, name); break
    for name in ("relation_emb", "relations", "relation_embeddings", "rel_emb"):
        if hasattr(m, name) and isinstance(getattr(m, name), nn.Embedding):
            rel = getattr(m, name); break
    return ent, rel

def grads_for_rows(ent_emb_mod, rel_emb_mod, idxs: torch.LongTensor):
    GE = ent_emb_mod.weight.grad
    GR = rel_emb_mod.weight.grad
    if GE is None or GR is None:
        raise RuntimeError("Embedding grads are None; check for .detach() / no_grad in forward.")
    if GE.is_sparse: GE = GE.to_dense()
    if GR.is_sparse: GR = GR.to_dense()
    h_i, r_i, t_i = idxs[0,0].item(), idxs[0,1].item(), idxs[0,2].item()
    return GE[h_i], GR[r_i], GE[t_i]



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_triple_centrality(graph_triples, adverserial_triples):

    corrupted_triples = [item[0] for item in adverserial_triples]

    G = nx.Graph()

    for h, r, t in graph_triples + corrupted_triples:
        G.add_edge(h, t, relation=r)

    deg, clo = nx.degree_centrality(G), nx.closeness_centrality(G)

    results = {}
    for h, r, t in corrupted_triples:
        results[(h, r, t)] = {
            "degree":   (deg[h] + deg[t]) / 2,
            "closeness": (clo[h] + clo[t]) / 2,
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

def plot_3d_score_gradient_centrality(triples_data, name, save_dir):
    scores = [item[1] for item in triples_data]
    gradients = [item[2] for item in triples_data]
    centralities = [item[3] for item in triples_data]

    degree = [c['degree'] for c in centralities]
    betweenness = [c['betweenness'] for c in centralities]
    closeness = [c['closeness'] for c in centralities]
    #eigenvector = [c['eigenvector'] for c in centralities]

    measures = {
        "Degree Centrality": degree,
        #"Betweenness Centrality": betweenness,
        "Closeness Centrality": closeness,
        #"Eigenvector Centrality": eigenvector
    }

    for measure_name, values in measures.items():
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(scores, gradients, values, c=values, cmap='viridis', s=50, alpha=0.8)
        ax.set_xlabel('Score')
        ax.set_ylabel('Gradient')
        ax.set_zlabel(measure_name)
        ax.set_title(f"3D Plot: Score vs Gradient vs {measure_name}")

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(measure_name)

        file_name = f"{name}_{measure_name.replace(' ', '_').lower()}.png"
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def scatter_score_vs_gradient(scores, gradients, name, save_dir):
    plt.figure(figsize=(8,6))
    plt.scatter(scores, gradients, alpha=0.6, c='blue', edgecolors='k')
    plt.xlabel("Score")
    plt.ylabel("Gradient")
    plt.title("Score vs Gradient")
    plt.grid(alpha=0.3)

    file_name = f"{name}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def joint_kde_score_gradient(scores, gradients, name, save_dir):
    df = pd.DataFrame({'Score': scores, 'Gradient': gradients})
    sns.jointplot(data=df, x="Score", y="Gradient", kind="kde", fill=True)
    plt.suptitle("Density of Score vs Gradient", y=1.02)
    file_name = f"{name}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

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

def get_high_impact_triples(centrality_dict, k=10, weights=None):
    if weights is None:
        weights = {
                'degree': 0.5,
                #'betweenness': 0.25,
                'closeness': 0.5,
                #'eigenvector': 0.25
        }

    metrics = {key: [v[key] for v in centrality_dict.values()] for key in weights.keys()}

    normalized_metrics = {}
    for key, values in metrics.items():
        arr = np.array(values)
        norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
        normalized_metrics[key] = norm

    triples = list(centrality_dict.keys())
    impact_scores = []
    for i, triple in enumerate(triples):
        score = sum(
            weights[key] * normalized_metrics[key][i]
            for key in weights.keys()
        )
        impact_scores.append((triple, score))

    impact_scores.sort(key=lambda x: x[1], reverse=True)

    return impact_scores[:k]

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

def plot_kde_dual_axis(file_path):
    scores = []
    gradients = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triple, score_str, grad = eval(line)
                scores.append(float(score_str))
                gradients.append(float(grad))
            except Exception as e:
                print(f"Error parsing line: {line} -> {e}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.kdeplot(scores, fill=True, color="blue", ax=ax1, label="Scores")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Score Density", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    sns.kdeplot(gradients, fill=True, color="orange", ax=ax2, label="Gradients")
    ax2.set_ylabel("Gradient Density", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    plt.title("KDE of Scores and Gradients")
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()

def plot_scores_and_gradients_simple(file_path, limit=None):
    scores = []
    gradients = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triple, score_str, grad, centrality = eval(line)
                scores.append(float(score_str))
                gradients.append(float(grad))
            except Exception as e:
                print(f"Error parsing line: {line} -> {e}")

    if limit:
        scores = scores[:limit]
        gradients = gradients[:limit]

    x = range(len(scores))

    plt.figure(figsize=(200, 6))
    plt.plot(x, gradients, color='blue', alpha=0.7, label='Gradients')
    plt.plot(x, scores, color='green', alpha=0.7, label='Scores')

    plt.xlabel("Triples (index)")
    plt.ylabel("Value")
    plt.title("Scores and Gradients per Triple")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def compute_spearman(scores, gradients):
    corr, p_value = spearmanr(scores, gradients)
    return corr, p_value

def compute_pearson(scores, gradients):
    corr, p_value = pearsonr(scores, gradients)
    return corr, p_value

def quantify_relation(scores, gradients):
    pearson_corr, _ = pearsonr(scores, gradients)
    spearman_corr, _ = spearmanr(scores, gradients)
    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
    }

def select_high_diff_triples(triples_data, k, absolute=True):
    ranked = []
    for triple, score, grad, centrality in triples_data:
        diff = abs(grad - score) if absolute else (grad - score)
        ranked.append((triple, score, grad, diff))

    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked[:k]

def select_low_score_high_diff(triples_data, k, score_percentile=20):
    triples_sorted = sorted(triples_data, key=lambda x: x[1])
    cutoff_index = max(1, int(len(triples_data) * (score_percentile / 100)))
    low_score_candidates = triples_sorted[:cutoff_index]
    ranked = []
    for triple, score, grad, centrality in low_score_candidates:
        diff = abs(grad - score)
        ranked.append((triple, score, grad, diff))
    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked[:k]

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

# ---------- helpers ----------
def _as_tuple(t):
    return tuple(t) if isinstance(t, (list, tuple)) else t

def make_sg_dict(adversarial_triples):
    out = {}
    for t, s, g in adversarial_triples:
        t = _as_tuple(t)
        out[t] = {"score": float(s), "gradient": float(g)}
    return out

def _select_by(cent, adversarial_triples, *,
               centrality="degree", centrality_high=True,
               metric="score", metric_high=False,
               top_k=None):

    assert centrality in ("degree", "closeness")
    assert metric in ("score", "gradient")

    cent = { _as_tuple(t): v for t, v in cent.items() }
    sg_dict = make_sg_dict(adversarial_triples)

    triples = []
    for t in cent.keys():
        if (
            t in sg_dict
            and isinstance(cent[t].get(centrality), (int, float))
            and isinstance(sg_dict[t].get(metric), (int, float))
        ):
            triples.append(t)

    if not triples:
        return []


    signed_c_entries = []
    c_sign = -1.0 if centrality_high else 1.0
    for t in triples:
        c_val = cent[t][centrality]
        signed_c_entries.append((c_sign * c_val, t))

    signed_c_entries.sort()
    triples = [t for _, t in signed_c_entries]


    if top_k is not None and top_k > 0:
        triples = triples[:top_k]


    m_sign = -1.0 if metric_high else 1.0
    mc_entries = []
    for t in triples:
        m_val = sg_dict[t][metric]
        c_val = cent[t][centrality]
        m_signed = m_sign * m_val
        c_signed = c_sign * c_val
        mc_entries.append(((m_signed, c_signed), t))

    mc_entries.sort()
    triples = [t for _, t in mc_entries]

    return triples


def degree_high_score_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=True,
                      metric="score", metric_high=True, top_k=top_k)

def degree_high_score_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=True,
                      metric="score", metric_high=False, top_k=top_k)

def degree_low_score_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=False,
                      metric="score", metric_high=True, top_k=top_k)

def degree_low_score_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=False,
                      metric="score", metric_high=False, top_k=top_k)


def degree_high_grad_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=True,
                      metric="gradient", metric_high=True, top_k=top_k)

def degree_high_grad_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=True,
                      metric="gradient", metric_high=False, top_k=top_k)

def degree_low_grad_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=False,
                      metric="gradient", metric_high=True, top_k=top_k)

def degree_low_grad_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="degree", centrality_high=False,
                      metric="gradient", metric_high=False, top_k=top_k)


def closeness_high_score_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=True,
                      metric="score", metric_high=True, top_k=top_k)

def closeness_high_score_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=True,
                      metric="score", metric_high=False, top_k=top_k)

def closeness_low_score_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=False,
                      metric="score", metric_high=True, top_k=top_k)

def closeness_low_score_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=False,
                      metric="score", metric_high=False, top_k=top_k)


def closeness_high_grad_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=True,
                      metric="gradient", metric_high=True, top_k=top_k)

def closeness_high_grad_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=True,
                      metric="gradient", metric_high=False, top_k=top_k)

def closeness_low_grad_high(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=False,
                      metric="gradient", metric_high=True, top_k=top_k)

def closeness_low_grad_low(cent, adv, top_k=None):
    return _select_by(cent, adv, centrality="closeness", centrality_high=False,
                      metric="gradient", metric_high=False, top_k=top_k)


def low_degree(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["degree"])

def high_degree(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["degree"], reverse=True)

def low_closeness(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["closeness"])

def high_closeness(cent):
    return sorted((tuple(t) for t in cent.keys()), key=lambda t: cent[t]["closeness"], reverse=True)


def select_adverserial_triples_blackbox(
        proxy_model,
        triples,
        entity_emb,
        relation_emb,
        loss_fn,
        top_k,
        corruption_type,
        device
        ):

    entity_list = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relation_list = list(set([r for _, r, _ in triples]))

    adverserial_triples = []

    for triple in triples:
        h, r, t = triple

        attempts = 0
        max_attempts = 10

        while True:
            attempts += 1
            if corruption_type == 'all':
                corrupt_h = random.choice([i for i in entity_list if i != h])
                corrupt_r = random.choice([i for i in relation_list if i != r])
                corrupt_t = random.choice([i for i in entity_list if i != t])
                corrupted = (corrupt_h, corrupt_r, corrupt_t)
            elif corruption_type == 'head':
                corrupt_h = random.choice([i for i in entity_list if i != h])
                corrupted = (corrupt_h, r, t)
            elif corruption_type == 'rel':
                corrupt_r = random.choice([i for i in relation_list if i != r])
                corrupted = (h, corrupt_r, t)
            elif corruption_type == 'tail':
                corrupt_t = random.choice([i for i in entity_list if i != t])
                corrupted = (h, r, corrupt_t)
            elif corruption_type == 'head-tail':
                corrupt_h = random.choice([i for i in entity_list if i != h])
                corrupt_t = random.choice([i for i in entity_list if i != t])
                corrupted = (corrupt_h, r, corrupt_t)
            else:
                raise ValueError("Invalid corruption_type")

            if corrupted not in triples:
                corrupted_triple = corrupted
                break

            if attempts >= max_attempts:
                break

        proxy_model.train()
        proxy_model.zero_grad()

        h, r, t = corrupted_triple
        h_emb = entity_emb[h].to(device)
        r_emb = relation_emb[r].to(device)
        t_emb = entity_emb[t].to(device)

        x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)
        pred = proxy_model(x)

        label = torch.tensor([1.0], dtype=torch.float)
        loss = loss_fn(pred, label)
        loss.backward()

        param_norm = 0
        for param in proxy_model.parameters():
            if param.grad is not None:
                param_norm += param.grad.norm().item()

        #if (not math.isnan(param_norm)) and param_norm != 0.0:
        adverserial_triples.append((corrupted_triple, pred.item(), param_norm))

    adverserial_triples.sort(key=lambda x: x[1], reverse=True)
    high_scores = adverserial_triples

    adverserial_triples.sort(key=lambda x: x[1], reverse=False)
    low_scores = adverserial_triples

    mixed_scores = low_scores[:top_k // 2] + high_scores[:top_k // 2]

    #----------------------------------------------------------------

    adverserial_triples.sort(key=lambda x: x[2], reverse=True)
    high_gradients = adverserial_triples

    adverserial_triples.sort(key=lambda x: x[2], reverse=False)
    low_gradients = adverserial_triples

    mixed_gradients = low_scores[:top_k // 2] + high_scores[:top_k // 2]

    triples_with_low_score_high_gradient = select_low_score_high_gradient(adverserial_triples, score_percentile=50)


    return high_scores, low_scores, mixed_scores, high_gradients, low_gradients, mixed_gradients, triples_with_low_score_high_gradient


"""
def fgsm_delta(grad: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:
    if norm == "linf":
        return eps * grad.sign()
    if norm == "l2":
        gnorm = grad.norm() + 1e-12
        return eps * (grad / gnorm)
    raise ValueError("norm must be 'linf' or 'l2'")


def nearest_idx_excluding(vec: torch.Tensor, table: torch.Tensor, exclude_idx: int) -> int:
    # vec: [d]; table: [N, d]
    v = vec.unsqueeze(0)               # [1, d]
    d = torch.cdist(v, table)[0]       # [N]
    d[exclude_idx] = float("inf")      # force change
    return int(torch.argmin(d).item())


def logits_for_indices(model, h_i: int, r_i: int, t_i: int, device) -> torch.Tensor:
    idxs = torch.tensor([[h_i, r_i, t_i]], dtype=torch.long, device=device)
    return model.forward_triples(idxs)    # raw model score (logit or energy)


# ---------- main FGSM selector with global polarity flip ----------

def select_adversarial_triples_fgsm(
    triples,
    corruption_type,
    oracle,
    seed=None,
    eps: float = 1e-2,
    norm: str = "linf",
    SIGN: int = -1,  # <<< GLOBAL FLIP: -1 if model returns energy (lower=true); +1 if real logits (higher=true)
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = next(oracle.model.parameters()).device

    # maps
    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}
    I2R = {i: r for r, i in R2I.items()}

    # set of clean triples in index space (fast membership)
    triples_set_idx = {(E2I[h], R2I[r], E2I[t]) for (h, r, t) in triples}

    # embedding modules
    ent_emb = oracle.model.entity_embeddings
    rel_emb = oracle.model.relation_embeddings
    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        raise RuntimeError("Expected nn.Embedding at oracle.model.entity_embeddings / relation_embeddings")

    # frozen copies for nearest snapping
    E = ent_emb.weight.detach()    # [n_ent, d]
    R = rel_emb.weight.detach()    # [n_rel, d]

    adverserial_triples = []

    for (h, r, t) in triples:
        h_i = E2I[h]
        r_i = R2I[r]
        t_i = E2I[t]

        # ---- 1) grads on CLEAN triple (target=1) with GLOBAL SIGN ----
        for p in oracle.model.parameters():
            p.requires_grad_(True)
        oracle.model.zero_grad(set_to_none=True)
        oracle.model.train(False)

        raw_clean = logits_for_indices(oracle.model, h_i, r_i, t_i, device)  # raw score (logit OR energy)
        logits = SIGN * raw_clean                                            # global polarity fix
        target_pos = torch.ones_like(logits)
        loss = F.binary_cross_entropy_with_logits(logits, target_pos)
        loss.backward()

        GE = ent_emb.weight.grad
        GR = rel_emb.weight.grad
        if GE is None or GR is None:
            raise RuntimeError("Embedding grads are None; ensure forward_triples doesn't detach / no_grad.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h = GE[h_i]   # [d]
        g_r = GR[r_i]   # [d]
        g_t = GE[t_i]   # [d]

        triple_grad_norm = torch.linalg.vector_norm(torch.cat([g_h.flatten(), g_r.flatten(), g_t.flatten()]))

        # ---- 2) FGSM steps in embedding space ----
        dh = fgsm_delta(g_h, eps, norm)
        dr = fgsm_delta(g_r, eps, norm)
        dt = fgsm_delta(g_t, eps, norm)

        Eh = E[h_i] + dh
        Er = R[r_i] + dr
        Et = E[t_i] + dt

        # nearest valid symbols (exclude originals)
        h_adv_i = nearest_idx_excluding(Eh, E, h_i)
        r_adv_i = nearest_idx_excluding(Er, R, r_i)
        t_adv_i = nearest_idx_excluding(Et, E, t_i)

        # ---- 3) choose adversarial triple by corruption_type ----
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
            # evaluate three single-field changes; pick the one with MAX BCE on SIGN*raw
            with torch.no_grad():
                lh = F.binary_cross_entropy_with_logits(
                    SIGN * logits_for_indices(oracle.model, h_adv_i, r_i, t_i, device),
                    target_pos
                ).item()
                lr = F.binary_cross_entropy_with_logits(
                    SIGN * logits_for_indices(oracle.model, h_i, r_adv_i, t_i, device),
                    target_pos
                ).item()
                lt = F.binary_cross_entropy_with_logits(
                    SIGN * logits_for_indices(oracle.model, h_i, r_i, t_adv_i, device),
                    target_pos
                ).item()
            if lh >= lr and lh >= lt:
                cand = (h_adv_i, r_i, t_i)
            elif lr >= lt:
                cand = (h_i, r_adv_i, t_i)
            else:
                cand = (h_i, r_i, t_adv_i)
        else:
            raise ValueError("Invalid corruption_type")

        # ---- 4) ensure candidate is not the original and not a clean triple ----
        attempts = 0
        max_attempts = 5
        while (cand == (h_i, r_i, t_i)) or (cand in triples_set_idx):
            attempts += 1
            if attempts > max_attempts:
                break
            # mask current choice and re-pick nearest for the changed field(s)
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

        # ---- 5) evaluate adversarial triple and store (ALWAYS using SIGN) ----
        h_c, r_c, t_c = cand
        raw_adv = logits_for_indices(oracle.model, h_c, r_c, t_c, device)
        pred_prob = torch.sigmoid(SIGN * raw_adv).item()  # probability after polarity fix

        corrupted_triple = (I2E[h_c], I2R[r_c], I2E[t_c])  # names
        clean_triple = (h, r, t)
        adverserial_triples.append(
            (corrupted_triple, clean_triple, pred_prob, float(triple_grad_norm))
        )

    # ---- 6) rank selections ----
    low_scores = sorted(adverserial_triples.copy(), key=itemgetter(2))                 # ascending prob
    corrupted_centerality = compute_triple_centrality(triples, adverserial_triples)    # your function
    high_close = high_closeness(corrupted_centerality)                                 # your function
    high_gradients = sorted(adverserial_triples.copy(), key=itemgetter(3), reverse=True)

    return low_scores, high_close, high_gradients
"""


# ------- helpers (top-level) -------

def fgsm_delta(grad: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:
    if norm == "linf":
        return eps * grad.sign()
    if norm == "l2":
        gnorm = grad.norm() + 1e-12
        return eps * (grad / gnorm)
    raise ValueError("norm must be 'linf' or 'l2'")

def nearest_idx_excluding(vec: torch.Tensor, table: torch.Tensor, exclude_idx: int) -> int:
    # vec: [d], table: [N, d]
    v = vec.unsqueeze(0)               # [1, d]
    d = torch.cdist(v, table)[0]       # [N] (L2 distances)
    d[exclude_idx] = float("inf")      # force a change
    return int(torch.argmin(d).item())

def logits_for_indices(model, h_i: int, r_i: int, t_i: int, device) -> torch.Tensor:
    idxs = torch.tensor([[h_i, r_i, t_i]], dtype=torch.long, device=device)
    return model.forward_triples(idxs)  # raw logits expected (do not sigmoid here)


# ------- main FGSM selector on CLEAN triples -------
"""
def select_adversarial_triples_fgsm(
    triples,
    corruption_type,
    oracle,
    seed,
    eps: float = 1e-2,   # try 1e-3 .. 5e-2 depending on embedding scale
    norm: str = "linf"   # "linf" (sign step) or "l2"
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = next(oracle.model.parameters()).device

    # maps and reverse maps
    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}
    I2R = {i: r for r, i in R2I.items()}

    # for fast membership checks (avoid generating an existing clean triple)
    triples_set_idx = {(E2I[h], R2I[r], E2I[t]) for (h, r, t) in triples}

    # embedding modules (assumed nn.Embedding)
    ent_emb = oracle.model.entity_embeddings
    rel_emb = oracle.model.relation_embeddings
    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        raise RuntimeError("Expected nn.Embedding at model.entity_embeddings / relation_embeddings")

    # frozen tables for nearest snapping
    E = ent_emb.weight.detach()  # [n_ent, d]
    R = rel_emb.weight.detach()  # [n_rel, d]

    adverserial_triples = []

    for (h, r, t) in triples:
        h_i = E2I[h]
        r_i = R2I[r]
        t_i = E2I[t]

        # ---- 1) gradients on the CLEAN triple (label = 1) ----
        for p in oracle.model.parameters():
            p.requires_grad_(True)
        oracle.model.zero_grad(set_to_none=True)
        oracle.model.train(False)

        logits_clean = logits_for_indices(oracle.model, h_i, r_i, t_i, device)  # RAW logits
        target_pos = torch.ones_like(logits_clean)
        loss = F.binary_cross_entropy_with_logits(logits_clean, target_pos)
        loss.backward()

        GE = ent_emb.weight.grad
        GR = rel_emb.weight.grad
        if GE is None or GR is None:
            raise RuntimeError("Embedding grads are None; ensure forward_triples doesn't detach / no_grad.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h = GE[h_i]      # [d]
        g_r = GR[r_i]      # [d]
        g_t = GE[t_i]      # [d]

        triple_grad_norm = torch.linalg.vector_norm(torch.cat([g_h.flatten(), g_r.flatten(), g_t.flatten()]))

        # ---- 2) FGSM step in embedding space ----
        dh = fgsm_delta(g_h, eps, norm)
        dr = fgsm_delta(g_r, eps, norm)
        dt = fgsm_delta(g_t, eps, norm)

        Eh = E[h_i] + dh
        Er = R[r_i] + dr
        Et = E[t_i] + dt

        # nearest valid symbols (exclude originals)
        h_adv_i = nearest_idx_excluding(Eh, E, h_i)
        r_adv_i = nearest_idx_excluding(Er, R, r_i)
        t_adv_i = nearest_idx_excluding(Et, E, t_i)

        # ---- 3) pick candidate according to corruption_type ----
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
            # evaluate head/rel/tail changes; pick the one with highest loss on the clean label
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

        # ---- 4) avoid duplicates (original or existing clean triples) ----
        attempts = 0
        max_attempts = 5
        while (cand == (h_i, r_i, t_i)) or (cand in triples_set_idx):
            attempts += 1
            if attempts > max_attempts:
                break
            # mask current choice and re-pick nearest for the changed field(s)
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

        # ---- 5) evaluate and store ----
        h_c, r_c, t_c = cand
        logits_adv = logits_for_indices(oracle.model, h_c, r_c, t_c, device)
        pred_prob = torch.sigmoid(logits_adv).item()  # probability after corruption

        corrupted_triple = (I2E[h_c], I2R[r_c], I2E[t_c])
        clean_triple = (h, r, t)
        adverserial_triples.append(
            (corrupted_triple, clean_triple, pred_prob, float(triple_grad_norm))
        )

    # ---- 6) build outputs ----
    fgsm_adverserial_triples = adverserial_triples.copy()
    low_scores = sorted(adverserial_triples.copy(), key=itemgetter(2))                 # ascending prob
    corrupted_centerality = compute_triple_centrality(triples, adverserial_triples.copy())    # your function
    high_close = high_closeness(corrupted_centerality)                                 # your function
    high_gradients = sorted(adverserial_triples.copy(), key=itemgetter(3), reverse=True)

    return low_scores, high_close, high_gradients, fgsm_adverserial_triples
"""


def select_adversarial_triples_fgsm(
    triples,
    corruption_type,
    oracle,
    seed,
    eps: float = 1e-2,
    norm: str = "linf",
    avoid_triples_idx=None,  # optional: pass a set of (h_i,r_i,t_i) from train+valid+test
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = next(oracle.model.parameters()).device

    # maps
    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}
    I2R = {i: r for r, i in R2I.items()}

    # base "seen" set: everything we must avoid
    base_seen = {(E2I[h], R2I[r], E2I[t]) for (h, r, t) in triples}
    if avoid_triples_idx is not None:
        base_seen = set(base_seen) | set(avoid_triples_idx)
    seen = set(base_seen)  # will also include newly created adversarials

    # embeddings
    ent_emb = oracle.model.entity_embeddings
    rel_emb = oracle.model.relation_embeddings
    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        raise RuntimeError("Expected nn.Embedding at model.entity_embeddings / relation_embeddings")

    # frozen tables
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
                    logits_for_indices(oracle.model, h_adv_i, r_i, t_i, device), target_pos
                ).item()
                lr = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, h_i, r_adv_i, t_i, device), target_pos
                ).item()
                lt = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, h_i, r_i, t_adv_i, device), target_pos
                ).item()
            if lh >= lr and lh >= lt:
                cand = (h_adv_i, r_i, t_i)
            elif lr >= lt:
                cand = (h_i, r_adv_i, t_i)
            else:
                cand = (h_i, r_i, t_adv_i)
        else:
            raise ValueError("Invalid corruption_type")

        # small retry to walk to next-nearest if duplicate/unchanged

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

        # FINAL GUARD: if still duplicate or unchanged, skip
        if (cand == (h_i, r_i, t_i)) or (cand in seen):
            continue

        # mark as seen to prevent future duplicates across outputs
        seen.add(cand)

        # evaluate/store
        h_c, r_c, t_c = cand
        logits_adv = logits_for_indices(oracle.model, h_c, r_c, t_c, device)
        pred_prob = torch.sigmoid(logits_adv).item()

        corrupted_triple = (I2E[h_c], I2R[r_c], I2E[t_c])
        clean_triple = (h, r, t)
        adverserial_triples.append((corrupted_triple, clean_triple, pred_prob, float(triple_grad_norm)))

    # build outputs (unchanged from your version)
    fgsm_adverserial_triples = adverserial_triples.copy()
    low_scores = sorted(adverserial_triples.copy(), key=itemgetter(2))
    corrupted_centerality = compute_triple_centrality(triples, adverserial_triples.copy())
    high_close = high_closeness(corrupted_centerality)
    high_gradients = sorted(adverserial_triples.copy(), key=itemgetter(3), reverse=True)

    return low_scores, high_close, high_gradients, fgsm_adverserial_triples



def select_adversarial_triples_fgsm_simplified(
    triples,
    oracle,
    seed,
    eps: float = 1e-2,
    norm: str = "linf"
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = next(oracle.model.parameters()).device

    # maps and reverse maps
    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}
    I2R = {i: r for r, i in R2I.items()}

    # for fast membership checks (avoid generating an existing clean triple)
    triples_set_idx = {(E2I[h], R2I[r], E2I[t]) for (h, r, t) in triples}

    # embedding modules (assumed nn.Embedding)
    ent_emb = oracle.model.entity_embeddings
    rel_emb = oracle.model.relation_embeddings
    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        raise RuntimeError("Expected nn.Embedding at model.entity_embeddings / relation_embeddings")

    # frozen tables for nearest snapping
    E = ent_emb.weight.detach()  # [n_ent, d]
    R = rel_emb.weight.detach()  # [n_rel, d]

    adverserial_triples = []

    for (h, r, t) in triples:
        h_i = E2I[h]
        r_i = R2I[r]
        t_i = E2I[t]

        # ---- 1) gradients on the CLEAN triple (label = 1) ----
        for p in oracle.model.parameters():
            p.requires_grad_(True)
        oracle.model.zero_grad(set_to_none=True)
        oracle.model.train(False)

        logits_clean = logits_for_indices(oracle.model, h_i, r_i, t_i, device)  # RAW logits
        target_pos = torch.ones_like(logits_clean)
        loss = F.binary_cross_entropy_with_logits(logits_clean, target_pos)
        loss.backward()

        GE = ent_emb.weight.grad
        GR = rel_emb.weight.grad
        if GE is None or GR is None:
            raise RuntimeError("Embedding grads are None; ensure forward_triples doesn't detach / no_grad.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h = GE[h_i]      # [d]
        g_r = GR[r_i]      # [d]
        g_t = GE[t_i]      # [d]

        triple_grad_norm = torch.linalg.vector_norm(torch.cat([g_h.flatten(), g_r.flatten(), g_t.flatten()]))

        # ---- 2) FGSM step in embedding space ----
        dh = fgsm_delta(g_h, eps, norm)
        dr = fgsm_delta(g_r, eps, norm)
        dt = fgsm_delta(g_t, eps, norm)

        Eh = E[h_i] + dh
        Er = R[r_i] + dr
        Et = E[t_i] + dt

        # nearest valid symbols (exclude originals)
        h_adv_i = nearest_idx_excluding(Eh, E, h_i)
        r_adv_i = nearest_idx_excluding(Er, R, r_i)
        t_adv_i = nearest_idx_excluding(Et, E, t_i)

        # ---- 3) pick candidate according to corruption_type ----

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

        # ---- 4) avoid duplicates (original or existing clean triples) ----
        attempts = 0
        max_attempts = 5
        while (cand == (h_i, r_i, t_i)) or (cand in triples_set_idx):
            attempts += 1
            if attempts > max_attempts:
                break
            # mask current choice and re-pick nearest for the changed field(s)
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

        # ---- 5) evaluate and store ----
        h_c, r_c, t_c = cand
        logits_adv = logits_for_indices(oracle.model, h_c, r_c, t_c, device)
        pred_prob = torch.sigmoid(logits_adv).item()  # probability after corruption

        corrupted_triple = (I2E[h_c], I2R[r_c], I2E[t_c])
        clean_triple = (h, r, t)
        adverserial_triples.append(
            (corrupted_triple, clean_triple, pred_prob, float(triple_grad_norm))
        )

    # ---- 6) build outputs ----
    fgsm_adverserial_triples = adverserial_triples.copy()
    low_scores = sorted(adverserial_triples.copy(), key=itemgetter(2))
    corrupted_centerality = compute_triple_centrality(triples, adverserial_triples.copy())
    high_close = high_closeness(corrupted_centerality)
    high_gradients = sorted(adverserial_triples.copy(), key=itemgetter(3), reverse=True)

    return low_scores, high_close, high_gradients, fgsm_adverserial_triples


def select_adverserial_triples_whitebox(
        entity_emb,
        relation_emb,
        triples,
        corruption_type,
        oracle,
        seed,
        device="cpu"
):

    random.seed(seed)
    torch.manual_seed(seed)

    entity_list = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relation_list = list(set([r for _, r, _ in triples]))

    adverserial_triples = []

    #all_combinations = list(product(entity_list, relation_list, entity_list))
    #for h, r, t in all_combinations[:50000]:

    for triple in triples:
        h, r, t = triple

        attempts = 0
        max_attempts = 10

        while True:
            attempts += 1
            if corruption_type == 'all':
                corrupt_h = random.choice([i for i in entity_list if i != h])
                corrupt_r = random.choice([i for i in relation_list if i != r])
                corrupt_t = random.choice([i for i in entity_list if i != t])
                corrupted = (corrupt_h, corrupt_r, corrupt_t)
            elif corruption_type == 'head':
                corrupt_h = random.choice([i for i in entity_list if i != h])
                corrupted = (corrupt_h, r, t)
            elif corruption_type == 'rel':
                corrupt_r = random.choice([i for i in relation_list if i != r])
                corrupted = (h, corrupt_r, t)
            elif corruption_type == 'tail':
                corrupt_t = random.choice([i for i in entity_list if i != t])
                corrupted = (h, r, corrupt_t)
            elif corruption_type == 'head-tail':
                corrupt_h = random.choice([i for i in entity_list if i != h])
                corrupt_t = random.choice([i for i in entity_list if i != t])
                corrupted = (corrupt_h, r, corrupt_t)
            elif corruption_type == 'head-rel':
                corrupt_h = random.choice([i for i in entity_list if i != h])
                corrupt_r = random.choice([i for i in relation_list if i != r])
                corrupted = (corrupt_h, corrupt_r, t)
            elif corruption_type == 'random-one':
                choice = random.choice(['head', 'rel', 'tail'])
                if choice == 'head':
                    corrupt_h = random.choice([i for i in entity_list if i != h])
                    corrupted = (corrupt_h, r, t)
                elif choice == 'rel':
                    corrupt_r = random.choice([i for i in relation_list if i != r])
                    corrupted = (h, corrupt_r, t)
                else:
                    corrupt_t = random.choice([i for i in entity_list if i != t])
                    corrupted = (h, r, corrupt_t)
            else:
                raise ValueError("Invalid corruption_type")

            if corrupted not in triples:
                corrupted_triple = corrupted
                clean_triple = triple
                break

            if attempts >= max_attempts:
                break

        hc, rc, tc = corrupted_triple

        for p in oracle.model.parameters():
            p.requires_grad_(True)

        device = next(oracle.model.parameters()).device
        idxs = torch.tensor([[oracle.entity_to_idx[hc],
                              oracle.relation_to_idx[rc],
                              oracle.entity_to_idx[tc]]],
                            dtype=torch.long, device=device)

        oracle.model.zero_grad()
        oracle.model.train(False)

        logits = oracle.model.forward_triples(idxs)
        target = torch.ones_like(logits)
        pred_prob = torch.sigmoid(logits)

        loss = F.binary_cross_entropy_with_logits(logits, target)

        loss.backward()

        GE = oracle.model.entity_embeddings.weight.grad
        GR = oracle.model.relation_embeddings.weight.grad
        if GE is None or GR is None:
            raise RuntimeError(
                "Embedding grads are None. Check that forward_triples doesn't detach or run under no_grad.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        h_i = oracle.entity_to_idx[hc]
        r_i = oracle.relation_to_idx[rc]
        t_i = oracle.entity_to_idx[tc]

        g_h = GE[h_i]
        g_r = GR[r_i]
        g_t = GE[t_i]

        # local gradient
        triple_grad_norm_local = torch.linalg.vector_norm(torch.cat([g_h.flatten(), g_r.flatten(), g_t.flatten()]))


        # global gradient
        """
        total = torch.zeros((), device=device)
        for p in oracle.model.parameters():
            if p.grad is None:
                continue
            g = p.grad
            if g.is_sparse:
                g = g.coalesce().values()
            total = total + (g * g).sum()
        triple_grad_norm_global = total.sqrt()
        """

        adverserial_triples.append(
            (corrupted_triple,
             clean_triple,
             pred_prob.item(),
             triple_grad_norm_local.item(),
             #triple_grad_norm_global.item()
             ))

    low_scores = sorted(adverserial_triples.copy(), key=lambda x: x[2])  # ascending

    #clean_version_of_corrupted_triples = [item[1] for item in adverserial_triples]
    #triples_without_edited_ones = [t for t in triples if t not in clean_version_of_corrupted_triples]

    corrupted_centerality = compute_triple_centrality(triples, adverserial_triples)
    high_close = high_closeness(corrupted_centerality)

    high_gradients_local = sorted(adverserial_triples.copy(), key=lambda x: x[3], reverse=True)
    #high_gradients_global = sorted(adverserial_triples.copy(), key=lambda x: x[4], reverse=True)

    return (
            low_scores,
            high_close,
            high_gradients_local,
            #high_gradients_global
            )

def save_triples(triple_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triple_list:
            f.write(f"{h}\t{r}\t{t}\n")