import pandas as pd
import os
import torch, random, heapq, statistics
import math
import seaborn as sns
import numpy as np
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import statistics
from itertools import product
import matplotlib


matplotlib.use("Agg")


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_triple_centrality(graph_triples, query_triples):

    G = nx.Graph()

    for h, r, t in graph_triples + query_triples:
        G.add_edge(h, t, relation=r)

    deg, clo = nx.degree_centrality(G), nx.closeness_centrality(G)

    results = {}
    for h, r, t in query_triples:
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

def get_low_score_high_gradient_triples(triples_list, top_k):
    scores = [float(item[1]) for item in triples_list]
    avg_score = sum(scores) / len(scores)

    filtered = []
    for item in triples_list:
        score = float(item[1])
        if score < avg_score:
            filtered.append((item[0], item[1], item[2]))

    filtered.sort(key=lambda x: x[2], reverse=True)

    return filtered[:top_k]

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

def select_low_score_high_gradient(triples_data, k, score_percentile):
    triples_data_sorted = sorted(triples_data, key=lambda x: x[1])

    cutoff_index = max(1, int(len(triples_data) * (score_percentile / 100)))
    low_score_candidates = triples_data_sorted[:cutoff_index]

    low_score_high_grad = sorted(low_score_candidates, key=lambda x: x[2], reverse=True)

    return low_score_high_grad[:k]

def select_high_score_high_gradient(triples_data, k, score_percentile):
    triples_by_score = sorted(triples_data, key=lambda x: x[1], reverse=True)

    cutoff = max(1, int(len(triples_by_score) * (score_percentile / 100)))
    high_score_candidates = triples_by_score[:cutoff]

    candidates_by_grad = sorted(high_score_candidates, key=lambda x: x[2], reverse=True)

    return candidates_by_grad[:k]

def select_low_score_low_gradient(triples_data, k, score_percentile):
    triples_by_score = sorted(triples_data, key=lambda x: x[1])
    cutoff = max(1, int(len(triples_by_score) * (score_percentile / 100)))
    low_score_candidates = triples_by_score[:cutoff]

    low_grad = sorted(low_score_candidates, key=lambda x: x[2])
    return low_grad[:k]

def select_high_score_low_gradient(triples_data, k, score_percentile):
    triples_by_score = sorted(triples_data, key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(triples_by_score) * (score_percentile / 100)))
    high_score_candidates = triples_by_score[:cutoff]

    low_grad = sorted(high_score_candidates, key=lambda x: x[2])
    return low_grad[:k]


def get_most_impactful_triples(triples_data, k, weights=None):
    if weights is None:
        weights = {'centrality': 0.1, 'gradient': 0.45, 'score': 0.45}

    scores = np.array([item[1] for item in triples_data])
    gradients = np.array([item[2] for item in triples_data])
    centralities = np.array([np.mean(list(item[3].values())) for item in triples_data])

    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    norm_scores = 1 - norm_scores

    norm_gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-12)
    norm_centralities = (centralities - centralities.min()) / (centralities.max() - centralities.min() + 1e-12)

    combined = (
        weights['centrality'] * norm_centralities +
        weights['gradient'] * norm_gradients +
        weights['score'] * norm_scores
    )

    sorted_indices = np.argsort(-combined)
    top_triples = []
    for idx in sorted_indices[:k]:
        triple, score, grad, cent_dict = triples_data[idx]
        top_triples.append((triple, combined[idx], score, grad, cent_dict))

    return top_triples


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _score_grad_dict(score_grad_info):
    """{triple: {"score": s, "gradient": g}}"""
    return {t: {"score": s, "gradient": g} for t, s, g in score_grad_info}

# ------------------------------------------------------------------
# sorters
# ------------------------------------------------------------------
def sort_by_degree(cent_dict, high=True):
    """Return triples ordered by degree."""
    return sorted(cent_dict, key=lambda t: cent_dict[t]["degree"], reverse=high)


def sort_by_closeness(cent_dict, high=True):
    """Return triples ordered by closeness."""
    return sorted(cent_dict, key=lambda t: cent_dict[t]["closeness"], reverse=high)


# ------------------------------------------------------------------
# selectors
# ------------------------------------------------------------------
def pick_by_score(triples, sg_dict, k, high=True):
    triples = [t for t in triples if t in sg_dict]
    triples.sort(key=lambda t: sg_dict[t]["score"], reverse=high)
    return triples[:k]


def pick_by_gradient(triples, sg_dict, k, high=True):
    triples = [t for t in triples if t in sg_dict]
    triples.sort(key=lambda t: sg_dict[t]["gradient"], reverse=high)
    return triples[:k]


# ------------------------------------------------------------------
# combined functions
# ------------------------------------------------------------------
def degree_high_score_high(cent, sg, k):
    return pick_by_score(sort_by_degree(cent, high=True), _score_grad_dict(sg), k, high=True)

def degree_high_score_low(cent, sg, k):
    return pick_by_score(sort_by_degree(cent, high=True), _score_grad_dict(sg), k, high=False)

def degree_low_score_high(cent, sg, k):
    return pick_by_score(sort_by_degree(cent, high=False), _score_grad_dict(sg), k, high=True)

def degree_low_score_low(cent, sg, k):
    return pick_by_score(sort_by_degree(cent, high=False), _score_grad_dict(sg), k, high=False)

def degree_high_grad_high(cent, sg, k):
    return pick_by_gradient(sort_by_degree(cent, high=True), _score_grad_dict(sg), k, high=True)

def degree_high_grad_low(cent, sg, k):
    return pick_by_gradient(sort_by_degree(cent, high=True), _score_grad_dict(sg), k, high=False)

def degree_low_grad_high(cent, sg, k):
    return pick_by_gradient(sort_by_degree(cent, high=False), _score_grad_dict(sg), k, high=True)

def degree_low_grad_low(cent, sg, k):
    return pick_by_gradient(sort_by_degree(cent, high=False), _score_grad_dict(sg), k, high=False)

def closeness_high_score_high(cent, sg, k):
    return pick_by_score(sort_by_closeness(cent, high=True), _score_grad_dict(sg), k, high=True)

def closeness_high_score_low(cent, sg, k):
    return pick_by_score(sort_by_closeness(cent, high=True), _score_grad_dict(sg), k, high=False)

def closeness_low_score_high(cent, sg, k):
    return pick_by_score(sort_by_closeness(cent, high=False), _score_grad_dict(sg), k, high=True)

def closeness_low_score_low(cent, sg, k):
    return pick_by_score(sort_by_closeness(cent, high=False), _score_grad_dict(sg), k, high=False)

def closeness_high_grad_high(cent, sg, k):
    return pick_by_gradient(sort_by_closeness(cent, high=True), _score_grad_dict(sg), k, high=True)

def closeness_high_grad_low(cent, sg, k):
    return pick_by_gradient(sort_by_closeness(cent, high=True), _score_grad_dict(sg), k, high=False)

def closeness_low_grad_high(cent, sg, k):
    return pick_by_gradient(sort_by_closeness(cent, high=False), _score_grad_dict(sg), k, high=True)

def closeness_low_grad_low(cent, sg, k):
    return pick_by_gradient(sort_by_closeness(cent, high=False), _score_grad_dict(sg), k, high=False)

def low_degree(cent, k):
    return sort_by_degree(cent, high=False)[:k]

def high_degree(cent, k):
    return sort_by_degree(cent, high=True)[:k]

def low_openness(cent, k):
    return sort_by_closeness(cent, high=False)[:k]

def high_openness(cent, k):
    return sort_by_closeness(cent, high=True)[:k]


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
    high_scores = adverserial_triples[:top_k]

    adverserial_triples.sort(key=lambda x: x[1], reverse=False)
    low_scores = adverserial_triples[:top_k]

    mixed_scores = low_scores[:top_k // 2] + high_scores[:top_k // 2]

    #----------------------------------------------------------------

    adverserial_triples.sort(key=lambda x: x[2], reverse=True)
    high_gradients = adverserial_triples[:top_k]

    adverserial_triples.sort(key=lambda x: x[2], reverse=False)
    low_gradients = adverserial_triples[:top_k]

    mixed_gradients = low_scores[:top_k // 2] + high_scores[:top_k // 2]

    triples_with_low_score_high_gradient = select_low_score_high_gradient(adverserial_triples, top_k, score_percentile=50)


    return high_scores, low_scores, mixed_scores, high_gradients, low_gradients, mixed_gradients, triples_with_low_score_high_gradient

def select_adverserial_triples_whitebox(
        triples,
        top_k,
        corruption_type,
        oracle,
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

        hc, rc, tc = corrupted_triple

        idxs = torch.LongTensor([
            oracle.entity_to_idx[hc],
            oracle.relation_to_idx[rc],
            oracle.entity_to_idx[tc]
        ]).unsqueeze(0)

        for param in oracle.model.parameters():
            param.requires_grad = True

        oracle.model.train()
        oracle.model.zero_grad()

        pred = oracle.model.forward_triples(idxs)
        pred_prob = torch.sigmoid(pred)

        label = torch.tensor([1.0], dtype=torch.float)
        loss = oracle.model.loss(pred_prob, label, current_epoch=101)
        #loss = loss_fn(pred_prob, label)
        loss.backward()

        oracle_grad_norm = torch.norm(
            torch.cat([p.grad.view(-1) for p in oracle.model.parameters() if p.grad is not None])
        )

        #if not torch.isnan(oracle_grad_norm) and oracle_grad_norm.item() > 0:
        adverserial_triples.append(
            (corrupted_triple, pred_prob.item(), oracle_grad_norm.item()))

    mixed_low, mixed_high = top_k // 2, (top_k + 1) // 2


    adverserial_triples.sort(key=lambda x: x[1], reverse=True)
    high_scores = adverserial_triples[:top_k]

    adverserial_triples.sort(key=lambda x: x[1], reverse=False)
    low_scores = adverserial_triples[:top_k]

    mixed_scores = low_scores[:mixed_low] + high_scores[:mixed_high]

    # ----------------------------------------------------------------

    adverserial_triples.sort(key=lambda x: x[2], reverse=True)
    high_gradients = adverserial_triples[:top_k]

    adverserial_triples.sort(key=lambda x: x[2], reverse=False)
    low_gradients = adverserial_triples[:top_k]

    mixed_gradients = low_scores[:mixed_low] + high_scores[:mixed_high]

    triples_with_low_score_high_gradient = select_low_score_high_gradient(adverserial_triples, top_k,
                                                                          score_percentile=50)

    triples_with_high_score_high_gradient = select_high_score_high_gradient(adverserial_triples, top_k, score_percentile=50)

    triples_with_low_score_low_gradient = select_low_score_low_gradient(adverserial_triples, top_k, score_percentile=50)

    triples_with_high_score_low_gradient = select_high_score_low_gradient(adverserial_triples, top_k, score_percentile=50)

    corrupted_triples = [item[0] for item in adverserial_triples]

    corrupted_centerality = compute_triple_centrality(triples, corrupted_triples)

    adverserial_triples_high_scores = [item[0] for item in high_scores]
    adverserial_triples_low_scores = [item[0] for item in low_scores]
    adverserial_triples_mixed_scores = [item[0] for item in mixed_scores]
    adverserial_triples_high_gradients = [item[0] for item in high_gradients]
    adverserial_triples_low_gradients = [item[0] for item in low_gradients]
    adverserial_triples_mixed_gradients = [item[0] for item in mixed_gradients]
    adverserial_triples_low_score_high_gradient = [item[0] for item in triples_with_low_score_high_gradient]
    adverserial_triples_high_score_high_gradient = [item[0] for item in triples_with_high_score_high_gradient]
    adverserial_triples_low_score_low_gradient = [item[0] for item in triples_with_low_score_low_gradient]
    adverserial_triples_high_score_low_gradient = [item[0] for item in triples_with_high_score_low_gradient]

    degree_high_score_high_triples = degree_high_score_high(corrupted_centerality, adverserial_triples, top_k)
    degree_high_score_low_triples = degree_high_score_low(corrupted_centerality, adverserial_triples, top_k)
    degree_low_score_high_triples = degree_low_score_high(corrupted_centerality, adverserial_triples, top_k)
    degree_low_score_low_triples = degree_low_score_low(corrupted_centerality, adverserial_triples, top_k)

    degree_high_grad_high_triples = degree_high_grad_high(corrupted_centerality, adverserial_triples, top_k)
    degree_high_grad_low_triples = degree_high_grad_low(corrupted_centerality, adverserial_triples, top_k)
    degree_low_grad_high_triples = degree_low_grad_high(corrupted_centerality, adverserial_triples, top_k)
    degree_low_grad_low_triples = degree_low_grad_low(corrupted_centerality, adverserial_triples, top_k)

    closeness_high_score_high_triples = closeness_high_score_high(corrupted_centerality, adverserial_triples, top_k)
    closeness_high_score_low_triples = closeness_high_score_low(corrupted_centerality, adverserial_triples, top_k)
    closeness_low_score_high_triples = closeness_low_score_high(corrupted_centerality, adverserial_triples, top_k)
    closeness_low_score_low_triples = closeness_low_score_low(corrupted_centerality, adverserial_triples, top_k)

    closeness_high_grad_high_triples = closeness_high_grad_high(corrupted_centerality, adverserial_triples, top_k)
    closeness_high_grad_low_triples = closeness_high_grad_low(corrupted_centerality, adverserial_triples, top_k)
    closeness_low_grad_high_triples = closeness_low_grad_high(corrupted_centerality, adverserial_triples, top_k)
    closeness_low_grad_low_triples = closeness_low_grad_low(corrupted_centerality, adverserial_triples, top_k)

    lowest_deg = low_degree(corrupted_centerality, top_k)
    highest_deg = high_degree(corrupted_centerality, top_k)
    lowest_opn = low_openness(corrupted_centerality, top_k)
    highest_opn = high_openness(corrupted_centerality, top_k)


    return (
            adverserial_triples_high_scores,
            adverserial_triples_low_scores,
            adverserial_triples_mixed_scores,

            adverserial_triples_high_gradients,
            adverserial_triples_low_gradients,
            adverserial_triples_mixed_gradients,

            adverserial_triples_low_score_high_gradient,
            adverserial_triples_high_score_high_gradient,
            adverserial_triples_low_score_low_gradient,
            adverserial_triples_high_score_low_gradient,

            degree_high_score_high_triples,
            degree_high_score_low_triples,
            degree_low_score_high_triples,
            degree_low_score_low_triples,

            degree_high_grad_high_triples,
            degree_high_grad_low_triples,
            degree_low_grad_high_triples,
            degree_low_grad_low_triples,

            closeness_high_score_high_triples,
            closeness_high_score_low_triples,
            closeness_low_score_high_triples,
            closeness_low_score_low_triples,

            closeness_high_grad_high_triples,
            closeness_high_grad_low_triples,
            closeness_low_grad_high_triples,
            closeness_low_grad_low_triples,

            lowest_deg,
            highest_deg,
            lowest_opn,
            highest_opn,
            )



def save_triples(triple_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triple_list:
            f.write(f"{h}\t{r}\t{t}\n")