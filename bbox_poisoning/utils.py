import pandas as pd
import torch
import random
import os
import numpy as np
import torch.nn as nn
from itertools import product
import torch
from torch.autograd import grad
import random
from itertools import product
import torch, random, statistics
from itertools import product, islice
import random
import statistics
from itertools import product, islice
from typing import List, Tuple, Dict, Iterable
import torch, random, heapq, statistics
from itertools import product, islice
from typing import List, Tuple, Dict, Iterable
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from baselines import poison_random


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
def load_embeddings(entity_csv, relation_csv):
    ent_df = pd.read_csv(entity_csv, index_col=0)
    rel_df = pd.read_csv(relation_csv, index_col=0)
    entity_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in ent_df.iterrows()}
    relation_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in rel_df.iterrows()}
    return entity_emb, relation_emb
"""

def load_embeddings(entity_csv, relation_csv):
    ent_df = pd.read_csv(entity_csv, index_col=0)
    rel_df = pd.read_csv(relation_csv, index_col=0)

    rel_df = rel_df[~rel_df.index.str.contains('_inverse')]
    print("################################## len rel_df : ", len(rel_df))

    entity_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in ent_df.iterrows()}
    relation_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in rel_df.iterrows()}

    return entity_emb, relation_emb

def load_triples(path):
    with open(path) as f:
        return [tuple(line.strip().split()[:3]) for line in f]

def get_local_grad_norm(model, h_idx, r_idx, t_idx):
    entity_grad = model.entity_embeddings.weight.grad
    relation_grad = model.relation_embeddings.weight.grad

    norm_parts = []
    #if entity_grad is not None:
    #    norm_parts.append(entity_grad[h_idx])
    #    norm_parts.append(entity_grad[t_idx])
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

def select_harmful_triples(
    #proxy_model,
    triples,
    entity_emb,
    relation_emb,
    loss_fn,
    num_candidate,
    val_triples,
    oracle,
    top_k,
    corruption_type,
    device='cuda'):

    #proxy_model.eval()
    harmful_triples = []

    for param in oracle.model.parameters():
        param.requires_grad = True

    #random_corruption = poison_random(triples, num_candidate, "rel")

    entity_list = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relation_list = list(set([r for _, r, _ in triples]))

    for triples in triples:
        h, r, t = triples

        if corruption_type == 'all':
            corrupt_h = random.choice(entity_list)
            corrupt_r = random.choice(relation_list)
            corrupt_t = random.choice(entity_list)
            corrupted = (corrupt_h, corrupt_r, corrupt_t)
        if corruption_type == 'head':
            corrupt_h = random.choice(entity_list)
            corrupted = (corrupt_h, r, t)
        if corruption_type == 'rel':
            relation_list_without_r = [i for i in relation_list if i != r]
            corrupt_r = random.choice(relation_list_without_r)
            corrupted = (h, corrupt_r, t)
        if corruption_type == 'tail':
            corrupt_t = random.choice(entity_list)
            corrupted = (h, r, corrupt_t)
        if corruption_type == 'head-tail':
            corrupt_h = random.choice(entity_list)
            corrupt_t = random.choice(entity_list)
            corrupted = (corrupt_h, r, corrupt_t)

        if corrupted != (h, r, t) and corrupted not in triples:

            idxs = torch.LongTensor([
                oracle.entity_to_idx[h],
                oracle.relation_to_idx[r],
                oracle.entity_to_idx[t]
            ]).unsqueeze(0)

            old_local_grad_norm = get_local_grad_norm(oracle.model, oracle.entity_to_idx[h],
                                                  oracle.relation_to_idx[r],
                                                  oracle.entity_to_idx[t])

            old_embedding = compute_embedding_change(oracle.model, oracle.entity_to_idx[h],
                                                     oracle.relation_to_idx[r],
                                                     oracle.entity_to_idx[t])

            oracle.model.train()
            oracle.model.zero_grad()
            pred = oracle.model.forward_triples(idxs)
            pred_prob = torch.sigmoid(pred)
            label = torch.tensor([1.0], dtype=torch.float)
            loss = oracle.model.loss(pred_prob, label, current_epoch=101)
            loss.backward()


            new_local_grad_norm = get_local_grad_norm(oracle.model, oracle.entity_to_idx[h],
                                                    oracle.relation_to_idx[r],
                                                    oracle.entity_to_idx[t])

            change = abs(old_local_grad_norm - new_local_grad_norm)
            print("gradient: ", corrupted, change)

            new_embedding = compute_embedding_change(oracle.model, oracle.entity_to_idx[h],
                                                  oracle.relation_to_idx[r],
                                                  oracle.entity_to_idx[t])
            #embedding_change = abs(old_embedding - new_embedding)
            #print("embedding_change: ", corrupted, embedding_change)
            harmful_triples.append((corrupted, change))

    harmful_triples.sort(key=lambda x: x[1], reverse=True) #True

    return harmful_triples[:top_k]

#----------------------------------------------------------------------------------------------------------------------
def get_oracle_score(triple, oracle):
    h, r, t = triple
    idxs = torch.LongTensor([
        oracle.entity_to_idx[h],
        oracle.relation_to_idx[r],
        oracle.entity_to_idx[t]
    ]).unsqueeze(0)
    with torch.no_grad():
        logit = oracle.model.forward_triples(idxs)
        return torch.sigmoid(logit).item()

def evaluate_proxy_model_against_oracle(
    proxy_model,
    oracle_model,
    val_triples,
    entity_emb,
    relation_emb,
    device='cpu'
):
    proxy_model.eval()
    x_batch = []
    oracle_scores = []

    for h, r, t in val_triples:
        try:
            h_emb = entity_emb[h.strip()].to(device)
            r_emb = relation_emb[r.strip()].to(device)
            t_emb = entity_emb[t.strip()].to(device)
        except KeyError as e:
            print(f"Skipping triple due to missing key: {e}")
            continue

        x = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        x_batch.append(x)

        oracle_score = get_oracle_score((h, r, t), oracle_model)
        oracle_scores.append(float(oracle_score))

    if not x_batch:
        raise ValueError("No valid triples to evaluate.")

    x_batch = torch.stack(x_batch).to(device)

    with torch.no_grad():
        proxy_scores = proxy_model(x_batch).squeeze().cpu().numpy()

    mse = mean_squared_error(oracle_scores, proxy_scores)
    mae = mean_absolute_error(oracle_scores, proxy_scores)
    corr, _ = pearsonr(oracle_scores, proxy_scores)

    return {
        'mse': mse,
        'mae': mae,
        'pearson': corr
    }

#----------------------------------------------------------------------------------------------------------------------
def select_easy_negative_triples(
        proxy_model,
        triples,
        entity_emb,
        relation_emb,
        loss_fn,
        num_candidate=50_000,
        top_k_return=1_000,
        device="cpu"
):
    proxy_model.eval()

    entity_list = list(entity_emb.keys())
    relation_list = list(relation_emb.keys())

    all_combinations = list(product(entity_list, relation_list, entity_list))
    random.shuffle(all_combinations)
    candidates = all_combinations[:num_candidate]

    scores = []
    filtered_candidates = []
    with torch.no_grad():
        for corrupted in candidates:
            if corrupted in triples:
                continue

            h, r, t = corrupted
            h_emb = entity_emb[h].to(device)
            r_emb = relation_emb[r].to(device)
            t_emb = entity_emb[t].to(device)

            x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)  # [1, d]
            score = proxy_model(x).item()
            scores.append(score)
            filtered_candidates.append(corrupted)

    if not scores:
        return []

    mean_score = sum(scores) / len(scores)

    easy_negatives = []
    label_zero = torch.tensor([[0.0]], device=device)

    for corrupted, s in zip(filtered_candidates, scores):
        if s >= mean_score:
            continue

        h, r, t = corrupted
        h_emb = entity_emb[h].to(device)
        r_emb = relation_emb[r].to(device)
        t_emb = entity_emb[t].to(device)

        x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)

        pred = proxy_model(x)
        loss = loss_fn(pred, label_zero)

        glist = grad(loss, proxy_model.parameters(),
                     retain_graph=False, create_graph=False)
        grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in glist)).item()

        easy_negatives.append((corrupted, grad_norm))

    easy_negatives.sort(key=lambda tup: tup[1])
    return easy_negatives[:top_k_return]

"""
def select_harmful_triples(
    proxy_model,
    triples,
    entity_emb,
    relation_emb,
    loss_fn,
    num_candidate,
    val_triples,
    device='cuda'
):
    proxy_model.eval()
    harmful_triples = []

    entity_list = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relation_list = list(set([r for _, r, _ in triples]))

    all_combinations = list(product(entity_list, relation_list, entity_list))
    print("Number of possibilities: ", len(all_combinations))

    random.shuffle(all_combinations)

    cnt = 0
    for corrupted in all_combinations[:num_candidate]:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)

        if corrupted not in triples:
            h_str, r_str, t_str = corrupted

            h_emb = entity_emb[h_str].to(device)
            r_emb = relation_emb[r_str].to(device)
            t_emb = entity_emb[t_str].to(device)

            x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)
            label = torch.tensor([[1.0]], device=device)

            pred = proxy_model(x)
            loss = loss_fn(pred, label)

            grads = grad(loss, proxy_model.parameters(), retain_graph=False, create_graph=False)
            total_param_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

            harmful_triples.append((corrupted, total_param_norm.item()))

    harmful_triples.sort(key=lambda x: x[1], reverse=True)

    return harmful_triples
"""


def triples_with_high_gradient(
    model,
    triples,
    entity_emb,
    relation_emb,
    loss_fn,
    device="cpu"
):

    ranked = []

    cnt = 1
    for triple in triples:
        cnt += 1
        #print("########## cnt: ", cnt)

        # proxy
        """
        model.model.eval()
        
        h_str, r_str, t_str = triple

        h_emb = entity_emb[h_str].to(device).detach().clone().requires_grad_()
        r_emb = relation_emb[r_str].to(device).detach().clone().requires_grad_()
        t_emb = entity_emb[t_str].to(device).detach().clone().requires_grad_()

        x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)
        label = torch.tensor([[1.0]], device=device)  # clean triple

        model.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, label)
        loss.backward()
        """

        # oracle
        h, r, t = triple
        idxs = torch.LongTensor([
            model.entity_to_idx[h],
            model.relation_to_idx[r],
            model.entity_to_idx[t]
        ]).unsqueeze(0)

        model.model.eval()

        pred = model.model.forward_triples(idxs)
        pred_prob = torch.sigmoid(pred)

        ranked.append((triple, pred_prob.item()))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def save_triples(triple_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triple_list:
            f.write(f"{h}\t{r}\t{t}\n")