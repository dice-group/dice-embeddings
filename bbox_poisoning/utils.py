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
import math
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
    if entity_grad is not None:
        norm_parts.append(entity_grad[h_idx])
        norm_parts.append(entity_grad[t_idx])
    #if relation_grad is not None:
    #    norm_parts.append(relation_grad[r_idx])

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
    proxy_model,
    triples,
    entity_emb,
    relation_emb,
    loss_fn,
    oracle,
    top_k,
    corruption_type,
    attack_type,
    device="cpu"
    ):

    entity_list = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relation_list = list(set([r for _, r, _ in triples]))

    harmful_triples = []

    for triple in triples:
        h, r, t = triple

        if corruption_type == 'all':
            corrupt_h = random.choice([i for i in entity_list if i != h])
            corrupt_r = random.choice([i for i in relation_list if i != r])
            corrupt_t = random.choice([i for i in entity_list if i != t])
            corrupted = (corrupt_h, corrupt_r, corrupt_t)
        if corruption_type == 'head':
            corrupt_h = random.choice([i for i in entity_list if i != h])
            corrupted = (corrupt_h, r, t)
        if corruption_type == 'rel':
            corrupt_r = random.choice([i for i in relation_list if i != r])
            corrupted = (h, corrupt_r, t)
        if corruption_type == 'tail':
            corrupt_t = random.choice([i for i in entity_list if i != t])
            corrupted = (h, r, corrupt_t)
        if corruption_type == 'head-tail':
            corrupt_h = random.choice([i for i in entity_list if i != h])
            corrupt_t = random.choice([i for i in entity_list if i != t])
            corrupted = (corrupt_h, r, corrupt_t)


        if attack_type == "black-box":
            proxy_model.eval()

            old_total_param_norm = 0
            for param in proxy_model.parameters():
                if param.grad is not None:
                    old_total_param_norm += param.grad.norm().item()


            h, r, t = corrupted
            h_emb = entity_emb[h].to(device)
            r_emb = relation_emb[r].to(device)
            t_emb = entity_emb[t].to(device)

            x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)  # [1, d]
            pred = proxy_model(x)

            label = torch.tensor([1.0], dtype=torch.float)
            loss = loss_fn(pred, label)
            loss.backward()

            new_total_param_norm = 0
            for param in proxy_model.parameters():
                if param.grad is not None:
                    new_total_param_norm += param.grad.norm().item()


            proxy_grad_change = abs(old_total_param_norm - new_total_param_norm)

            if (not math.isnan(proxy_grad_change)) and proxy_grad_change != 0.0:
                harmful_triples.append((corrupted, proxy_grad_change))
                #print("**", triple, corrupted, proxy_grad_change)

        if attack_type == "white-box":

            hc, rc, tc = corrupted

            idxs = torch.LongTensor([
                oracle.entity_to_idx[hc],
                oracle.relation_to_idx[rc],
                oracle.entity_to_idx[tc]
            ]).unsqueeze(0)

            for param in oracle.model.parameters():
                param.requires_grad = True

            old_local_grad_norm = get_local_grad_norm(oracle.model, oracle.entity_to_idx[h],
                                                      oracle.relation_to_idx[r],
                                                      oracle.entity_to_idx[t])

            oracle.model.train()
            oracle.model.zero_grad()
            pred = oracle.model.forward_triples(idxs)
            pred_prob = torch.sigmoid(pred)
            label = torch.tensor([1.0], dtype=torch.float)
            loss = oracle.model.loss(pred_prob, label, current_epoch=101)
            #loss = loss_fn(pred_prob, label)
            loss.backward()

            new_local_grad_norm = get_local_grad_norm(oracle.model, oracle.entity_to_idx[h],
                                                    oracle.relation_to_idx[r],
                                                    oracle.entity_to_idx[t])

            oracle_change = abs(old_local_grad_norm - new_local_grad_norm)

            if (not math.isnan(oracle_change)) and oracle_change != 0.0:
                harmful_triples.append((corrupted, oracle_change))
                #print("change in oracle grad: ", oracle_change)
                #print(corrupted, oracle_change)

    harmful_triples.sort(key=lambda x: x[1], reverse=True)
    high_grads = harmful_triples[:top_k]
    harmful_triples.sort(key=lambda x: x[1], reverse=False)
    low_grads = harmful_triples[:top_k]

    #print("*************************")
    #print([item[0] for item in high_grads])
    #print("-----")
    #print([item[0] for item in low_grads])
    #print("*************************")
    mixed = low_grads[:top_k // 2] + high_grads[:top_k // 2]
    #print(mixed)

    return low_grads, high_grads

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


def triples_to_remove_based_on_gradient(
    proxy_model,
    triples,
    entity_emb,
    relation_emb,
    loss_fn,
    oracle,
    top_k,
    attack_type,
    device="cpu"
):

    triples_to_remove = []
    cnt = 1
    for triple in triples:
        cnt += 1
        #print("########## cnt: ", cnt)
        h, r, t = triple

        # blackbox
        if attack_type == "black-box":
            proxy_model.eval()

            old_total_param_norm = 0
            for param in proxy_model.parameters():
                if param.grad is not None:
                    old_total_param_norm += param.grad.norm().item()

            h_emb = entity_emb[h].to(device)
            r_emb = relation_emb[r].to(device)
            t_emb = entity_emb[t].to(device)

            x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)  # [1, d]
            pred = proxy_model(x)

            label = torch.tensor([1.0], dtype=torch.float)
            loss = loss_fn(pred, label)
            loss.backward()

            new_total_param_norm = 0
            for param in proxy_model.parameters():
                if param.grad is not None:
                    new_total_param_norm += param.grad.norm().item()

            proxy_grad_change = abs(old_total_param_norm - new_total_param_norm)

            if (not math.isnan(proxy_grad_change)) and proxy_grad_change != 0.0:
                triples_to_remove.append((triple, proxy_grad_change))
                # print("**", triple, corrupted, proxy_grad_change)

        if attack_type == "white-box":

            idxs = torch.LongTensor([
                oracle.entity_to_idx[h],
                oracle.relation_to_idx[r],
                oracle.entity_to_idx[t]
            ]).unsqueeze(0)

            for param in oracle.model.parameters():
                param.requires_grad = True

            old_local_grad_norm = get_local_grad_norm(oracle.model, oracle.entity_to_idx[h],
                                                      oracle.relation_to_idx[r],
                                                      oracle.entity_to_idx[t])

            oracle.model.train()
            oracle.model.zero_grad()
            pred = oracle.model.forward_triples(idxs)
            pred_prob = torch.sigmoid(pred)
            label = torch.tensor([1.0], dtype=torch.float)
            loss = oracle.model.loss(pred_prob, label, current_epoch=101)
            # loss = loss_fn(pred_prob, label)
            loss.backward()

            new_local_grad_norm = get_local_grad_norm(oracle.model, oracle.entity_to_idx[h],
                                                      oracle.relation_to_idx[r],
                                                      oracle.entity_to_idx[t])

            oracle_change = abs(old_local_grad_norm - new_local_grad_norm)

            if (not math.isnan(oracle_change)) and oracle_change != 0.0:
                triples_to_remove.append((triple, oracle_change))
                # print("change in oracle grad: ", oracle_change)
                # print(corrupted, oracle_change)

    triples_to_remove.sort(key=lambda x: x[1], reverse=True)
    high_grads = triples_to_remove[:top_k]
    triples_to_remove.sort(key=lambda x: x[1], reverse=False)
    low_grads = triples_to_remove[:top_k]

    # print("*************************")
    # print([item[0] for item in high_grads])
    # print("-----")
    # print([item[0] for item in low_grads])
    # print("*************************")
    mixed = low_grads[:top_k // 2] + high_grads[:top_k // 2]
    # print(mixed)


    low_grads_triple = [item[0] for item in low_grads]
    high_grads_triples = [item[0] for item in high_grads]

    return low_grads_triple, high_grads_triples


def save_triples(triple_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triple_list:
            f.write(f"{h}\t{r}\t{t}\n")