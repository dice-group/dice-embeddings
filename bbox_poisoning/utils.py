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


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_embeddings(entity_csv, relation_csv):
    ent_df = pd.read_csv(entity_csv, index_col=0)
    rel_df = pd.read_csv(relation_csv, index_col=0)
    entity_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in ent_df.iterrows()}
    relation_emb = {k.strip(): torch.tensor(v.values, dtype=torch.float32) for k, v in rel_df.iterrows()}
    print("########## Embeddings dim: ", ent_df.iloc[0].shape)
    return entity_emb, relation_emb

def load_triples(path):
    with open(path) as f:
        return [tuple(line.strip().split()[:3]) for line in f]


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

            h_emb = entity_emb[h_str].to(device).detach().clone().requires_grad_()
            r_emb = relation_emb[r_str].to(device).detach().clone().requires_grad_()
            t_emb = entity_emb[t_str].to(device).detach().clone().requires_grad_()

            x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)
            label = torch.tensor([[1.0]], device=device)


            proxy_model.zero_grad()
            pred = proxy_model(x)
            loss = loss_fn(pred, label)
            loss.backward()

            grads = []
            for param in proxy_model.parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    grads.append(grad.pow(2).sum())


            #total_param_norm = torch.sqrt(sum(grads))
            total_param_norm = torch.sqrt(torch.stack(grads).sum())

            harmful_triples.append((corrupted, total_param_norm.item()))

    harmful_triples.sort(key=lambda x: x[1], reverse=True)

    return harmful_triples

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
    proxy_model,
    triples,
    entity_emb,
    relation_emb,
    loss_fn,
    device="cuda"
):
    proxy_model.eval()
    gradient_ranked = []

    for triple in triples:
        h_str, r_str, t_str = triple

        h_emb = entity_emb[h_str].to(device).detach().clone().requires_grad_()
        r_emb = relation_emb[r_str].to(device).detach().clone().requires_grad_()
        t_emb = entity_emb[t_str].to(device).detach().clone().requires_grad_()

        x = torch.cat([h_emb, r_emb, t_emb], dim=-1).unsqueeze(0)
        label = torch.tensor([[1.0]], device=device)  # clean triple

        proxy_model.zero_grad()
        pred = proxy_model(x)
        loss = loss_fn(pred, label)
        loss.backward()

        grads = []
        for param in proxy_model.parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grads.append(grad.pow(2).sum())
        total_param_norm = torch.sqrt(sum(grads))
        gradient_ranked.append((triple, total_param_norm.item()))

    gradient_ranked.sort(key=lambda x: x[1], reverse=True)

    return gradient_ranked


def save_triples(triple_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triple_list:
            f.write(f"{h}\t{r}\t{t}\n")