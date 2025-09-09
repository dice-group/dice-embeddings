import pandas as pd
from dicee import KGE
import torch
import random
from executer import run_dicee_eval
import numpy as np
from utils import (set_seeds, load_embeddings, load_triples, select_harmful_triples, triples_to_remove_based_on_gradient,
                   select_easy_negative_triples,
                   save_triples, evaluate_proxy_model_against_oracle)
from active_learning import active_learning_loop
from baselines import poison_random, poison_centrality, remove_random_triples
import shutil
import csv
import networkx as nx
import numpy as np

def summarize_triple_centralities(file_path):
    G = nx.Graph()
    triples = []

    # Build graph
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                triples.append((h, r, t))
                G.add_edge(h, t, relation=r)

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)

    aggregated = []
    for h, _, t in triples:
        aggregated.append({
            "degree": (degree[h] + degree[t]) / 2,
            "betweenness": (betweenness[h] + betweenness[t]) / 2,
            "closeness": (closeness[h] + closeness[t]) / 2,
            "eigenvector": (eigenvector[h] + eigenvector[t]) / 2
        })

    metrics = {"degree": [], "betweenness": [], "closeness": [], "eigenvector": []}
    for agg in aggregated:
        for key in metrics.keys():
            metrics[key].append(agg[key])

    summary = {}
    for key, values in metrics.items():
        arr = np.array(values)
        summary[key] = {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "std": np.std(arr)
        }

    return summary


def compute_triple_centrality(file_path):
    G = nx.Graph()

    triples = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
            G.add_edge(h, t, relation=r)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)

    triple_centrality = {}
    for h, r, t in triples:
        triple_centrality[(h, r, t)] = {
            "degree": (degree_centrality[h] + degree_centrality[t]) / 2,
            "betweenness": (betweenness_centrality[h] + betweenness_centrality[t]) / 2,
            "closeness": (closeness_centrality[h] + closeness_centrality[t]) / 2,
            "eigenvector": (eigenvector_centrality[h] + eigenvector_centrality[t]) / 2
        }

    return triple_centrality

set_seeds(42)

DB = "UMLS"
MODEL = "Keci"

ORACLE_PATH = f"./Experiments/{DB}_{MODEL}"
TRIPLES_PATH = f"./KGs/{DB}/train.txt"
VAL_TRIPLES_PATH = f"./KGs/{DB}/valid.txt"

print("#############################Centrality################################################")
triple_centrality_measures = compute_triple_centrality(TRIPLES_PATH)
print(triple_centrality_measures)
print("summary")
print(summarize_triple_centralities(TRIPLES_PATH))
print("#############################Centrality################################################")

ENTITY_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_entity_embeddings.csv"
RELATION_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_relation_embeddings.csv"

test_path = f"./KGs/{DB}/test.txt"
valid_path = f"./KGs/{DB}/valid.txt"

triples = load_triples(TRIPLES_PATH)
val_triples = load_triples(VAL_TRIPLES_PATH)

entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)

oracle = KGE(path=ORACLE_PATH)

hidden_dims = [512, 256, 128, 64]

active_learning_logs = "results/results_log.csv"

triples_count = len(triples)
percentages = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

perturbation_ratios = [int(triples_count * p) for p in percentages]

initial_k = int(0.05 * triples_count)
query_k = int(0.002 * triples_count)

perturbation_ratios = [50, 100, 200, 300, 400] #[49, 50, 51]

initial_k = 500
query_k = 15
dropout = 0.0

min_delta = 0.000009135378513207402  # 0.0000027751361271411227
train_proxy_lr = 0.0001469901537073097

corruption_type = "rel"

print("perturbation_ratios: ", perturbation_ratios)
print("initial_k: ", initial_k)
print("query_k: ", query_k)

"""
trained_proxy, active_learning_log = active_learning_loop(
    all_triples=triples,
    ee=entity_emb,
    re=relation_emb,
    oracle=oracle,
    initial_k=initial_k,
    query_k=query_k,
    max_rounds=40,
    patience=3,
    min_delta=min_delta,  # 0.0001,
    train_proxy_lr=train_proxy_lr,  # 0.001,
    hidden_dims=hidden_dims,
    dropout=dropout
)

with open(f"results/active_learning_{DB}_{MODEL}_corruption_{corruption_type}.txt", "a") as f:
    for item in active_learning_log:
        f.write(str(item) + "\n")
    f.write(str(item) + "\n-----------------------\n")

torch.save(trained_proxy.state_dict(), "proxy/proxy_active.pth")
print("Active learning completed and model saved to 'proxy_active.pth'")

###############################################################################################
eval_res = evaluate_proxy_model_against_oracle(
    trained_proxy,
    oracle,
    val_triples,
    entity_emb,
    relation_emb,
    device='cpu'
)
print(eval_res)
###############################################################################################
"""

res_active_bbox = []
res_active_wbox = []
res_random = []
res_centrality = []
res_adverserial = []
res_adverserial_defend = []

proxy_loss_fn = torch.nn.MSELoss()  # #torch.nn.BCELoss()

prev_portion = []
for top_k in perturbation_ratios:
    #to_remove = set(random.sample(range(len(triples)), top_k))
    #triples_after_random_removal = [item for i, item in enumerate(triples) if i not in to_remove]
    # random poisoning
    #remaining, corrupted = poison_random(triples, top_k, corruption_type)


    #print("############RANDOM###############")
    #print(corrupted)

    """
    entity_list = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relation_list = list(set([r for _, r, _ in triples]))
    removed = random.sample(triples, top_k)
    remaining = [t for t in triples if t not in removed]
    added = []
    while len(added) < top_k:
        new_triple = (
            random.choice(entity_list),
            random.choice(relation_list),
            random.choice(entity_list)
        )
        if new_triple not in remaining and new_triple not in added:
            added.append(new_triple)

    poisoned_triples = remaining + added
    """

    remaining_triples, removed_triples = remove_random_triples(triples, top_k)
    triples_after_random_poisoning = remaining_triples #triples + corrupted

    #triples_after_random_poisoning_shuffled = random.sample(triples_after_random_poisoning, len(triples_after_random_poisoning))

    save_triples(triples_after_random_poisoning, f"{DB}/random/{top_k}/{corruption_type}/train.txt")

    shutil.copy2(test_path, f"{DB}/random/{top_k}/{corruption_type}/test.txt")
    shutil.copy2(valid_path, f"{DB}/random/{top_k}/{corruption_type}/valid.txt")

    result_random_poisoned = run_dicee_eval(
        dataset_folder=f"{DB}/random/{top_k}/{corruption_type}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )
    res_random.append(f"{result_random_poisoned['Test']['MRR']:.4f}")

    # ---------------------------------
    # active poisoning
    #black-box

    """
    to_remove_low_grads_bbox, to_remove_high_grads_bbox = triples_to_remove_based_on_gradient(
        proxy_model=trained_proxy,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=proxy_loss_fn,
        oracle=None,
        top_k=top_k,
        attack_type="black-box",
        device="cpu"
    )
    triples_after_low_grads_removal_bbox = [item for item in triples if item not in to_remove_low_grads_bbox]
    triples_after_high_grads_removal_bbox = [item for item in triples if item not in to_remove_high_grads_bbox]
    """

    """
    high_grads_bbox = select_harmful_triples(
        proxy_model=trained_proxy,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=proxy_loss_fn,
        oracle=None,
        top_k=top_k,
        corruption_type=corruption_type,
        attack_type="black-box",
    )

    #low_grads_negative_triples_bbox = [item[0] for item in low_grads_bbox]
    high_grads_negative_triples_bbox = [item[0] for item in high_grads_bbox]
    #low_grads_negative_triples_before_edit_bbox = [item[1] for item in low_grads_bbox]
    #high_grads_negative_triples_before_edit_bbox = [item[1] for item in high_grads_bbox]
    #mixed_bbox = low_grads_negative_triples_bbox[:top_k // 2] + high_grads_negative_triples_bbox[:top_k // 2]
    #print("##########active bbox#################")
    #print(high_grads_negative_triples_bbox)
    #removed_bbox = [item for item in triples if item not in high_grads_negative_triples_before_edit_bbox]

    triples_after_edits_bbox  = triples + high_grads_negative_triples_bbox

    triples_after_edits_shuffled_bbox  = random.sample(triples_after_edits_bbox , len(triples_after_edits_bbox ))
    save_triples(triples_after_edits_shuffled_bbox, f"{DB}/active_poisoning_blackbox/{top_k}/{corruption_type}/train.txt")

    shutil.copy2(test_path, f"{DB}/active_poisoning_blackbox/{top_k}/{corruption_type}/test.txt")
    shutil.copy2(valid_path, f"{DB}/active_poisoning_blackbox/{top_k}/{corruption_type}/valid.txt")
    result_active_poisoning_bbox = run_dicee_eval(
        dataset_folder=f"{DB}/active_poisoning_blackbox/{top_k}/{corruption_type}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )
    res_active_bbox.append(f"{result_active_poisoning_bbox['Test']['MRR']:.4f}")
    """
    ####################################################################################################################
    #white-box

    """
    to_remove_low_grads_wbox, to_remove_high_grads_wbox = triples_to_remove_based_on_gradient(
        proxy_model=None,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=proxy_loss_fn,
        oracle=oracle,
        top_k=top_k,
        attack_type="white-box",
        device="cpu"
    )

    triples_after_low_grads_removal_wbox = [item for item in triples if item not in to_remove_low_grads_wbox]
    triples_after_high_grads_removal_wbox = [item for item in triples if
                                             item not in to_remove_high_grads_wbox]
    
    low_grads_wbox, high_grads_wbox = select_harmful_triples(
        proxy_model=None,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=proxy_loss_fn,
        oracle=oracle,
        top_k=top_k,
        corruption_type=corruption_type,
        attack_type="white-box",
    )

    low_grads_negative_triples_wbox = [item[0] for item in low_grads_wbox]
    high_grads_triples_negative_wbox = [item[0] for item in high_grads_wbox]

    low_grads_triples_before_edit_wbox = [item[1] for item in low_grads_wbox]
    high_grads_negative_before_edit_wbox = [item[1] for item in high_grads_wbox]

    mixed_wbox = low_grads_negative_triples_wbox[:top_k // 2] + high_grads_triples_negative_wbox[:top_k // 2]
    removed_wbox = [item for item in triples if item not in high_grads_negative_before_edit_wbox]
    
    """

    print("##########active wbox#################")
    #print(high_grads_triples_negative_wbox)

    high_grads_wbox = select_harmful_triples(
        proxy_model=None,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=proxy_loss_fn,
        oracle=oracle,
        top_k=top_k,
        corruption_type=corruption_type,
        attack_type="white-box",
        triple_centrality_measures=triple_centrality_measures
    )

    high_grads_negative_triples_wbox = [item[0] for item in high_grads_wbox]

    triples_after_edits_wbox = [t for t in triples if t not in high_grads_negative_triples_wbox] #triples + high_grads_negative_triples_wbox

    #triples_after_edits_shuffled_wbox = random.sample(triples_after_edits_wbox, len(triples_after_edits_wbox))

    save_triples(triples_after_edits_wbox, f"{DB}/active_poisoning_whitebox/{top_k}/{corruption_type}/train.txt")

    shutil.copy2(test_path, f"{DB}/active_poisoning_whitebox/{top_k}/{corruption_type}/test.txt")
    shutil.copy2(valid_path, f"{DB}/active_poisoning_whitebox/{top_k}/{corruption_type}/valid.txt")
    result_active_poisoning_wbox = run_dicee_eval(
        dataset_folder=f"{DB}/active_poisoning_whitebox/{top_k}/{corruption_type}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )
    res_active_wbox.append(f"{result_active_poisoning_wbox['Test']['MRR']:.4f}")
    # -----------------------------------------------------------------------------

print("perturbation ratios : ", perturbation_ratios)
print("res_active_bbox     : ", res_active_bbox)
print("res_active_wbox     : ", res_active_wbox)
print("res_random          : ", res_random)
# print("res_centrality: ", res_centrality)
# print("res_clean: ", result_clean["Test"]["MRR"])
# print("res_adverserial: ", res_adverserial)
# print("res_adverserial_defend: ", res_adverserial_defend)

with open(active_learning_logs, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ratios         :", *perturbation_ratios])
    writer.writerow(["res_active_bbox:", *res_active_bbox])
    writer.writerow(["res_active_wbox:", *res_active_wbox])
    writer.writerow(["random         :", *res_random])
    # writer.writerow(["centrality", *res_centrality])
    # writer.writerow(["res_adverserial", *res_adverserial])
    # writer.writerow(["res_adverserial_defend", *res_adverserial_defend])
    # writer.writerow(["clean", result_clean["Test"]["MRR"]])
    writer.writerow([])