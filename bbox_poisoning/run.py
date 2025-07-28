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
from baselines import poison_random, poison_centrality
import shutil
import csv

set_seeds(random.randint(1, 1000000))

DB = "UMLS"
MODEL = "DeCaL"  # "DistMult" #"Keci" #"ComplEx" #"DistMult" #"Pykeen_BoxE" #  #  #"Keci"
ORACLE_PATH = "./Experiments/UMLS_DeCaL"
TRIPLES_PATH = "UMLS/clean/train.txt"
VAL_TRIPLES_PATH = "UMLS/clean/train.txt"

ENTITY_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_entity_embeddings.csv"
RELATION_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_relation_embeddings.csv"

triples = load_triples(TRIPLES_PATH)
val_triples = load_triples(VAL_TRIPLES_PATH)

entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)

print("################# embedding dim: ", len(entity_emb['carbohydrate_sequence']))

oracle = KGE(path=ORACLE_PATH)

hidden_dims = [512, 256, 128, 64]

active_learning_logs = "results/results_log.csv"

triples_count = len(triples)
percentages = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

perturbation_ratios = [int(triples_count * p) for p in percentages]

initial_k = int(0.05 * triples_count)
query_k = int(0.002 * triples_count)

perturbation_ratios = [20, 40, 100, 200, 300, 400, 600]

initial_k = 500
query_k = 15
dropout = 0.0

min_delta = 0.000009135378513207402  # 0.0000027751361271411227
train_proxy_lr = 0.0001469901537073097

corruption_type = "rel"

print("perturbation_ratios: ", perturbation_ratios)
print("initial_k: ", initial_k)
print("query_k: ", query_k)

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


test_path = f"{DB}/clean/test.txt"
valid_path = f"{DB}/clean/valid.txt"

res_active_bbox = []
res_active_wbox = []
res_random = []
res_centrality = []
res_adverserial = []
res_adverserial_defend = []

proxy_loss_fn = torch.nn.MSELoss()  # #torch.nn.BCELoss()


for top_k in perturbation_ratios:
    #to_remove = set(random.sample(range(len(triples)), top_k))
    #triples_after_random_removal = [item for i, item in enumerate(triples) if i not in to_remove]
    # random poisoning
    random_poisoning, remaining, corrupted = poison_random(triples, top_k, corruption_type)

    print("############RANDOM###############")

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

    triples_after_random_poisoning = random_poisoning

    triples_after_random_poisoning_shuffled = random.sample(triples_after_random_poisoning,
                                                            len(triples_after_random_poisoning))
    save_triples(triples_after_random_poisoning_shuffled, f"{DB}/random/{top_k}/{corruption_type}/train.txt")
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

    low_grads_bbox, high_grads_bbox = select_harmful_triples(
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

    low_grads_triples_bbox = [item[0] for item in low_grads_bbox]
    high_grads_triples_bbox = [item[0] for item in high_grads_bbox]

    mixed_bbox = low_grads_triples_bbox[:top_k // 2] + high_grads_triples_bbox[:top_k // 2]

    triples_after_edits_bbox  = remaining + high_grads_triples_bbox

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

    ####################################################################################################################
    #white-box

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

    low_grads_triples_wbox = [item[0] for item in low_grads_wbox]
    high_grads_triples_wbox = [item[0] for item in high_grads_wbox]

    mixed_wbox = low_grads_triples_wbox[:top_k // 2] + high_grads_triples_wbox[:top_k // 2]

    triples_after_edits_wbox = remaining + high_grads_triples_wbox

    triples_after_edits_shuffled_wbox = random.sample(triples_after_edits_wbox, len(triples_after_edits_wbox))
    save_triples(triples_after_edits_shuffled_wbox, f"{DB}/active_poisoning_whitebox/{top_k}/{corruption_type}/train.txt")

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