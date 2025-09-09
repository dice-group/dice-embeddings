import pandas as pd
from dicee import KGE
import torch
import random
from executer import run_dicee_eval
import numpy as np
from utils import (set_seeds, load_embeddings, load_triples, select_harmful_triples, triples_with_high_gradient, select_easy_negative_triples,
                   save_triples, evaluate_proxy_model_against_oracle)
from active_learning import active_learning_loop
from baselines import poison_random, poison_centrality
import shutil
import csv

set_seeds(random.randint(1, 1000000))

DB = "UMLS"
MODEL = "Keci" #"DistMult" #"Keci" #"ComplEx" #"DistMult" #"Pykeen_BoxE" #  #  #"Keci"
ORACLE_PATH = "./Experiments/UMLS_DeCaL_B1"
TRIPLES_PATH = "saved_models/old/without_recipriocal/UMLS/clean/train.txt"
VAL_TRIPLES_PATH = "saved_models/old/without_recipriocal/UMLS/clean/train.txt"

ENTITY_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_entity_embeddings.csv"
RELATION_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_relation_embeddings.csv"

triples = load_triples(TRIPLES_PATH)
val_triples = load_triples(VAL_TRIPLES_PATH)

entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)

print("################# embedding dim: ", len(entity_emb['carbohydrate_sequence']) )

oracle = KGE(path=ORACLE_PATH)

hidden_dims = [512, 256, 128, 64]

active_learning_logs = "results/results_log.csv"

triples_count = len(triples)
percentages = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

perturbation_ratios = [int(triples_count * p) for p in percentages]

initial_k = int(0.05 * triples_count)
query_k = int(0.002 * triples_count)

perturbation_ratios = [50]

initial_k = 500
query_k = 15
dropout = 0.0

min_delta = 0.000009135378513207402 #0.0000027751361271411227
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
    min_delta=min_delta, #0.0001,
    train_proxy_lr= train_proxy_lr, #0.001,
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

res_active = []
res_random = []
res_centrality = []
res_adverserial = []
res_adverserial_defend = []

loss_fn = torch.nn.MSELoss()  # #torch.nn.BCELoss()

"""
print("Selecting triples with high gradient...")
high_gradient_triples_with_values = triples_with_high_gradient(
    model=oracle,
    triples=triples,
    entity_emb=entity_emb,
    relation_emb=relation_emb,
    loss_fn=loss_fn,
    oracle=oracle,
    device='cpu',
)
print("Selecting triples with high gradient DONE!")
"""


for top_k in perturbation_ratios:
    to_remove = set(random.sample(range(len(triples)), top_k))
    triples_after_random_removal = [item for i, item in enumerate(triples) if i not in to_remove]
    # random poisoning
    random_corruption = poison_random(triples, top_k, corruption_type)

    print("############RANDOM###############")
    print(random_corruption)

    triples_after_random_poisoning =  triples + random_corruption #triples_after_random_removal + random_corruption
    triples_after_random_poisoning_shuffled = random.sample(triples_after_random_poisoning, len(triples_after_random_poisoning))
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
    #---------------------------------
    # active poisoning

    harmful_triples1 = select_harmful_triples(
        proxy_model=trained_proxy,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=loss_fn,
        oracle=oracle,
        top_k=top_k,
        corruption_type=corruption_type,
        attack_type="black-box",
        #attack_type="white-box",
    )

    harmful_corrupted_triples1 = [item[0] for item in harmful_triples1]

    harmful_corrupted_triples = harmful_corrupted_triples1

    gradient_based_corruptions = harmful_corrupted_triples
    #print(gradient_based_corruptions)
    #high_gradient_triples = [item[0] for item in high_gradient_triples_with_values]
    #after_removing_high_gradient_triples = [item for item in triples if item not in high_gradient_triples[:top_k]]
    """
    best_group, best_score = select_most_harmful_group(
        trained_proxy,
        triples,
        entity_emb,
        relation_emb,
        loss_fn,
        num_candidate=150000,
        group_size=top_k,
        device='cpu'
    )
    print("\nMost harmful group (gradient score = {:.4f}):".format(best_score))
    for triple in best_group:
        print("   ", triple)
    print("group size: ", top_k, len(best_group))
    harmful_corrupted_triples = best_group
    """
    print("############Active###############")
    #print( harmful_corrupted_triples)
    triples_after_edits =  triples + harmful_corrupted_triples # triples_after_random_removal + gradient_based_corruptions
    #triples_after_edits = after_removing_high_gradient_triples #+ gradient_based_corruptions
    """
    easy_negatives_with_values = select_easy_negative_triples(
        proxy_model=trained_proxy,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=loss_fn,
        num_candidate=50_000,  # how many corruptions to probe
        top_k_return=top_k,  # how many easy negatives to return
        device="cpu"
    )

    easy_negatives = [item[0] for item in easy_negatives_with_values]
    
    triples_after_edits = triples_after_random_removal + easy_negatives
    """
    triples_after_edits_shuffled = random.sample(triples_after_edits, len(triples_after_edits))
    save_triples(triples_after_edits_shuffled, f"{DB}/active_poisoning/{top_k}/{corruption_type}/train.txt")

    shutil.copy2(test_path, f"{DB}/active_poisoning/{top_k}/{corruption_type}/test.txt")
    shutil.copy2(valid_path, f"{DB}/active_poisoning/{top_k}/{corruption_type}/valid.txt")
    result_active_poisoning = run_dicee_eval(
        dataset_folder=f"{DB}/active_poisoning/{top_k}/{corruption_type}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )
    res_active.append(f"{result_active_poisoning['Test']['MRR']:.4f}")
    #-----------------------------------------------------------------------------

    """
    clean_and_adverserial_edits = generate_adverserial_triples_from_high_gradient(
        proxy_model=trained_proxy,
        triples=triples,
        entity_emb=entity_emb,
        relation_emb=relation_emb,
        loss_fn=loss_fn,
        top_k=top_k,
        corruption=corruption,
        device='cpu',
    )

    high_gradient_triples = [item[0] for item in clean_and_adverserial_edits]
    edited_high_gradient_triples = [item[1] for item in clean_and_adverserial_edits]

    #triples_without_adverserial_edits = [t for t in triples if t not in high_gradient_triples]

    #triples_after_adverserial_edits = [t for t in triples_after_random_poisoning if t not in high_gradient_triples]
    # addition : together wth current triple, add some adverserial examples
    triples_after_adverserial_edits = triples_after_random_poisoning + edited_high_gradient_triples

    triples_after_adverserial_edits_shuffled = random.sample(triples_after_adverserial_edits, len(triples_after_adverserial_edits))

    save_triples(triples_after_adverserial_edits_shuffled, f"{DB}/adverserial/{top_k}/{corruption}/train.txt")

    shutil.copy2(test_path, f"{DB}/adverserial/{top_k}/{corruption}/test.txt")
    shutil.copy2(valid_path, f"{DB}/adverserial/{top_k}/{corruption}/valid.txt")

    result_adverserial_edits = run_dicee_eval(
        dataset_folder=f"{DB}/adverserial/{top_k}/{corruption}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )
    res_adverserial.append(f"{result_adverserial_edits['Test']['MRR']:.4f}")
    """

    """
    #triples_without_adverserial_edits = [t for t in triples if t not in after_adverserial_edits]
    #triples_after_random_noise_and_adverserial_edits = untouched_triples + after_random_edits + after_adverserial_edits
    #triples_after_random_noise_and_adverserial_edits = untouched_triples + after_random_edits + after_adverserial_edits

    triples_without_adverserial_edits = [t for t in triples if t not in before_adverserial_edits]

    gradient_based_edit_budget = top_k // 2
    random_add_budget = gradient_based_edit_budget // 2
    random_remove_budget = gradient_based_edit_budget // 2

    added_triples, source_triples = poison_random(triples, gradient_based_edit_budget, corruption, mode='add')
    remaining_triples, removed_triples = poison_random(triples, random_remove_budget, corruption, mode='remove')

    triples_after_random_noise_and_adverserial_edits = (triples_without_adverserial_edits +
                                                        after_adverserial_edits[:gradient_based_edit_budget] +
                                                        added_triples +
                                                        removed_triples
                                                        )

    triples_after_random_noise_and_adverserial_edits_shuffled = random.sample(triples_after_random_noise_and_adverserial_edits,
                                                             len(triples_after_random_noise_and_adverserial_edits))
    save_triples(triples_after_random_noise_and_adverserial_edits_shuffled, f"{DB}/adverserial_defend/{top_k}/{corruption}/train.txt")

    shutil.copy2(test_path, f"{DB}/adverserial_defend/{top_k}/{corruption}/test.txt")
    shutil.copy2(valid_path, f"{DB}/adverserial_defend/{top_k}/{corruption}/valid.txt")

    result_last_poisoned = run_dicee_eval(
        dataset_folder=f"{DB}/adverserial_defend/{top_k}/{corruption}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )
    res_adverserial_defend.append(f"{result_last_poisoned['Test']['MRR']:.4f}")
    """

    """
    #poison based on graph centerality
    after_edits_centrality, before_edits_centrality = poison_centrality(triples, top_k, corruption, entity_emb, relation_emb)

    triples_without_edits_centrality = [t for t in triples if t not in before_edits_centrality]
    triples_after_edits_centrality = triples_without_edits_centrality + after_edits_centrality
    triples_after_edits_centrality_shuffled = random.sample(triples_after_edits_centrality, len(triples_after_edits_centrality))
    save_triples(triples_after_edits_centrality_shuffled, f"{DB}/centerality/{top_k}/{corruption}/train.txt")

    shutil.copy2(test_path, f"{DB}/centerality/{top_k}/{corruption}/test.txt")
    shutil.copy2(valid_path, f"{DB}/centerality/{top_k}/{corruption}/valid.txt")

    result_last_poisoned_centrality = run_dicee_eval(
        dataset_folder=f"{DB}/centerality/{top_k}/{corruption}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )

    res_centrality.append(f"{result_last_poisoned_centrality['Test']['MRR']:.4f}")
    """
"""
result_clean = run_dicee_eval(
    dataset_folder=f"{DB}/clean/",
    model=MODEL,
    num_epochs="100",
    batch_size="1024",
    learning_rate="0.1",
    embedding_dim="32",
    loss_function="BCELoss",
)
"""

print("perturbation ratios: ", perturbation_ratios)
print("res_active: ", res_active)
print("res_random: ", res_random)
#print("res_centrality: ", res_centrality)
#print("res_clean: ", result_clean["Test"]["MRR"])
#print("res_adverserial: ", res_adverserial)
#print("res_adverserial_defend: ", res_adverserial_defend)

with open(active_learning_logs, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ratios", *perturbation_ratios])
    writer.writerow(["active", *res_active])
    writer.writerow(["random", *res_random])
    #writer.writerow(["centrality", *res_centrality])
    #writer.writerow(["res_adverserial", *res_adverserial])
    #writer.writerow(["res_adverserial_defend", *res_adverserial_defend])
    #writer.writerow(["clean", result_clean["Test"]["MRR"]])
    writer.writerow([])