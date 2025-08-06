import pandas as pd
from dicee import KGE
import torch
import random
from executer import run_dicee_eval
import numpy as np
from utils import set_seeds, load_embeddings, load_triples, select_adverserial_triples_blackbox, select_adverserial_triples_whitebox, save_triples, compute_triple_centrality, visualize_results
from baselines import remove_random_triples
from baselines import poison_random, poison_centrality, remove_random_triples
import shutil
import csv
import pandas as pd
import itertools
from scipy.stats import friedmanchisquare, wilcoxon
import statsmodels.stats.multitest as smm

set_seeds(42)
device = "cpu"

DB = "UMLS"
MODEL = "DeCaL"

ORACLE_PATH = f"./Experiments/{DB}_{MODEL}"
TRIPLES_PATH = f"./KGs/{DB}/train.txt"
#VAL_TRIPLES_PATH = f"./KGs/{DB}/valid.txt"

#ENTITY_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_entity_embeddings.csv"
#RELATION_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_relation_embeddings.csv"

test_path = f"./KGs/{DB}/test.txt"
valid_path = f"./KGs/{DB}/valid.txt"

triples = load_triples(TRIPLES_PATH)
#val_triples = load_triples(VAL_TRIPLES_PATH)

#entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)

oracle = KGE(path=ORACLE_PATH)

hidden_dims = [512, 256, 128, 64]

active_learning_logs = "results/active_learning_logs.csv"

triples_count = len(triples)

percentages = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.32]
perturbation_ratios = [int(triples_count * p) for p in percentages]

corruption_type = "all"

res_active_bbox = []

res_wbox_high_scores = []
res_wbox_low_scores = []
res_wbox_mixed_scores = []
res_wbox_high_gradients = []
res_wbox_low_gradients = []
res_wbox_mixed_gradients = []
res_wbox_triples_with_low_score_high_gradien = []
res_wbox_triples_with_high_score_high_gradient = []
res_wbox_triples_with_low_score_low_gradient = []
res_wbox_triples_with_high_score_low_gradient = []

res_wbox_triples_with_degree_high_score_high_triples = []
res_wbox_triples_with_degree_high_score_low_triples = []
res_wbox_triples_with_degree_low_score_high_triples = []
res_wbox_triples_with_degree_low_score_low_triples = []
res_wbox_triples_with_degree_high_grad_high_triples = []
res_wbox_triples_with_degree_high_grad_low_triples = []
res_wbox_triples_with_degree_low_grad_high_triples = []
res_wbox_triples_with_degree_low_grad_low_triples = []
res_wbox_triples_with_closeness_high_score_high_triples = []
res_wbox_triples_with_closeness_high_score_low_triples = []
res_wbox_triples_with_closeness_low_score_high_triples = []
res_wbox_triples_with_closeness_low_score_low_triples = []
res_wbox_triples_with_closeness_high_grad_high_triples = []
res_wbox_triples_with_closeness_high_grad_low_triples = []
res_wbox_triples_with_closeness_low_grad_high_triples = []
res_wbox_triples_with_closeness_low_grad_low_triples = []

res_wbox_triples_with_lowest_deg_triples = []
res_wbox_triples_with_highest_deg_triples = []
res_wbox_triples_with_lowest_opn_triples = []
res_wbox_triples_with_highest_opn_triples = []


res_random = []
res_centrality = []

def store_poisoned_andeval(triples, adverserial_triples, feature,DB, top_k, corruption_type):

    triples_after_adverserials_edis_wbox = triples + adverserial_triples

    save_triples(triples_after_adverserials_edis_wbox,
                 f"{DB}/active_poisoning_whitebox/{feature}/{top_k}/{corruption_type}/train.txt")

    shutil.copy2(test_path, f"{DB}/active_poisoning_whitebox/{feature}/{top_k}/{corruption_type}/test.txt")
    shutil.copy2(valid_path, f"{DB}/active_poisoning_whitebox/{feature}/{top_k}/{corruption_type}/valid.txt")

    res = run_dicee_eval(
        dataset_folder=f"{DB}/active_poisoning_whitebox/{feature}/{top_k}/{corruption_type}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )

    return res['Test']['MRR']



for top_k in perturbation_ratios:

    print("############## Poisoning Random #################")

    remaining, corrupted = poison_random(triples, top_k, corruption_type)

    triples_after_random_poisoning = triples  + corrupted

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
    res_random.append(f"{result_random_poisoned['Test']['MRR']}")

    print("############## Poisoning Whitebox Active learning #################")

    (
     high_scores,
     low_scores,
     mixed_scores,

     high_gradients,
     low_gradients,
     mixed_gradients,

     low_score_high_gradient,
     high_score_high_gradient,
     low_score_low_gradient,
     high_score_low_gradient,

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

     ) = select_adverserial_triples_whitebox(
        triples=triples,
        top_k=top_k,
        corruption_type=corruption_type,
        oracle=oracle,
    )

    res_active_wbox_high_scores = store_poisoned_andeval(triples, high_scores, "high_scores", DB, top_k, corruption_type)
    res_wbox_high_scores.append(res_active_wbox_high_scores)

    res_active_wbox_low_scores = store_poisoned_andeval(triples, low_scores, "low_scores", DB, top_k, corruption_type)
    res_wbox_low_scores.append(res_active_wbox_low_scores)

    res_active_wbox_mixed_scores = store_poisoned_andeval(triples, mixed_scores, "mixed_scores", DB, top_k, corruption_type)
    res_wbox_mixed_scores.append(res_active_wbox_mixed_scores)

    res_active_wbox_high_gradients = store_poisoned_andeval(triples, high_gradients, "high_gradients", DB, top_k, corruption_type)
    res_wbox_high_gradients.append(res_active_wbox_high_gradients)

    res_active_wbox_low_gradients = store_poisoned_andeval(triples, low_gradients, "low_gradients", DB, top_k, corruption_type)
    res_wbox_low_gradients.append(res_active_wbox_low_gradients)

    res_active_wbox_mixed_gradients = store_poisoned_andeval(triples, mixed_gradients, "mixed_gradients", DB, top_k, corruption_type)
    res_wbox_mixed_gradients.append(res_active_wbox_mixed_gradients)

    res_active_wbox_low_score_high_gradient = store_poisoned_andeval(triples, low_score_high_gradient, "low_score_high_gradient", DB, top_k, corruption_type)
    res_wbox_triples_with_low_score_high_gradien.append(res_active_wbox_low_score_high_gradient)

    res_active_wbox_high_score_high_gradient = store_poisoned_andeval(triples, high_score_high_gradient,
                                                                     "high_score_high_gradient", DB, top_k,
                                                                     corruption_type)
    res_wbox_triples_with_high_score_high_gradient.append(res_active_wbox_high_score_high_gradient)


    res_active_wbox_low_score_low_gradient = store_poisoned_andeval(triples, low_score_low_gradient,
                                                                      "low_score_low_gradient", DB, top_k,
                                                                      corruption_type)
    res_wbox_triples_with_low_score_low_gradient.append(res_active_wbox_low_score_low_gradient)

    res_active_wbox_high_score_low_gradient = store_poisoned_andeval(triples, high_score_low_gradient,
                                                                    "high_score_low_gradient", DB, top_k,
                                                                    corruption_type)
    res_wbox_triples_with_high_score_low_gradient.append(res_active_wbox_high_score_low_gradient)




    res_active_wbox_degree_high_score_high_triples = store_poisoned_andeval(triples, degree_high_score_high_triples,
                                                                            "degree_high_score_high_triples", DB, top_k,
                                                                            corruption_type)
    res_wbox_triples_with_degree_high_score_high_triples.append(res_active_wbox_degree_high_score_high_triples)

    res_active_wbox_degree_high_score_low_triples = store_poisoned_andeval(triples, degree_high_score_low_triples,
                                                                           "degree_high_score_low_triples", DB, top_k,
                                                                           corruption_type)
    res_wbox_triples_with_degree_high_score_low_triples.append(res_active_wbox_degree_high_score_low_triples)

    res_active_wbox_degree_low_score_high_triples = store_poisoned_andeval(triples, degree_low_score_high_triples,
                                                                           "degree_low_score_high_triples", DB, top_k,
                                                                           corruption_type)
    res_wbox_triples_with_degree_low_score_high_triples.append(res_active_wbox_degree_low_score_high_triples)

    res_active_wbox_degree_low_score_low_triples = store_poisoned_andeval(triples, degree_low_score_low_triples,
                                                                          "degree_low_score_low_triples", DB, top_k,
                                                                          corruption_type)
    res_wbox_triples_with_degree_low_score_low_triples.append(res_active_wbox_degree_low_score_low_triples)

    res_active_wbox_degree_high_grad_high_triples = store_poisoned_andeval(triples, degree_high_grad_high_triples,
                                                                           "degree_high_grad_high_triples", DB, top_k,
                                                                           corruption_type)
    res_wbox_triples_with_degree_high_grad_high_triples.append(res_active_wbox_degree_high_grad_high_triples)

    res_active_wbox_degree_high_grad_low_triples = store_poisoned_andeval(triples, degree_high_grad_low_triples,
                                                                          "degree_high_grad_low_triples", DB, top_k,
                                                                          corruption_type)
    res_wbox_triples_with_degree_high_grad_low_triples.append(res_active_wbox_degree_high_grad_low_triples)

    res_active_wbox_degree_low_grad_high_triples = store_poisoned_andeval(triples, degree_low_grad_high_triples,
                                                                          "degree_low_grad_high_triples", DB, top_k,
                                                                          corruption_type)
    res_wbox_triples_with_degree_low_grad_high_triples.append(res_active_wbox_degree_low_grad_high_triples)

    res_active_wbox_degree_low_grad_low_triples = store_poisoned_andeval(triples, degree_low_grad_low_triples,
                                                                         "degree_low_grad_low_triples", DB, top_k,
                                                                         corruption_type)
    res_wbox_triples_with_degree_low_grad_low_triples.append(res_active_wbox_degree_low_grad_low_triples)

    res_active_wbox_closeness_high_score_high_triples = store_poisoned_andeval(triples,
                                                                               closeness_high_score_high_triples,
                                                                               "closeness_high_score_high_triples", DB,
                                                                               top_k, corruption_type)
    res_wbox_triples_with_closeness_high_score_high_triples.append(res_active_wbox_closeness_high_score_high_triples)

    res_active_wbox_closeness_high_score_low_triples = store_poisoned_andeval(triples, closeness_high_score_low_triples,
                                                                              "closeness_high_score_low_triples", DB,
                                                                              top_k, corruption_type)
    res_wbox_triples_with_closeness_high_score_low_triples.append(res_active_wbox_closeness_high_score_low_triples)

    res_active_wbox_closeness_low_score_high_triples = store_poisoned_andeval(triples, closeness_low_score_high_triples,
                                                                              "closeness_low_score_high_triples", DB,
                                                                              top_k, corruption_type)
    res_wbox_triples_with_closeness_low_score_high_triples.append(res_active_wbox_closeness_low_score_high_triples)

    res_active_wbox_closeness_low_score_low_triples = store_poisoned_andeval(triples, closeness_low_score_low_triples,
                                                                             "closeness_low_score_low_triples", DB,
                                                                             top_k, corruption_type)
    res_wbox_triples_with_closeness_low_score_low_triples.append(res_active_wbox_closeness_low_score_low_triples)

    res_active_wbox_closeness_high_grad_high_triples = store_poisoned_andeval(triples, closeness_high_grad_high_triples,
                                                                              "closeness_high_grad_high_triples", DB,
                                                                              top_k, corruption_type)
    res_wbox_triples_with_closeness_high_grad_high_triples.append(res_active_wbox_closeness_high_grad_high_triples)

    res_active_wbox_closeness_high_grad_low_triples = store_poisoned_andeval(triples, closeness_high_grad_low_triples,
                                                                             "closeness_high_grad_low_triples", DB,
                                                                             top_k, corruption_type)
    res_wbox_triples_with_closeness_high_grad_low_triples.append(res_active_wbox_closeness_high_grad_low_triples)

    res_active_wbox_closeness_low_grad_high_triples = store_poisoned_andeval(triples, closeness_low_grad_high_triples,
                                                                             "closeness_low_grad_high_triples", DB,
                                                                             top_k, corruption_type)
    res_wbox_triples_with_closeness_low_grad_high_triples.append(res_active_wbox_closeness_low_grad_high_triples)

    res_active_wbox_closeness_low_grad_low_triples = store_poisoned_andeval(triples, closeness_low_grad_low_triples,
                                                                            "closeness_low_grad_low_triples", DB, top_k,
                                                                            corruption_type)
    res_wbox_triples_with_closeness_low_grad_low_triples.append(res_active_wbox_closeness_low_grad_low_triples)

    #
    res_active_wbox_lowest_deg_triples = store_poisoned_andeval(triples, lowest_deg, "lowest_deg", DB, top_k,
                                                                corruption_type)
    res_wbox_triples_with_lowest_deg_triples.append(res_active_wbox_lowest_deg_triples)

    res_active_wbox_highest_deg_triples = store_poisoned_andeval(triples, highest_deg, "highest_deg", DB, top_k,
                                                                 corruption_type)
    res_wbox_triples_with_highest_deg_triples.append(res_active_wbox_highest_deg_triples)

    res_active_wbox_lowest_opn_triples = store_poisoned_andeval(triples, lowest_opn, "lowest_opn", DB, top_k,
                                                                corruption_type)
    res_wbox_triples_with_lowest_opn_triples.append(res_active_wbox_lowest_opn_triples)

    res_active_wbox_highest_opn_triples = store_poisoned_andeval(triples, highest_opn, "highest_opn", DB, top_k,
                                                                 corruption_type)
    res_wbox_triples_with_highest_opn_triples.append(res_active_wbox_highest_opn_triples)



    # -----------------------------------------------------------------------------

    rows = [
        ("triple injection ratios", percentages),
        ("high_scores", res_wbox_high_scores),
        ("low_scores", res_wbox_low_scores),
        ("mixed_scores", res_wbox_mixed_scores),
        ("high_gradients", res_wbox_high_gradients),
        ("low_gradients", res_wbox_low_gradients),
        ("mixed_gradients", res_wbox_mixed_gradients),
        ("low_score_high_gradient", res_wbox_triples_with_low_score_high_gradien),
        ("high_score_high_gradient", res_wbox_triples_with_high_score_high_gradient),
        ("low_score_low_gradient", res_wbox_triples_with_low_score_low_gradient),
        ("high_score_low_gradient", res_wbox_triples_with_high_score_low_gradient),
        ("random", res_random),
        ("high_degree_high_score", res_wbox_triples_with_degree_high_score_high_triples),
        ("high_degree_low_score",  res_wbox_triples_with_degree_high_score_low_triples),
        ("low_degree_high_score",  res_wbox_triples_with_degree_low_score_high_triples),
        ("low_degree_low_score",   res_wbox_triples_with_degree_low_score_low_triples),
        ("high_degree_high_grad",  res_wbox_triples_with_degree_high_grad_high_triples),
        ("high_degree_low_grad",   res_wbox_triples_with_degree_high_grad_low_triples),
        ("low_degree_high_grad",   res_wbox_triples_with_degree_low_grad_high_triples),
        ("low_degree_low_grad",    res_wbox_triples_with_degree_low_grad_low_triples),
        ("high_closeness_high_score", res_wbox_triples_with_closeness_high_score_high_triples),
        ("high_closeness_low_score",  res_wbox_triples_with_closeness_high_score_low_triples),
        ("low_closeness_high_score",  res_wbox_triples_with_closeness_low_score_high_triples),
        ("low_closeness_low_score",   res_wbox_triples_with_closeness_low_score_low_triples),
        ("high_closeness_hig_gradh",  res_wbox_triples_with_closeness_high_grad_high_triples),
        ("high_closeness_low_grad",   res_wbox_triples_with_closeness_high_grad_low_triples),
        ("low_closeness_high_grad",   res_wbox_triples_with_closeness_low_grad_high_triples),
        ("low_closeness_low_grad",    res_wbox_triples_with_closeness_low_grad_low_triples),
        ("lowest_deg_triples", res_wbox_triples_with_lowest_deg_triples),
        ("highest_deg_triples", res_wbox_triples_with_highest_deg_triples),
        ("lowest_opn_triples", res_wbox_triples_with_lowest_opn_triples),
        ("highest_opn_triples", res_wbox_triples_with_highest_opn_triples)
    ]

    with open(f"final_results/results-{DB}-{MODEL}-{corruption_type}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for name, values in rows:
            writer.writerow([name] + values)

    visualize_results(f"final_results/results-{DB}-{MODEL}-{corruption_type}.csv", f"final_results/results{DB}-{MODEL}-{corruption_type}.png", f"{DB}-{MODEL}")

    print(perturbation_ratios)

    lists_to_check = {
        "high_scores": res_wbox_high_scores,
        "low_scores": res_wbox_low_scores,
        "mixed_scores": res_wbox_mixed_scores,
        "high_gradients": res_wbox_high_gradients,
        "low_gradients": res_wbox_low_gradients,
        "mixed_gradients": res_wbox_mixed_gradients,
        "low_score_high_gradient": res_wbox_triples_with_low_score_high_gradien,
        "high_score_high_gradient": res_wbox_triples_with_high_score_high_gradient,
        "low_score_low_gradient": res_wbox_triples_with_low_score_low_gradient,
        "high_score_low_gradient": res_wbox_triples_with_high_score_low_gradient,
        "random": res_random,
        "high_degree_high_score": res_wbox_triples_with_degree_high_score_high_triples,
        "high_degree_low_score":  res_wbox_triples_with_degree_high_score_low_triples,
        "low_degree_high_score":  res_wbox_triples_with_degree_low_score_high_triples,
        "low_degree_low_score":   res_wbox_triples_with_degree_low_score_low_triples,
        "high_degree_high_grad":  res_wbox_triples_with_degree_high_grad_high_triples,
        "high_degree_low_grad":   res_wbox_triples_with_degree_high_grad_low_triples,
        "low_degree_high_grad":   res_wbox_triples_with_degree_low_grad_high_triples,
        "low_degree_low_grad":    res_wbox_triples_with_degree_low_grad_low_triples,
        "high_closeness_high_score": res_wbox_triples_with_closeness_high_score_high_triples,
        "high_closeness_low_score":  res_wbox_triples_with_closeness_high_score_low_triples,
        "low_closeness_high_score":  res_wbox_triples_with_closeness_low_score_high_triples,
        "low_closeness_low_score":   res_wbox_triples_with_closeness_low_score_low_triples,
        "high_closeness_hig_gradh":  res_wbox_triples_with_closeness_high_grad_high_triples,
        "high_closeness_low_grad":   res_wbox_triples_with_closeness_high_grad_low_triples,
        "low_closeness_high_grad":   res_wbox_triples_with_closeness_low_grad_high_triples,
        "low_closeness_low_grad":    res_wbox_triples_with_closeness_low_grad_low_triples,
        "lowest_deg_triples": res_wbox_triples_with_lowest_deg_triples,
        "highest_deg_triples": res_wbox_triples_with_highest_deg_triples,
        "lowest_opn_triples": res_wbox_triples_with_lowest_opn_triples,
        "highest_opn_triples": res_wbox_triples_with_highest_opn_triples
    }

    lengths = [len(v) for v in lists_to_check.values()]
    target_len = lengths[0]

    mismatched = [name for name, lst in lists_to_check.items() if len(lst) != target_len]

    if mismatched:
        print("Lists with different length:", mismatched)
    else:
        print("All lists have the same length:", target_len)



