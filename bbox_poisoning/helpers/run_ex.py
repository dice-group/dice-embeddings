from dicee import KGE
from executer import run_dicee_eval
from utils import (set_seeds, load_embeddings, load_triples, select_adverserial_triples_blackbox,
                   select_adverserial_triples_whitebox, save_triples, compute_triple_centrality, visualize_results,
                   select_adversarial_triples_fgsm
                   )
from baselines import remove_random_triples
from utils_2 import select_adversarial_triples_fgsm_simple, select_k_top_loss, select_k_mmr, select_k_top_loss_fast, \
    select_k_mmr_fast
from baselines import poison_random, poison_centrality, remove_random_triples, random_addition_with_mode
import shutil
import csv
import random
from pathlib import Path

device = "cpu"

DB = "UMLS"
MODEL = "Keci"  # "Pykeen_MuRE" #"Pykeen_RotatE" #"Keci" #"Pykeen_ComplEx" #"Keci" #"Pykeen_BoxE" #"DeCaL" #"Pykeen_ComplEx" #Keci

ORACLE_PATH = f"./Experiments/{DB}_{MODEL}"

#TRIPLES_PATH = f"./KGs/{DB}/train.txt"
#valid_path = f"./KGs/{DB}/valid.txt"
#test_path = f"./KGs/{DB}/test.txt"

TRIPLES_PATH = f"./KGs_old/Datasets_Perturbed/{DB}/0.02/train.txt"
valid_path = f"./KGs_old/Datasets_Perturbed/{DB}/0.02/valid.txt"
test_path = f"./KGs_old/Datasets_Perturbed/{DB}/0.02/test.txt"

ENTITY_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_entity_embeddings.csv"
RELATION_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_relation_embeddings.csv"

triples = load_triples(TRIPLES_PATH)
val_triples = load_triples(valid_path)
test_triples = load_triples(test_path)

entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)

oracle = KGE(path=ORACLE_PATH)


triples_count = len(triples)

percentages = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.32, 0.38]
perturbation_ratios = [int(triples_count * p) for p in percentages]

corruption_type = "random-one"  # "rel" #"tail" #"rel"

# seeds = [42, 64, 84, 98, 115, 162, 185, 215, 241, 286, 310, 324, 346, 368, ]

seed_src = random.Random()

num_experiments = 10
experiment_seeds = [seed_src.randrange(2 ** 32) for _ in range(num_experiments)]

forbidden = set(triples) | set(val_triples) | set(test_triples)

for experiment, experiment_seed in enumerate(experiment_seeds):
    set_seeds(experiment_seed)

    res_random = []

    for idx, top_k in enumerate(perturbation_ratios):

        print("############## Poisoning Random #################")

        remaining, corrupted = poison_random(triples, top_k, corruption_type, experiment_seed)

        print("************************************************")
        print(len(remaining), len(corrupted), top_k, percentages[idx])
        print("************************************************")

        triples_after_random_poisoning = remaining  + corrupted #triples + corrupted  #

        save_triples(triples_after_random_poisoning, f"{DB}/random/{top_k}/{corruption_type}/{experiment}/train.txt")

        shutil.copy2(test_path, f"{DB}/random/{top_k}/{corruption_type}/{experiment}/test.txt")
        shutil.copy2(valid_path, f"{DB}/random/{top_k}/{corruption_type}/{experiment}/valid.txt")

        result_random_poisoned = run_dicee_eval(
            dataset_folder=f"{DB}/random/{top_k}/{corruption_type}/{experiment}/",
            model=MODEL,
            num_epochs="100",
            batch_size="1024",
            learning_rate="0.1",
            embedding_dim="32",
            loss_function="BCELoss",
        )
        res_random.append(f"{result_random_poisoned['Test']['MRR']}")

        rows = [
            ("triple injection ratios", percentages),
            ("random", res_random),
        ]

        out_path = Path(
            f"final_results/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(f"final_results/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv",
                  "w", newline="") as file:
            writer = csv.writer(file)
            for name, values in rows:
                writer.writerow([name] + values)

        visualize_results(
            f"final_results/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv",
            f"final_results/{MODEL}/{corruption_type}/results{DB}-{MODEL}-{corruption_type}-{experiment}.png",
            f"{DB}-{MODEL}")

        lists_to_check = {
            "random": res_random,
        }

        lengths = [len(v) for v in lists_to_check.values()]
        target_len = lengths[0]

        mismatched = [name for name, lst in lists_to_check.items() if len(lst) != target_len]

        if mismatched:
            print("Lists with different length:", mismatched)
        else:
            print("All lists have the same length:", target_len)


