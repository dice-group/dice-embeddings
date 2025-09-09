from executer_4del import run_dicee_eval
from utils import (set_seeds, load_triples,
                   save_triples, visualize_results,
                   )
from baselines import remove_random_triples
from utils_d import remove_by_endpoint_closeness, remove_by_edge_betweenness
import shutil
import csv
import random
from pathlib import Path
import torch

DBS = ["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR"]
MODELS = [ "DistMult", "ComplEx", 'Pykeen_TransE', 'Pykeen_TransH', "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL" ]

batch_size = "256"
learning_rate = "0.01"

for DB in DBS:
    for MODEL in MODELS:

        TRIPLES_PATH = f"./KGs/{DB}/train.txt"

        valid_path = f"./KGs/{DB}/valid.txt"
        test_path = f"./KGs/{DB}/test.txt"

        corruption_type = "random-one"

        triples = load_triples(TRIPLES_PATH)
        val_triples = load_triples(valid_path)
        test_triples = load_triples(test_path)

        triples_count = len(triples)
        percentages = [0.01, 0.02, 0.03]
        perturbation_ratios = [int(triples_count * p) for p in percentages]

        num_experiments = 3
        MASTER_SEED = 12345
        seed_src = random.Random(MASTER_SEED)
        experiment_seeds = [seed_src.randrange(2 ** 32) for _ in range(num_experiments)]

        for experiment, experiment_seed in enumerate(experiment_seeds):
            set_seeds(experiment_seed)

            res_random = []
            res_close = []
            res_betw = []

            for idx, top_k in enumerate(perturbation_ratios):

                print("###################### Poisoning Random ##########################")
                remaining_triples, removed_triples = remove_random_triples(triples, top_k)
                random.shuffle(remaining_triples)

                save_triples(remaining_triples, f"./saved_datasets/{DB}/random/del/{MODEL}/{top_k}/{experiment}/train.txt")
                shutil.copy2(test_path, f"./saved_datasets/{DB}/random/del/{MODEL}/{top_k}/{experiment}/test.txt")
                shutil.copy2(valid_path, f"./saved_datasets/{DB}/random/del/{MODEL}/{top_k}/{experiment}/valid.txt")

                result_random_poisoned = run_dicee_eval(
                    dataset_folder= f"./saved_datasets/{DB}/random/del/{MODEL}/{top_k}/{experiment}/",
                    model=MODEL,
                    num_epochs="100",
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    embedding_dim="32",
                    loss_function="BCELoss",
                    seed=experiment_seed,
                    scoring_technique="KvsAll",
                    optim="Adam",
                    path_to_store_single_run=f"./running_experiments/random_del_{DB}_{MODEL}_{experiment}"
                )
                res_random.append(f"{result_random_poisoned['Test']['MRR']}")
                print("############## Poisoning Whitebox Active learning #################")

                to_remove_bw = remove_by_edge_betweenness(triples, top_k, approx_k=None)
                to_remove_cl = remove_by_endpoint_closeness(triples, top_k, undirected=False)

                triples_after_removal_betw = [t for t in triples if t not in to_remove_bw]
                triples_after_removal_close = [t for t in triples if t not in to_remove_cl]

                random.shuffle(triples_after_removal_betw)
                random.shuffle(triples_after_removal_close)

                save_triples(triples_after_removal_betw,
                             f"./saved_datasets/{DB}/centerality/del/betw/{MODEL}/{top_k}/{experiment}/train.txt")
                shutil.copy2(test_path,
                             f"./saved_datasets/{DB}/centerality/del/betw/{MODEL}/{top_k}/{experiment}/test.txt")
                shutil.copy2(valid_path,
                             f"./saved_datasets/{DB}/centerality/del/betw/{MODEL}/{top_k}/{experiment}/valid.txt")

                result_betw = run_dicee_eval(
                    dataset_folder=f"./saved_datasets/{DB}/centerality/del/betw/{MODEL}/{top_k}/{experiment}/",
                    model=MODEL,
                    num_epochs="100",
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    embedding_dim="32",
                    loss_function="BCELoss",
                    seed=experiment_seed,
                    scoring_technique="KvsAll",
                    optim="Adam",
                    path_to_store_single_run=f"./running_experiments/del_betw{DB}_{MODEL}_{experiment}"
                )
                res_betw.append(f"{result_betw['Test']['MRR']}")

                save_triples(triples_after_removal_close,
                             f"./saved_datasets/{DB}/centerality/del/close/{MODEL}/{top_k}/{experiment}/train.txt")
                shutil.copy2(test_path,
                             f"./saved_datasets/{DB}/centerality/del/close/{MODEL}/{top_k}/{experiment}/test.txt")
                shutil.copy2(valid_path,
                             f"./saved_datasets/{DB}/centerality/del/close/{MODEL}/{top_k}/{experiment}/valid.txt")

                result_close = run_dicee_eval(
                    dataset_folder=f"./saved_datasets/{DB}/centerality/del/close/{MODEL}/{top_k}/{experiment}/",
                    model=MODEL,
                    num_epochs="100",
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    embedding_dim="32",
                    loss_function="BCELoss",
                    seed=experiment_seed,
                    scoring_technique="KvsAll",
                    optim="Adam",
                    path_to_store_single_run=f"./running_experiments/del_close{DB}_{MODEL}_{experiment}"
                )
                res_close.append(f"{result_close['Test']['MRR']}")

                lists = {
                    "result_random_poisoned": result_random_poisoned,
                    "res_close":res_close,
                    "res_betw": res_betw,
                }

                lengths = {}
                for k, v in lists.items():
                    try:
                        lengths[k] = len(v)
                    except TypeError:
                        lengths[k] = 0 if v is None else 1

                # -----------------------------------------------------------------------------

                rows = [
                    ("Triple Injection Ratios", percentages),
                    ("Random", res_random),
                    ("Closeness", res_close),
                    ("Betweenness", res_betw),
                ]

                out_path = Path(
                    f"final_results/wo/{DB}/{MODEL}/del/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.csv")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(
                        f"final_results/wo/{DB}/{MODEL}/del/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.csv",
                        "w", newline="") as file:
                    writer = csv.writer(file)
                    for name, values in rows:
                        writer.writerow([name] + values)

                visualize_results(
                    f"final_results/wo/{DB}/{MODEL}/del/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.csv",
                    f"final_results/wo/{DB}/{MODEL}/del/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.png",
                    f"{DB}-{MODEL}")



