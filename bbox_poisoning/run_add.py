from executer import run_dicee_eval
from utils import (set_seeds, load_triples,
                   save_triples, visualize_results,
                   )
from baselines import poison_random
import shutil
import csv
from pathlib import Path
from datetime import datetime
import json
from centerality_utils import (
    add_corrupted_by_pagerank,
    add_corrupted_by_harmonic_closeness,
)
import random

DBS = ["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR"]
MODELS = [ "DistMult", "ComplEx", 'Pykeen_TransE', 'Pykeen_TransH', "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL" ]

batch_size = "256"
learning_rate = "0.01"

def store_poisoned_andeval(triples, adverserial_triples, feature, DB, top_k, corruption_type, experiment, MODEL,
                           experiment_seed,  test_path, valid_path):

    triples_after_adverserials_edis_wbox = triples + adverserial_triples[:top_k]

    random.shuffle(triples_after_adverserials_edis_wbox)

    save_triples(triples_after_adverserials_edis_wbox,
                 f"./saved_datasets/{DB}/centerality/add/{feature}/{MODEL}/{top_k}/{experiment}/train.txt")

    shutil.copy2(test_path,
                 f"./saved_datasets/{DB}/centerality/add/{feature}/{MODEL}/{top_k}/{experiment}/test.txt")
    shutil.copy2(valid_path,
                 f"./saved_datasets/{DB}/centerality/add/{feature}/{MODEL}/{top_k}/{experiment}/valid.txt")

    res = run_dicee_eval(
        dataset_folder=f"./saved_datasets/{DB}/centerality/add/{feature}/{MODEL}/{top_k}/{experiment}/",
        model=MODEL,
        num_epochs="100",
        batch_size=batch_size,
        learning_rate=learning_rate,
        embedding_dim="32",
        loss_function="BCELoss",
        seed=experiment_seed,
        scoring_technique="KvsAll",
        optim="Adam",
        path_to_store_single_run=f"./running_experiments/add/{feature}_{DB}_{MODEL}_{experiment}"
    )

    return res['Test']['MRR']


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
            res_add_pr = []
            res_add_hc = []

            for idx, top_k in enumerate(perturbation_ratios):

                print("###################### Poisoning Random ##########################")

                remaining, corrupted = poison_random(triples, top_k, corruption_type, experiment_seed)
                triples_after_random_poisoning = triples + corrupted
                random.shuffle(triples_after_random_poisoning)

                save_triples(triples_after_random_poisoning,
                             f"./saved_datasets/{DB}/random/add/{MODEL}/{top_k}/{experiment}/train.txt")
                shutil.copy2(test_path,
                             f"./saved_datasets/{DB}/random/add/{MODEL}/{top_k}/{experiment}/test.txt")
                shutil.copy2(valid_path,
                             f"./saved_datasets/{DB}/random/add/{MODEL}/{top_k}/{experiment}/valid.txt")

                result_random_poisoned = run_dicee_eval(
                    dataset_folder=f"./saved_datasets/{DB}/random/add/{MODEL}/{top_k}/{experiment}/",
                    model=MODEL,
                    num_epochs="100",
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    embedding_dim="32",
                    loss_function="BCELoss",
                    seed=experiment_seed,
                    scoring_technique="KvsAll",
                    optim="Adam",
                    path_to_store_single_run=f"./running_experiments/random_{DB}_{MODEL}_{experiment}"
                )
                res_random.append(f"{result_random_poisoned['Test']['MRR']}")

                print("############## Poisoning Whitebox Active learning #################")

                # add_pr
                add_pr = add_corrupted_by_pagerank(triples, top_k, mode="both", top_k_nodes=1000,
                                                   avoid_existing_edge=True)

                add_hc = add_corrupted_by_harmonic_closeness(triples, top_k, mode="both", top_k_nodes=1000,
                                                             undirected=True, avoid_existing_edge=True)

                add_pr_store = store_poisoned_andeval(triples, add_pr, "pr", DB, top_k, corruption_type, experiment, MODEL, experiment_seed, test_path=test_path, valid_path=valid_path)
                res_add_pr.append(add_pr_store)

                # add_hc
                add_hc_store = store_poisoned_andeval(triples, add_hc, "hc", DB, top_k, corruption_type, experiment, MODEL, experiment_seed, test_path=test_path, valid_path=valid_path)
                res_add_hc.append(add_hc_store)

                #####################
                lists = {
                    "triples_after_random_poisoning": triples_after_random_poisoning,
                    "add_pr_store": add_pr_store,
                    "add_hc_store": add_hc_store,
                }

                lengths = {}
                for k, v in lists.items():
                    try:
                        lengths[k] = len(v)
                    except TypeError:
                        lengths[k] = 0 if v is None else 1

                with open(f"./reports/{DB}_{MODEL}_length_of_added_triples_check_report.json", "w") as f:
                    json.dump(lengths, f, indent=2)

                # -----------------------------------------------------------------------------

                rows = [
                    ("Triple Injection Ratios", percentages),
                    ("Random", res_random),
                    ("PageRank", res_add_pr),
                    ("Harmonic closeness", res_add_hc),
                ]


                out_path = Path(
                    f"final_results/wo/{DB}/{MODEL}/add/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.csv")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(
                        f"final_results/wo/{DB}/{MODEL}/add/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.csv",
                        "w", newline="") as file:
                    writer = csv.writer(file)
                    for name, values in rows:
                        writer.writerow([name] + values)

                visualize_results(
                     f"final_results/wo/{DB}/{MODEL}/add/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.csv",
                    f"final_results/wo/{DB}/{MODEL}/add/results-{DB}-{MODEL}-{experiment}-seed-{experiment_seed}.png",
                    f"{DB}-{MODEL}")

                lists_to_check = {
                    "random": res_random,
                    "PageRank": res_add_pr,
                    "Harmonic closeness": res_add_hc,
                }

                lengths_map = {name: len(lst) for name, lst in lists_to_check.items()}

                report_path = Path(f"./reports/{DB}_{MODEL}_len_report_exp{experiment}_k{top_k}.json")
                report_path.parent.mkdir(parents=True, exist_ok=True)

                if not lengths_map:
                    report = {
                        "status": "empty",
                        "message": "lists_to_check is empty; nothing to compare.",
                        "timestamp": datetime.now().isoformat(timespec="seconds")
                    }
                    report_path.write_text(json.dumps(report, indent=2))
                    raise AssertionError(report["message"])

                target_len = next(iter(lengths_map.values()))
                mismatched = [name for name, L in lengths_map.items() if L != target_len]

                report = {
                    "status": "ok" if not mismatched else "mismatch",
                    "target_len": target_len,
                    "lengths": lengths_map,
                    "mismatched": {name: lengths_map[name] for name in mismatched},
                    "timestamp": datetime.now().isoformat(timespec="seconds")
                }
                report_path.write_text(json.dumps(report, indent=2))

                assert not mismatched, (
                        "Length mismatch: expected all lists to have length "
                        f"{target_len}, but these differ: "
                        + ", ".join(f"{name} (len={lengths_map[name]})" for name in mismatched)
                )






