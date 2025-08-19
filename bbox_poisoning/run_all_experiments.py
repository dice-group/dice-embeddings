from dicee import KGE
from executer import run_dicee_eval
from utils import (set_seeds, load_embeddings, load_triples,
                   select_adverserial_triples_whitebox, save_triples, visualize_results,
                   select_adversarial_triples_fgsm
                   )
from baselines import poison_random
import shutil
import csv
import random
from pathlib import Path
from datetime import datetime
import json


def store_poisoned_andeval(triples, adverserial_triples, feature, DB, top_k, corruption_type, experiment, MODEL):
    triples_after_adverserials_edis_wbox = triples + adverserial_triples[:top_k]

    random.shuffle(triples_after_adverserials_edis_wbox)

    save_triples(triples_after_adverserials_edis_wbox,
                 f"./saved_datasets/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/train.txt")

    shutil.copy2(test_path,
                 f"./saved_datasets/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/test.txt")
    shutil.copy2(valid_path,
                 f"./saved_datasets/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/valid.txt")

    res = run_dicee_eval(
        dataset_folder=f"./saved_datasets/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
        path_to_store_single_run=f"./running_experiments/FGSM_{DB}_{MODEL}_{feature}_{experiment}"
    )

    return res['Test']['MRR']



device = "cpu"

DBS = ["UMLS", "KINSHIP", "FB15k-237", "NELL-995-h100", "WN18RR", "YAGO3-10"]
MODELS = ["QMult", "Keci", "ComplEx", "DistMult", "Pykeen_MuRE", "Pykeen_RotatE", "Pykeen_BoxE", "DeCaL"]

for DB in DBS:
    for MODEL in MODELS:

        ORACLE_PATH = f"./saved_models/{DB}/{MODEL}"
        TRIPLES_PATH = f"./KGs/{DB}/train.txt"

        valid_path = f"./KGs/{DB}/valid.txt"
        test_path = f"./KGs/{DB}/test.txt"

        if MODEL.startswith("Pykeen_"):
            embedding_file_prefix = MODEL.split("Pykeen_")[1]
        else:
            embedding_file_prefix = MODEL

        ENTITY_CSV = f"./saved_models/{DB}/{MODEL}/{embedding_file_prefix}_entity_embeddings.csv"
        RELATION_CSV = f"./saved_models/{DB}/{MODEL}/{embedding_file_prefix}_relation_embeddings.csv"

        active_learning_logs = f"results/{DB}/{MODEL}/active_learning_logs.csv"

        corruption_type = "random-one"

        triples = load_triples(TRIPLES_PATH)
        val_triples = load_triples(valid_path)
        test_triples = load_triples(test_path)

        entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)
        oracle = KGE(path=ORACLE_PATH)

        triples_count = len(triples)
        percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50]
        perturbation_ratios = [int(triples_count * p) for p in percentages]

        seed_src = random.Random()
        num_experiments = 5
        experiment_seeds = [seed_src.randrange(2 ** 32) for _ in range(num_experiments)]


        for experiment, experiment_seed in enumerate(experiment_seeds):
            set_seeds(experiment_seed)

            res_random = []

            res_wbox_low_scores_simple = []
            res_wbox_high_closeness_simple = []
            res_wbox_high_gradients_simple = []

            res_wbox_low_scores_fgsm = []
            res_high_closeness_fgsm = []
            res_high_gradients_fgsm = []
            res_wbox_adverserial_fgsm = []

            (
                low_scores_simple,
                high_close_simple,
                high_gradients_simple,
            ) = select_adverserial_triples_whitebox(
                entity_emb=entity_emb,
                relation_emb=relation_emb,
                triples=triples,
                corruption_type=corruption_type,
                oracle=oracle,
                seed=experiment_seed
            )

            low_scores_fgsm, high_close_fgsm, high_gradients_fgsm, fgsm_adverserial_triples = select_adversarial_triples_fgsm(
                triples=triples,
                corruption_type=corruption_type,
                oracle=oracle,
                seed=experiment_seed,
                eps=1e-2,
                norm="linf",
            )

            for idx, top_k in enumerate(perturbation_ratios):

                print("###################### Poisoning Random ##########################")

                remaining, corrupted = poison_random(triples, top_k, corruption_type, experiment_seed)
                triples_after_random_poisoning = triples + corrupted
                random.shuffle(triples_after_random_poisoning)

                save_triples(triples_after_random_poisoning, f"./saved_models/{DB}/{MODEL}/random/{top_k}/{corruption_type}/{experiment}/train.txt")
                shutil.copy2(test_path, f"./saved_models/{DB}/{MODEL}/random/{top_k}/{corruption_type}/{experiment}/test.txt")
                shutil.copy2(valid_path, f"./saved_models/{DB}/{MODEL}/random/{top_k}/{corruption_type}/{experiment}/valid.txt")

                result_random_poisoned = run_dicee_eval(
                    dataset_folder=f"./saved_models/{DB}/{MODEL}/random/{top_k}/{corruption_type}/{experiment}/",
                    model=MODEL,
                    num_epochs="100",
                    batch_size="1024",
                    learning_rate="0.1",
                    embedding_dim="32",
                    loss_function="BCELoss",
                    path_to_store_single_run=f"./running_experiments/random_{DB}_{MODEL}_{experiment}"
                )
                res_random.append(f"{result_random_poisoned['Test']['MRR']}")

                print("############## Poisoning Whitebox Active learning #################")

                low_scores_triples_simple = [item[0] for item in low_scores_simple]
                low_scores_triples_simple_to_store = store_poisoned_andeval(triples, low_scores_triples_simple,"low_scores_triples_simple", DB, top_k, corruption_type, experiment, MODEL)
                res_wbox_low_scores_simple.append(low_scores_triples_simple_to_store)
                # -----------
                high_close_triples_simple_to_store = store_poisoned_andeval(triples, high_close_simple, "high_close_simple", DB, top_k, corruption_type, experiment, MODEL)
                res_wbox_high_closeness_simple.append(high_close_triples_simple_to_store)
                # -----------
                high_gradients_triples_simple = [item[0] for item in high_gradients_simple]
                high_gradients_triples_simple_to_store = store_poisoned_andeval(triples, high_gradients_triples_simple,"high_gradients_simple", DB, top_k, corruption_type, experiment, MODEL)
                res_wbox_high_gradients_simple.append(high_gradients_triples_simple_to_store)
                # -----------

                if experiment == 0: # because fgsm experiments are deterministic and multiple runs produces the same results, therefore, only one run is enough.
                    adverserial_fgsm_triples = [item[0] for item in fgsm_adverserial_triples]
                    adverserial_fgsm_triples_to_store = store_poisoned_andeval(triples, adverserial_fgsm_triples, "adverserial_fgsm_triples", DB, top_k, corruption_type, experiment, MODEL)
                    res_wbox_adverserial_fgsm.append(adverserial_fgsm_triples_to_store)
                    # -----------
                    low_scores_fgsm_triples = [item[0] for item in low_scores_fgsm]
                    low_scores_fgsm_triples_to_store = store_poisoned_andeval(triples, low_scores_fgsm_triples, "low_scores_fgsm_triples", DB, top_k, corruption_type, experiment, MODEL)
                    res_wbox_low_scores_fgsm.append(low_scores_fgsm_triples_to_store)
                    # -----------
                    high_closeness_fgsm_triples_to_store = store_poisoned_andeval(triples, high_close_fgsm, "high_close_fgsm", DB, top_k, corruption_type, experiment, MODEL)
                    res_high_closeness_fgsm.append(high_closeness_fgsm_triples_to_store)
                    # -----------
                    high_gradients_fgsm_triples = [item[0] for item in high_gradients_fgsm]
                    high_gradients_fgsm_to_store = store_poisoned_andeval(triples, high_gradients_fgsm_triples,"high_gradients_fgsm_triples", DB, top_k, corruption_type, experiment, MODEL)
                    res_high_gradients_fgsm.append(high_gradients_fgsm_to_store)
                # -----------------------------------------------------------------------------

                rows = [
                    ("Triple Injection Ratios", percentages),
                    ("Random", res_random),
                    ("Low_Scores", res_wbox_low_scores_simple),
                    ("High_Closeness", res_wbox_high_closeness_simple),
                    ("High_Gradients", res_wbox_high_gradients_simple),
                    ("FGSM_Low_Scores", res_wbox_low_scores_fgsm),
                    ("FGSM_High_Closeness", res_high_closeness_fgsm),
                    ("FGSM_High_Gradients", res_high_gradients_fgsm),
                    ("FGSM_High_Loss", res_wbox_adverserial_fgsm),
                ]

                out_path = Path(
                    f"final_results/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(
                        f"final_results/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv",
                        "w", newline="") as file:
                    writer = csv.writer(file)
                    for name, values in rows:
                        writer.writerow([name] + values)

                visualize_results(
                    f"final_results/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv",
                    f"final_results/{DB}/{MODEL}/{corruption_type}/results{DB}-{MODEL}-{corruption_type}-{experiment}.png",
                    f"{DB}-{MODEL}")

                if experiment == 0:
                    lists_to_check = {
                        "random": res_random,
                        "res_wbox_low_scores_simple": res_wbox_low_scores_simple,
                        "res_wbox_high_closeness_simple": res_wbox_high_closeness_simple,
                        "res_wbox_high_gradients_simple": res_wbox_high_gradients_simple,
                        "res_wbox_low_scores_fgsm": res_wbox_low_scores_fgsm,
                        "res_high_closeness_fgsm": res_high_closeness_fgsm,
                        "res_high_gradients_fgsm": res_high_gradients_fgsm,
                        "res_wbox_adverserial_fgsm": res_wbox_adverserial_fgsm,
                    }


                    lengths_map = {name: len(lst) for name, lst in lists_to_check.items()}
                    report_path = Path(f"./reports/{DB}_{MODEL}_length_check_report.json")

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



