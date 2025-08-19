from dicee import KGE
from executer import run_dicee_eval
from utils import (set_seeds, load_embeddings, load_triples, select_adverserial_triples_blackbox,
                   select_adverserial_triples_whitebox, save_triples, compute_triple_centrality, visualize_results,
                    select_adversarial_triples_fgsm
                   )
from baselines import remove_random_triples
from utils_2 import select_adversarial_triples_fgsm_simple, select_k_top_loss, select_k_mmr, select_k_top_loss_fast, select_k_mmr_fast, select_adversarial_removals_fgsm
from baselines import poison_random, poison_centrality, remove_random_triples
from uttils_3 import select_adversarial_removals_fgsm_hardneg
import shutil
import csv
import random
from pathlib import Path

device = "cpu"

DB = "Countries-S1" #"KINSHIP" #"UMLS"
MODEL = "Keci" #"Pykeen_BoxE" #"Pykeen_RotatE" #"Pykeen_ComplEx" #"Keci" #"Pykeen_MuRE" #"Keci" #"Pykeen_MuRE" #"Pykeen_RotatE" #"Keci" #"Pykeen_ComplEx" #"Keci" #"Pykeen_BoxE" #"DeCaL" #"Pykeen_ComplEx" #Keci

ORACLE_PATH = f"./Experiments/{DB}_{MODEL}"
TRIPLES_PATH = f"./KGs/{DB}/train.txt"

ENTITY_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_entity_embeddings.csv"
RELATION_CSV = f"./Experiments/{DB}_{MODEL}/{MODEL}_relation_embeddings.csv"

valid_path = f"./KGs/{DB}/valid.txt"
test_path = f"./KGs/{DB}/test.txt"

triples = load_triples(TRIPLES_PATH)
val_triples = load_triples(valid_path)
test_triples = load_triples(test_path)

entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)

oracle = KGE(path=ORACLE_PATH)

hidden_dims = [512, 256, 128, 64]

active_learning_logs = "results/active_learning_logs.csv"

triples_count = len(triples)

percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50] #[0.1, 0.2, 0.30, 0.40, 0.50, 0.60, 0.70, 0.8]
perturbation_ratios = [int(triples_count * p) for p in percentages]

corruption_type = "random-one" #"all" #"random-one" #"rel" #"random-one" #"tail" #"rel"

def interleave_many(*lists_):
    out = []
    if not lists_:
        return out
    n = max(len(lst) for lst in lists_)
    for i in range(n):
        for lst in lists_:
            if i < len(lst):
                out.append(lst[i])
    return out


def store_poisoned_andeval(triples, adverserial_triples, feature, DB, top_k, corruption_type, experiment, MODEL):

    triples_after_adverserials_edis_wbox = triples + adverserial_triples[:top_k]

    random.shuffle(triples_after_adverserials_edis_wbox)

    save_triples(triples_after_adverserials_edis_wbox,
                 f"{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/train.txt")

    shutil.copy2(test_path, f"{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/test.txt")
    shutil.copy2(valid_path, f"{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/valid.txt")

    res = run_dicee_eval(
        dataset_folder=f"{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )

    return res['Test']['MRR']

#seeds = [42, 64, 84, 98, 115, 162, 185, 215, 241, 286, 310, 324, 346, 368, ]

seed_src = random.Random()

num_experiments = 10
experiment_seeds = [seed_src.randrange(2**32) for _ in range(num_experiments)]

forbidden = set(triples) | set(val_triples) | set(test_triples)

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

    #res_fgsm_main = []

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

    #fgsm_main = select_adversarial_triples_fgsm_simple(
    #    triples=triples,
    #    oracle=oracle,
    #    seed=experiment_seed,
    #    eps=1e-2,
    #    norm="linf",
    #)

    for idx, top_k in enumerate(perturbation_ratios):

        print("############## Poisoning Random #################")

        remaining, corrupted = poison_random(triples, top_k, corruption_type, experiment_seed)

        triples_after_random_poisoning = triples + corrupted

        random.shuffle(triples_after_random_poisoning)

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

        print("############## Poisoning Whitebox Active learning #################")


        low_scores_triples_simple = [item[0] for item in low_scores_simple]
        low_scores_triples_simple_to_store = store_poisoned_andeval(triples, low_scores_triples_simple, "low_scores_triples_simple", DB, top_k,
                                                            corruption_type, experiment, MODEL)
        res_wbox_low_scores_simple.append(low_scores_triples_simple_to_store)

        high_close_triples_simple_to_store = store_poisoned_andeval(triples, high_close_simple, "high_close_simple", DB, top_k,
                                                                     corruption_type, experiment, MODEL)
        res_wbox_high_closeness_simple.append(high_close_triples_simple_to_store)


        high_gradients_triples_simple = [item[0] for item in high_gradients_simple]
        high_gradients_triples_simple_to_store = store_poisoned_andeval(triples, high_gradients_triples_simple, "high_gradients_simple", DB, top_k,
                                                                corruption_type, experiment, MODEL)
        res_wbox_high_gradients_simple.append(high_gradients_triples_simple_to_store)

        #---------------

        #fgsm_main_triples = select_k_top_loss_fast(fgsm_main, k=top_k, forbidden=forbidden)
        #res_fgsm_main_to_store = store_poisoned_andeval(triples, fgsm_main_triples, "fgsm_main_triples", DB, top_k,
        #                                                        corruption_type, experiment, MODEL)
        #res_fgsm_main.append(res_fgsm_main_to_store)

        # ---------------

        adverserial_fgsm_triples = [item[0] for item in fgsm_adverserial_triples]
        adverserial_fgsm_triples_to_store = store_poisoned_andeval(triples, adverserial_fgsm_triples,
                                                                  "adverserial_fgsm_triples", DB, top_k,
                                                                  corruption_type, experiment, MODEL)
        res_wbox_adverserial_fgsm.append(adverserial_fgsm_triples_to_store)

        low_scores_fgsm_triples = [item[0] for item in low_scores_fgsm]
        low_scores_fgsm_triples_to_store = store_poisoned_andeval(triples, low_scores_fgsm_triples, "low_scores_fgsm_triples", DB, top_k,
                                                            corruption_type, experiment, MODEL)
        res_wbox_low_scores_fgsm.append(low_scores_fgsm_triples_to_store)

        #-----------
        high_closeness_fgsm_triples_to_store = store_poisoned_andeval(triples, high_close_fgsm, "high_close_fgsm", DB, top_k,
                                                                corruption_type, experiment, MODEL)
        res_high_closeness_fgsm.append(high_closeness_fgsm_triples_to_store)
        #-----------
        high_gradients_fgsm_triples = [item[0] for item in high_gradients_fgsm]
        high_gradients_fgsm_to_store = store_poisoned_andeval(triples, high_gradients_fgsm_triples,
                                                                      "high_gradients_fgsm_triples", DB, top_k,
                                                                      corruption_type, experiment, MODEL)
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
            #("res_fgsm_main", res_fgsm_main),
        ]

        out_path = Path(
            f"final_results/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(f"final_results/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for name, values in rows:
                    writer.writerow([name] + values)

        visualize_results(f"final_results/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}.csv", f"final_results/{DB}/{MODEL}/{corruption_type}/results{DB}-{MODEL}-{corruption_type}-{experiment}.png", f"{DB}-{MODEL}")

        lists_to_check = {
            "random":res_random,
            "triple injection ratios": percentages,
            "res_wbox_low_scores_simple": res_wbox_low_scores_simple,
            "res_wbox_high_closeness_simple": res_wbox_high_closeness_simple,
            "res_wbox_high_gradients_simple": res_wbox_high_gradients_simple,
            "res_wbox_low_scores_fgsm": res_wbox_low_scores_fgsm,
            "res_high_closeness_fgsm": res_high_closeness_fgsm,
            "res_high_gradients_fgsm": res_high_gradients_fgsm,
            "res_wbox_adverserial_fgsm": res_wbox_adverserial_fgsm,
            #"res_fgsm_main": res_fgsm_main,
        }
    
        lengths = [len(v) for v in lists_to_check.values()]
        target_len = lengths[0]
    
        mismatched = [name for name, lst in lists_to_check.items() if len(lst) != target_len]
    
        if mismatched:
            print("Lists with different length:", mismatched)
        else:
            print("All lists have the same length:", target_len)


