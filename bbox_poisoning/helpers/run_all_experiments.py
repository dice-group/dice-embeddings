from dicee import KGE
from executer import run_dicee_eval
from utils import (set_seeds, load_embeddings, load_triples,
                   save_triples, visualize_results,
                   select_adversarial_triples_fgsm
                   )
from baselines import poison_random
import shutil
import csv
import random
from pathlib import Path
from datetime import datetime
import json
import os, json, torch
import torch.nn.functional as F
import json
from utils import add_corrupted_by_betweenness, add_corrupted_by_closeness


from centerality_utils import (
    remove_by_edge_betweenness,
    remove_by_endpoint_pagerank,
    remove_by_endpoint_harmonic_closeness,
    add_corrupted_by_pagerank,
    add_corrupted_by_harmonic_closeness,
)




DBS = [ "UMLS", "KINSHIP",  "NELL-995-h100", "FB15k-237", "WN18RR"]
MODELS = [ "ComplEx", "DistMult", "ComplEx", "DistMult", 'Pykeen_TransE', 'Pykeen_TransH', "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL"] #, "Pykeen_BoxE" ]

batch_size = "256"
learning_rate = "0.01"


def store_poisoned_andeval(triples, adverserial_triples, feature, DB, top_k, corruption_type, experiment, MODEL, experiment_seed):
    set_seeds(experiment_seed)

    triples_after_adverserials_edis_wbox = triples + adverserial_triples[:top_k]

    random.shuffle(triples_after_adverserials_edis_wbox)

    save_triples(triples_after_adverserials_edis_wbox,
                 f"./saved_datasets/wo/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/train.txt")

    shutil.copy2(test_path,
                 f"./saved_datasets/wo/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/test.txt")
    shutil.copy2(valid_path,
                 f"./saved_datasets/wo/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/valid.txt")

    res = run_dicee_eval(
        dataset_folder=f"./saved_datasets/wo/{DB}/active_poisoning_whitebox/{MODEL}/{feature}/{top_k}/{corruption_type}/{experiment}/",
        model=MODEL,
        num_epochs="100",
        batch_size=batch_size,
        learning_rate=learning_rate,
        embedding_dim="32",
        loss_function="BCELoss",
        seed=experiment_seed,
        scoring_technique="KvsAll",
        optim="Adam",
        path_to_store_single_run=f"./running_experiments/FGSM_{DB}_{MODEL}_{feature}_{experiment}"
    )

    return res['Test']['MRR']


for DB in DBS:
    for MODEL in MODELS:

        ORACLE_PATH = f"./saved_models/wo/{DB}/{MODEL}"
        TRIPLES_PATH = f"./KGs/{DB}/train.txt"

        valid_path = f"./KGs/{DB}/valid.txt"
        test_path = f"./KGs/{DB}/test.txt"

        if MODEL.startswith("Pykeen_"):
            embedding_file_prefix = MODEL.split("Pykeen_")[1]
        else:
            embedding_file_prefix = MODEL

        ENTITY_CSV = f"./saved_models/wo/{DB}/{MODEL}/{embedding_file_prefix}_entity_embeddings.csv"
        RELATION_CSV = f"./saved_models/wo/{DB}/{MODEL}/{embedding_file_prefix}_relation_embeddings.csv"

        active_learning_logs = f"results/{DB}/{MODEL}/active_learning_logs.csv"

        corruption_type = "random-one"

        triples = load_triples(TRIPLES_PATH)
        val_triples = load_triples(valid_path)
        test_triples = load_triples(test_path)

        entity_emb, relation_emb = load_embeddings(ENTITY_CSV, RELATION_CSV)
        oracle = KGE(path=ORACLE_PATH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oracle.model.to(device)

        """
        with open(os.path.join(ORACLE_PATH, "configuration.json"), "r") as f:
            cfg = json.load(f)
        lr = cfg.get("lr", 1e-3)
        weight_decay = cfg.get("weight_decay", 0.0)
        optimizer = torch.optim.Adam(oracle.model.parameters(), lr=lr, weight_decay=weight_decay)

        ckpt = torch.load(os.path.join(ORACLE_PATH, "last.ckpt"), map_location="cpu")
        opt_states = ckpt.get("optimizer_states", None)
        assert opt_states is not None and len(opt_states) > 0, "No optimizer_states found in last.ckpt"

        optimizer.load_state_dict(opt_states[0])

        for st in optimizer.state.values():
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)

        oracle_optimizer = optimizer
        """

        triples_count = len(triples)
        percentages = [0.01, 0.02, 0.03, 0.04, 0.05] #[0.001, 0.002, 0.003, 0.004, 0.005] #[0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
        perturbation_ratios = [int(triples_count * p) for p in percentages]

        seed_src = random.Random()
        num_experiments = 5
        experiment_seeds = [seed_src.randrange(2 ** 32) for _ in range(num_experiments)]

        for experiment, experiment_seed in enumerate(experiment_seeds):

            set_seeds(experiment_seed)

            res_random = []

            #res_wbox_low_scores_simple = []
            #res_wbox_high_closeness_simple = []
            #res_wbox_high_gradients_simple = []
            #res_wbox_low_scores_fgsm = []
            #res_high_gradients_fgsm = []
            #res_wbox_adverserial_fgsm = []

            res_high_close_global_fgsm = []
            res_high_betw_global_fgsm = []
            res_high_close_local_fgsm = []
            res_high_betw_local_fgsm = []

            res_corrupted_bw = []
            res_corrupted_cl = []

            #res_to_remove_bw = []
            #res_to_remove_pr = []
            #res_to_remove_hc = []
            res_add_pr = []
            res_add_hc = []

            """
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
            """
            low_scores_fgsm, high_close_global_fgsm, high_betw_global_fgsm,  high_close_local_fgsm, high_betw_local_fgsm, high_gradients_fgsm, fgsm_adverserial_triples = select_adversarial_triples_fgsm(
                triples=triples,
                corruption_type=corruption_type,
                oracle=oracle,
                seed=experiment_seed,
                eps=1e-14,
                norm= "linf",
            )

            for idx, top_k in enumerate(perturbation_ratios):

                print("###################### Poisoning Random ##########################")

                remaining, corrupted = poison_random(triples, top_k, corruption_type, experiment_seed)
                triples_after_random_poisoning = triples + corrupted
                random.shuffle(triples_after_random_poisoning)

                save_triples(triples_after_random_poisoning, f"./saved_datasets/wo/{DB}/random/{MODEL}/{top_k}/{corruption_type}/{experiment}/train.txt")
                shutil.copy2(test_path, f"./saved_datasets/wo/{DB}/random/{MODEL}/{top_k}/{corruption_type}/{experiment}/test.txt")
                shutil.copy2(valid_path, f"./saved_datasets/wo/{DB}/random/{MODEL}/{top_k}/{corruption_type}/{experiment}/valid.txt")

                result_random_poisoned = run_dicee_eval(
                    dataset_folder= f"./saved_datasets/wo/{DB}/random/{MODEL}/{top_k}/{corruption_type}/{experiment}/",
                    model=MODEL,
                    num_epochs="100",
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    embedding_dim="32",
                    loss_function="BCELoss",
                    seed=experiment_seed,
                    path_to_store_single_run=f"./running_experiments/random_{DB}_{MODEL}_{experiment}"
                )
                res_random.append(f"{result_random_poisoned['Test']['MRR']}")

                print("############## Poisoning Whitebox Active learning #################")

                """
                low_scores_triples_simple = [item[0] for item in low_scores_simple]
                low_scores_triples_simple_to_store = store_poisoned_andeval(triples, low_scores_triples_simple,"low_scores_triples_simple", DB, top_k, corruption_type, experiment, MODEL, experiment_seed)
                res_wbox_low_scores_simple.append(low_scores_triples_simple_to_store)
                # -----------
                high_close_triples_simple_to_store = store_poisoned_andeval(triples, high_close_simple, "high_close_simple", DB, top_k, corruption_type, experiment, MODEL, experiment_seed)
                res_wbox_high_closeness_simple.append(high_close_triples_simple_to_store)
                # -----------
                high_gradients_triples_simple = [item[0] for item in high_gradients_simple]
                high_gradients_triples_simple_to_store = store_poisoned_andeval(triples, high_gradients_triples_simple,"high_gradients_simple", DB, top_k, corruption_type, experiment, MODEL, experiment_seed)
                res_wbox_high_gradients_simple.append(high_gradients_triples_simple_to_store)
                """
                # -----------

                #if experiment == 0: # because fgsm experiments are deterministic and multiple runs produces the same results, therefore, only one run is enough.
                #adverserial_fgsm_triples = [item[0] for item in fgsm_adverserial_triples]
                #adverserial_fgsm_triples_to_store = store_poisoned_andeval(triples, adverserial_fgsm_triples, "adverserial_fgsm_triples", DB, top_k, corruption_type, experiment, MODEL, experiment_seed)
                #res_wbox_adverserial_fgsm.append(adverserial_fgsm_triples_to_store)
                # -----------
                #low_scores_fgsm_triples = [item[0] for item in low_scores_fgsm]
                #low_scores_fgsm_triples_to_store = store_poisoned_andeval(triples, low_scores_fgsm_triples, "low_scores_fgsm_triples", DB, top_k, corruption_type, experiment, MODEL, experiment_seed)
                #res_wbox_low_scores_fgsm.append(low_scores_fgsm_triples_to_store)
                # -----------

                high_close_global_fgsm_to_store = store_poisoned_andeval(triples, high_close_global_fgsm, "high_close_global_fgsm", DB, top_k,
                                       corruption_type, experiment, MODEL, experiment_seed)
                res_high_close_global_fgsm.append(high_close_global_fgsm_to_store)
                # -----------
                high_betw_global_fgsm_to_store = store_poisoned_andeval(triples, high_betw_global_fgsm, "high_betw_global_fgsm", DB, top_k,
                                       corruption_type, experiment, MODEL, experiment_seed)
                res_high_betw_global_fgsm.append(high_betw_global_fgsm_to_store)
                # -----------
                high_close_local_fgsm_store = store_poisoned_andeval(triples, high_close_local_fgsm, "high_close_local_fgsm", DB, top_k,
                                       corruption_type, experiment, MODEL, experiment_seed)
                res_high_close_local_fgsm.append(high_close_local_fgsm_store)
                # -----------
                high_betw_local_fgsm_store = store_poisoned_andeval(triples, high_betw_local_fgsm, "high_betw_local_fgsm", DB, top_k,
                                       corruption_type, experiment, MODEL, experiment_seed)
                res_high_betw_local_fgsm.append(high_betw_local_fgsm_store)
                # -----------
                #high_gradients_fgsm_triples = [item[0] for item in high_gradients_fgsm]
                #high_gradients_fgsm_to_store = store_poisoned_andeval(triples, high_gradients_fgsm_triples,"high_gradients_fgsm_triples", DB, top_k, corruption_type, experiment, MODEL, experiment_seed)
                #res_high_gradients_fgsm.append(high_gradients_fgsm_to_store)
                # -----------------------------------------------------------------------------



                corrupted_bw = add_corrupted_by_betweenness(
                    triples, budget=top_k, mode="both", top_k_nodes=1000, avoid_existing_edge=True
                )
                corrupted_bw_store = store_poisoned_andeval(triples, corrupted_bw,
                                                                    "corrupted_bw", DB, top_k,
                                                                    corruption_type, experiment, MODEL, experiment_seed)
                res_corrupted_bw.append(corrupted_bw_store)
                #----
                corrupted_cl = add_corrupted_by_closeness(
                    triples, budget=top_k, mode="both", top_k_nodes=1000, undirected=False, avoid_existing_edge=True
                )
                corrupted_cl_store = store_poisoned_andeval(triples, corrupted_cl,
                                                            "corrupted_cl", DB, top_k,
                                                            corruption_type, experiment, MODEL, experiment_seed)
                res_corrupted_cl.append(corrupted_cl_store)


                """
                
                to_remove_bw = remove_by_edge_betweenness(triples, top_k, approx_k=256)  # set None for exact
                to_remove_pr = remove_by_endpoint_pagerank(triples, top_k)
                to_remove_hc = remove_by_endpoint_harmonic_closeness(triples, top_k, undirected=True)
                
                
                # to_remove_bw
                to_remove_bw_store = store_poisoned_andeval(
                    triples, to_remove_bw, "to_remove_bw", DB, top_k, corruption_type, experiment, MODEL,
                    experiment_seed
                )
                res_to_remove_bw.append(to_remove_bw_store)

                # to_remove_pr
                to_remove_pr_store = store_poisoned_andeval(
                    triples, to_remove_pr, "to_remove_pr", DB, top_k, corruption_type, experiment, MODEL,
                    experiment_seed
                )
                res_to_remove_pr.append(to_remove_pr_store)

                # to_remove_hc
                to_remove_hc_store = store_poisoned_andeval(
                    triples, to_remove_hc, "to_remove_hc", DB, top_k, corruption_type, experiment, MODEL,
                    experiment_seed
                )
                res_to_remove_hc.append(to_remove_hc_store)
                """

                # add_pr

                add_pr = add_corrupted_by_pagerank(triples, top_k, mode="both", top_k_nodes=1000,
                                                   avoid_existing_edge=True)
                add_hc = add_corrupted_by_harmonic_closeness(triples, top_k, mode="both", top_k_nodes=1000,
                                                             undirected=True, avoid_existing_edge=True)

                add_pr_store = store_poisoned_andeval(
                    triples, add_pr, "add_pr", DB, top_k, corruption_type, experiment, MODEL, experiment_seed
                )
                res_add_pr.append(add_pr_store)

                # add_hc
                add_hc_store = store_poisoned_andeval(
                    triples, add_hc, "add_hc", DB, top_k, corruption_type, experiment, MODEL, experiment_seed
                )
                res_add_hc.append(add_hc_store)


                lists = {
                    "triples_after_random_poisoning": triples_after_random_poisoning,
                    #"low_scores_triples_simple_to_store": low_scores_triples_simple_to_store,
                    #"high_gradients_triples_simple_to_store": high_gradients_triples_simple_to_store,
                    #"adverserial_fgsm_triples_to_store": adverserial_fgsm_triples_to_store,
                    #"low_scores_fgsm_triples_to_store": low_scores_fgsm_triples_to_store,
                    #"high_close_global_fgsm_to_store": high_close_global_fgsm_to_store,
                    "high_betw_global_fgsm_to_store": high_betw_global_fgsm_to_store,
                    "high_close_local_fgsm_store": high_close_local_fgsm_store,
                    "high_betw_local_fgsm_store": high_betw_local_fgsm_store,
                    "corrupted_bw_store": corrupted_bw_store,
                    "corrupted_cl_store": corrupted_cl_store,
                    #"high_gradients_fgsm_to_store": high_gradients_fgsm_to_store,
                    #"to_remove_bw_store": to_remove_bw_store,
                    #"to_remove_pr_store": to_remove_pr_store,
                    #"to_remove_hc_store": to_remove_hc_store,
                    "add_pr_store": add_pr_store,
                    "add_hc_store": add_hc_store,
                }

                lengths = {}
                for k, v in lists.items():
                    try:
                        lengths[k] = len(v)
                    except TypeError:
                        lengths[k] = 0 if v is None else 1  # treat scalars as 1 item

                with open(f"./reports/{DB}_{MODEL}_length_of_added_triples_check_report.json", "w") as f:
                    json.dump(lengths, f, indent=2)

                # -----------------------------------------------------------------------------

                rows = [
                    ("Triple Injection Ratios", percentages),
                    ("Random", res_random),
                    #("Low_Scores", res_wbox_low_scores_simple),
                    #("High_Closeness", res_wbox_high_closeness_simple),
                    #("High_Gradients", res_wbox_high_gradients_simple),
                    #("FGSM_Low_Scores", res_wbox_low_scores_fgsm),
                    ("FGSM_High_Closeness(Global)", res_high_close_global_fgsm),
                    ("FGSM_High_Betweenness(Global)", res_high_betw_global_fgsm),
                    ("FGSM_High_Closeness(Local)", res_high_close_local_fgsm),
                    ("FGSM_High_Betweenness(Local)", res_high_betw_local_fgsm),
                    ("Betweenness",res_corrupted_bw),
                    ("Closeness",res_corrupted_cl),
                    #("FGSM_High_Gradients", res_high_gradients_fgsm),
                    #("FGSM_High_Loss", res_wbox_adverserial_fgsm),

                    ("PageRank", res_add_pr),
                    ("Harmonic closeness", res_add_hc),
                ]

                out_path = Path(
                    f"final_results/wo/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}-seed-{experiment_seed}.csv")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(
                        f"final_results/wo/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}-seed-{experiment_seed}.csv",
                        "w", newline="") as file:
                    writer = csv.writer(file)
                    for name, values in rows:
                        writer.writerow([name] + values)

                visualize_results(
                    f"final_results/wo/{DB}/{MODEL}/{corruption_type}/results-{DB}-{MODEL}-{corruption_type}-{experiment}-seed-{experiment_seed}.csv",
                    f"final_results/wo/{DB}/{MODEL}/{corruption_type}/results{DB}-{MODEL}-{corruption_type}-{experiment}-seed-{experiment_seed}.png",
                    f"{DB}-{MODEL}")

                #if experiment == 0:
                lists_to_check = {
                    "random": res_random,
                    #"res_wbox_low_scores_simple": res_wbox_low_scores_simple,
                    #"res_wbox_high_closeness_simple": res_wbox_high_closeness_simple,
                    #"res_wbox_high_gradients_simple": res_wbox_high_gradients_simple,
                    #"res_wbox_low_scores_fgsm": res_wbox_low_scores_fgsm,
                    "res_high_close_global_fgsm": res_high_close_global_fgsm,
                    "res_high_betw_global_fgsm": res_high_betw_global_fgsm,
                    "res_high_close_local_fgsm": res_high_close_local_fgsm,
                    "res_high_betw_local_fgsm": res_high_betw_local_fgsm,
                    "Betweenness": res_corrupted_bw,
                    "Corrupted": res_corrupted_cl,
                    #"res_high_gradients_fgsm": res_high_gradients_fgsm,
                    #"res_wbox_adverserial_fgsm": res_wbox_adverserial_fgsm,
                    "PageRank": res_add_pr,
                    "Harmonic closeness":  res_add_hc,
                }

                lengths_map = {name: len(lst) for name, lst in lists_to_check.items()}
                report_path = Path(f"./reports/{DB}_{MODEL}_length_check_report.json")
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






