import os
import json
import shutil
import random
import csv
from pathlib import Path
from datetime import datetime
import torch
from dicee import KGE
from executer import run_dicee_eval
from utils import set_seeds, load_triples, save_triples, visualize_results
from baselines import poison_random
from centerality_utils import (
    add_corrupted_by_harmonic_closeness,
    add_corrupted_by_edge_betweenness,
)
from whitebox_poison_add import (
    add_corrupted_by_centrality_and_loss_forward,
    add_corrupted_by_global_argmax_forward,
    add_corrupted_by_fgsm_forward,
)

DBS = ["WN18RR"] #["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237"] #, ["WN18RR", "YAGO3-10"]
MODELS = ["DistMult", "ComplEx", "Pykeen_TransE", "Pykeen_TransH", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL", "Keci"]

recipriocal = "without_recipriocal"

ORACLE_ROOT = Path(f"./saved_models/{recipriocal}/")    
ORACLE_ROOT.mkdir(parents=True, exist_ok=True)

SAVED_DATASETS_ROOT = Path(f"./saved_datasets/{recipriocal}/")
SAVED_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

RUNS_ROOT = Path(f"./running_experiments/{recipriocal}/")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(f"./final_results/{recipriocal}/")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORTS_ROOT = Path(f"./reports/{recipriocal}")
REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 12345
seed_src = random.Random(MASTER_SEED)
NUM_EXPERIMENTS = 3
EXPERIMENT_SEEDS = [seed_src.randrange(2 ** 32) for _ in range(NUM_EXPERIMENTS)]
ORACLE_SEEDS = EXPERIMENT_SEEDS

PERCENTAGES = [0.02, 0.04, 0.08, 0.16, 0.32]

BATCH_SIZE = "256"
LEARNING_RATE = "0.01"
NUM_EPOCHS = "100"
EMB_DIM = "32"
LOSS_FN = "BCELoss"
SCORING_TECH = "KvsAll"
OPTIM = "Adam"

def store_poisoned__adverserial(
    triples,
    adversarial_triples,
    feature_tag,
    DB,
    top_k,
    experiment_idx,
    MODEL,
    experiment_seed,
    test_path,
    valid_path,
):
    poisons = adversarial_triples[:top_k]

    train_poisoned = triples + poisons
    random.shuffle(train_poisoned)

    out_dir = SAVED_DATASETS_ROOT / DB / "centerality" / "add" / feature_tag / MODEL / str(top_k) / str(experiment_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_triples(train_poisoned, str(out_dir / "train.txt"))
    shutil.copy2(test_path, str(out_dir / "test.txt"))
    shutil.copy2(valid_path, str(out_dir / "valid.txt"))

    res = run_dicee_eval(
        dataset_folder=str(out_dir),
        model=MODEL,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        embedding_dim=EMB_DIM,
        loss_function=LOSS_FN,
        seed=experiment_seed,
        scoring_technique=SCORING_TECH,
        optim=OPTIM,
        path_to_store_single_run=str(RUNS_ROOT / f"add/{top_k}/{feature_tag}_{DB}_{MODEL}_{experiment_idx}")
    )
    return res['Test']['MRR']

def load_oracle(DB: str, MODEL: str, oracle_seed: int):
    path = ORACLE_ROOT / DB / MODEL / str(oracle_seed)
    if not path.exists():
        raise FileNotFoundError(f"Oracle path not found: {path}")
    oracle = KGE(path=str(path))

    if not hasattr(oracle, "model"):
        raise AttributeError(f"Loaded oracle at {path} has no .model")
    if not hasattr(oracle, "entity_to_idx") or not hasattr(oracle, "relation_to_idx"):
        raise AttributeError(f"Loaded oracle at {path} is missing entity/relation maps")

    return oracle


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    

    for DB in DBS:
        TRIPLES_PATH = f"./KGs/{DB}/train.txt"
        VALID_PATH = f"./KGs/{DB}/valid.txt"
        TEST_PATH = f"./KGs/{DB}/test.txt"

        train_triples = load_triples(TRIPLES_PATH)
        val_triples = load_triples(VALID_PATH)
        test_triples = load_triples(TEST_PATH)
        forbidden = set(val_triples) | set(test_triples)

        n_train = len(train_triples)
        budgets = [int(n_train * p) for p in PERCENTAGES]

        for MODEL in MODELS:

            for exp_idx, (oracle_seed, exp_seed) in enumerate(zip(ORACLE_SEEDS, EXPERIMENT_SEEDS)):
                set_seeds(exp_seed)

                try:
                    oracle = load_oracle(DB, MODEL, oracle_seed)
                except (FileNotFoundError, AttributeError) as e:
                    print(f"[SKIP] {DB}/{MODEL}/seed={oracle_seed}: {e}")
                    continue

                device = next(oracle.model.parameters()).device if any(True for _ in oracle.model.parameters()) else torch.device("cpu")
                oracle.model.to(device).eval()

                res_random = []
                res_add_pr = []
                res_add_hc = []
                res_add_wb_centloss = []
                res_add_wb_gax = []
                res_add_wb_fgsm = []
                res_simple_betweenness = []
                res_simple_closeness = []

                for top_k in budgets:
                    print(f"\n=== {DB} | {MODEL} | oracle_seed={oracle_seed} | budget={top_k} ===")

                    # -------- Random baseline --------
                    print("[Random] generating...")
                    _, corrupted = poison_random(train_triples, top_k, "random-one", exp_seed)

                    mrr_random = store_poisoned__adverserial(
                        train_triples, corrupted, "random", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_random.append(f"{mrr_random}")

                    # -------- Harmonic closeness  --------
                    print("[Harmonic] generating...")
                    add_hc = add_corrupted_by_harmonic_closeness(
                        train_triples, top_k, mode="both", top_k_nodes=1000, undirected=True, avoid_existing_edge=True
                    )
                    mrr_hc = store_poisoned__adverserial(
                        train_triples, add_hc, "hc", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_add_hc.append(mrr_hc)

                    # -------- Harmonic closeness --------
                    print("[Betweenness] generating...")
                    add_bw = add_corrupted_by_edge_betweenness(
                        train_triples, top_k, mode="both", top_k_nodes=1000, undirected=True, avoid_existing_edge=True
                    )
                    mrr_bw = store_poisoned__adverserial(
                        train_triples, add_bw, "bw", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_simple_betweenness.append(mrr_bw)

                    # -------- Whitebox Cent+Loss --------
                    print("[WB Cent+Loss] generating...")
                    add_wb_centloss = add_corrupted_by_centrality_and_loss_forward(
                        train_triples,
                        model=oracle.model,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        budget=top_k,
                        centrality="harmonic",     
                        undirected=True,
                        mode="both",
                        top_k_nodes=1000,
                        avoid_existing_edge=True,
                        restrict_by_relation=False,
                        forbidden=forbidden,
                        batch_size=10000,
                        device=device,
                    )
                    mrr_wb_cent = store_poisoned__adverserial(
                        train_triples, add_wb_centloss, "wb_centloss_forward",
                        DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_add_wb_centloss.append(mrr_wb_cent)

                    # -------- Whitebox Global Argmax --------
                    print("[WB GlobalArgmax] generating (exhaustive)...")
                    add_wb_gax = add_corrupted_by_global_argmax_forward(
                        train_triples,
                        model=oracle.model,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        budget=top_k,
                        mode="both",
                        avoid_existing_edge=True,
                        forbidden=forbidden,
                        per_anchor_topk=1,          
                        batch_size=10000,          
                        anchor_cap=None,           
                        device=device,
                    )
                    mrr_wb_gax = store_poisoned__adverserial(
                        train_triples, add_wb_gax, "wb_gargmax_forward",
                        DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_add_wb_gax.append(mrr_wb_gax)

                    #---------------------------------------
                    print("[WB FGSM Add] generating...")
                    add_wb_fgsm = add_corrupted_by_fgsm_forward(
                        train_triples,
                        model=oracle.model,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        budget=top_k,
                        eps=0.25,
                        norm="linf",                # or "l2"
                        pattern="best-of-three",    
                        topk_neighbors=32,
                        avoid_existing_edge=True,
                        restrict_by_relation=False, # True for schema-consistent poisons
                        forbidden=forbidden,
                        device=next(oracle.model.parameters()).device,
                        progress_every=5000,      
                    )
                    mrr_wb_fgsm = store_poisoned__adverserial(
                        train_triples, add_wb_fgsm, "wb_fgsm_forward",
                        DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_add_wb_fgsm.append(mrr_wb_fgsm)

                    # ----- quick length checks report -----
                    lengths_map = {
                        "Random": len(res_random),
                        "Cent+Loss": len(res_add_wb_centloss),
                        "GlobalArgmax": len(res_add_wb_gax),
                        "FGSM": len(res_add_wb_fgsm),
                        "Simple Closeness": len(res_add_hc),
                        "Simple Betweenness": len(res_simple_betweenness),
                    }
                    report_path = REPORTS_ROOT / f"{DB}_{MODEL}_len_report_exp{exp_idx}_k{top_k}.json"
                    report = {
                        "status": "ok" if len(set(lengths_map.values())) == 1 else "mismatch",
                        "lengths": lengths_map,
                        "timestamp": datetime.now().isoformat(timespec="seconds")
                    }
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(report, indent=2))

                # ----- write CSV + figure for this (DB, MODEL, exp_idx) -----
                out_dir = RESULTS_ROOT / DB / MODEL / "add"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_csv = out_dir / f"results-{DB}-{MODEL}-{exp_idx}-seed-{exp_seed}.csv"
                rows = [
                    ("Triple Injection Ratios", PERCENTAGES),
                    ("Random", res_random),
                    ("Cent+Loss", res_add_wb_centloss),
                    ("GlobalArgmax", res_add_wb_gax),
                    ("FGSM", res_add_wb_fgsm),
                    ("Simple Closeness", res_add_hc),
                    ("Simple Betweenness", res_simple_betweenness),
                ]
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    for name, values in rows:
                        w.writerow([name] + values)

                # Plot
                visualize_results(
                    str(out_csv),
                    str(out_dir / f"results-{DB}-{MODEL}-{exp_idx}-seed-{exp_seed}.png"),
                    f"{DB}-{MODEL}"
                )

if __name__ == "__main__":
    main()
