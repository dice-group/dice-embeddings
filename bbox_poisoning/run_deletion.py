import os
import json
import shutil
import random
import csv
from pathlib import Path
from datetime import datetime
import torch
from dicee import KGE
from executer_4del import run_dicee_eval
from utils import set_seeds, load_triples, save_triples, visualize_results
from whitebox_poison_del import (
    remove_by_centrality_plus_loss_forward,      
    remove_by_global_argmax_forward,            
    remove_by_gradient_influence_forward,       
    remove_by_endpoint_closeness,   
    remove_by_edge_betweenness        
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

def save_deleted_and_eval(
    original_triples,
    kept_triples,
    feature_tag,
    DB,
    top_k,
    experiment_idx,
    MODEL,
    experiment_seed,
    test_path,
    valid_path,
):
    out_dir = SAVED_DATASETS_ROOT / DB / "centerality" / "delete" / feature_tag / MODEL / str(top_k) / str(experiment_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    removed = set(original_triples) - set(kept_triples)
    save_triples(list(kept_triples), str(out_dir / "train.txt"))
    (out_dir / "removed.txt").write_text("\n".join(["\t".join(x) for x in removed]))

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
        path_to_store_single_run=str(RUNS_ROOT / f"delete/{top_k}/{feature_tag}_{DB}_{MODEL}_{experiment_idx}")
    )
    return res["Test"]["MRR"]

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

def delete_random(triples, k, seed):
    rng = random.Random(seed)
    n = len(triples)
    if k <= 0:
        return [], list(triples)
    k = min(k, n)
    idx = list(range(n))
    rng.shuffle(idx)
    pick = set(idx[:k])
    removed = [triples[i] for i in pick]
    kept = [t for i, t in enumerate(triples) if i not in pick]
    return removed, kept

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

        n_train = len(train_triples)
        budgets = [max(1, int(n_train * p)) for p in PERCENTAGES]

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

                res_rand = []
                res_wb_centloss = []
                res_wb_gargmax = []
                res_wb_ginfluence = []
                res_simple_closeness = []
                res_simple_betweenness = []

                for top_k in budgets:
                    print(f"\n=== DELETE | {DB} | {MODEL} | oracle_seed={oracle_seed} | budget={top_k} ===")

                    # -------- Random deletion --------
                    print("[RandomDelete] selecting...")
                    _, kept_rand = delete_random(train_triples, top_k, seed=exp_seed)
                    mrr_rand = save_deleted_and_eval(
                        train_triples, kept_rand, "random", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_rand.append(f"{mrr_rand}")

                    # -------- Whitebox: Centrality + Loss --------
                    print("[WB Cent+Loss Delete] selecting...")
                    removed_cl, kept_cl = remove_by_centrality_plus_loss_forward(
                        train_triples,
                        model=oracle.model,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        budget=top_k,
                        centrality="harmonic",
                        undirected=True,
                        mode="endpoint",              
                        alpha=1.0,                   
                        batch_size=10000,
                        device=device,
                    )
                    mrr_wb_cl = save_deleted_and_eval(
                        train_triples, kept_cl, "wb_centloss_forward", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_wb_centloss.append(mrr_wb_cl)

                    # -------- Whitebox: Global Argmax --------
                    print("[WB GlobalArgmax Delete] selecting...")
                    removed_ga, kept_ga = remove_by_global_argmax_forward(
                        train_triples,
                        model=oracle.model,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        budget=top_k,
                        criterion="low_loss",         
                        batch_size=10000,
                        device=device,
                    )
                    mrr_wb_ga = save_deleted_and_eval(
                        train_triples, kept_ga, "wb_gargmax_forward", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_wb_gargmax.append(mrr_wb_ga)

                    # -------- Whitebox: Gradient Influence (FGSM-style) --------
                    print("[WB GradInfluence Delete] selecting...")
                    removed_gi, kept_gi = remove_by_gradient_influence_forward(
                        train_triples,
                        model=oracle.model,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        budget=top_k,
                        p="l2",                       # or "linf"
                        device=device,
                        show_progress=False,
                    )
                    mrr_wb_gi = save_deleted_and_eval(
                        train_triples, kept_gi, "wb_gradinfluence_forward", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_wb_ginfluence.append(mrr_wb_gi)

                    #-----------------------------------------

                    to_remove_cl = remove_by_endpoint_closeness(train_triples, top_k, undirected=False)

                    triples_after_removal_close = [t for t in train_triples if t not in to_remove_cl]

                    random.shuffle(triples_after_removal_close)

                    mrr_wb_simple_closeness = save_deleted_and_eval(
                        train_triples, triples_after_removal_close, "simple_closeness", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_simple_closeness.append(mrr_wb_simple_closeness)

                    #-----------------------------------------

                    to_remove_bw = remove_by_edge_betweenness(train_triples, top_k, approx_k=100)

                    triples_after_removal_betw = [t for t in train_triples if t not in to_remove_bw]

                    random.shuffle(triples_after_removal_betw)

                    mrr_hc_simple_betweenness = save_deleted_and_eval(
                        train_triples, triples_after_removal_betw, "simple_betweenness", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_simple_betweenness.append(mrr_hc_simple_betweenness)

                    # ----- quick length check report -----
                    lengths_map = {
                        "Random": len(res_rand),
                        "Cent+Loss": len(res_wb_centloss),
                        "GlobalArgmax": len(res_wb_gargmax),
                        "GradInfluence": len(res_wb_ginfluence),
                        "Simple Closeness": len(res_simple_closeness),
                        "Simple Betweenness": len(res_simple_betweenness),
                    }
                    report_path = REPORTS_ROOT / f"{DB}_{MODEL}_len_report_delete_exp{exp_idx}_k{top_k}.json"
                    report = {
                        "status": "ok" if len(set(lengths_map.values())) == 1 else "mismatch",
                        "lengths": lengths_map,
                        "timestamp": datetime.now().isoformat(timespec="seconds")
                    }
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(report, indent=2))

                # ----- write CSV + figure for this (DB, MODEL, exp_idx) -----
                out_dir = RESULTS_ROOT / DB / MODEL / "delete"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_csv = out_dir / f"results-delete-{DB}-{MODEL}-{exp_idx}-seed-{exp_seed}.csv"
                rows = [
                    ("Deletion Ratios", PERCENTAGES),
                    ("RandomDelete", res_rand),
                    ("Cent+Loss ", res_wb_centloss),
                    ("GlobalArgmax ", res_wb_gargmax),
                    ("GradInfluence ", res_wb_ginfluence),
                    ("Simple Closeness ", res_simple_closeness),
                    ("Simple Betweenness ", res_simple_betweenness),
                ]
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    for name, values in rows:
                        w.writerow([name] + values)

                visualize_results(
                    str(out_csv),
                    str(out_dir / f"results-delete-{DB}-{MODEL}-{exp_idx}-seed-{exp_seed}.png"),
                    f"{DB}-{MODEL} (Delete)"
                )

if __name__ == "__main__":
    main()
