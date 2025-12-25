# run_kemeny_nx_experiments.py
import os
import shutil
import random
from pathlib import Path
from executer_4del import run_dicee_eval
from utils import set_seeds, load_triples, save_triples
from config import (DBS, MODELS, RECIPRIOCAL, PERCENTAGES, BATCH_SIZE, LEARNING_RATE,
                    NUM_EXPERIMENTS, NUM_EPOCHS, EMB_DIM, LOSS_FN, SCORING_TECH, OPTIM)

# import the builder
from kemeny_nx_make_datasets import build_kemeny_nx_datasets

# Roots
SAVED_DATASETS_ROOT = Path(f"./saved_datasets/{RECIPRIOCAL}/")      # per-run copies
SAVED_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

RUNS_ROOT = Path(f"./running_experiments/{RECIPRIOCAL}/")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(f"./final_results/{RECIPRIOCAL}/")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORTS_ROOT = Path(f"./reports/{RECIPRIOCAL}")
REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

# Precomputed NX datasets live here
KEMENY_NX_DATASETS_ROOT = Path(f"./kemeny_nx_datasets/{RECIPRIOCAL}/")
KEMENY_NX_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 12345
seed_src = random.Random(MASTER_SEED)
EXPERIMENT_SEEDS = [seed_src.randrange(2 ** 32) for _ in range(NUM_EXPERIMENTS)]

def save_deleted_and_eval(original_triples, kept_triples, feature_tag, DB, top_k_triples,
                          experiment_idx, MODEL, experiment_seed, test_path, valid_path):
    out_dir = (SAVED_DATASETS_ROOT / DB / "delete" / feature_tag / MODEL /
               str(top_k_triples) / str(experiment_seed))
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_set = set(original_triples)
    kept_set = set(kept_triples)
    removed = orig_set - kept_set

    save_triples(list(kept_triples), str(out_dir / "train.txt"))
    (out_dir / "removed.txt").write_text("\n".join("\t".join(x) for x in removed), encoding="utf-8")

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
        path_to_store_single_run=str(
            RUNS_ROOT / f"delete/{top_k_triples}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}"
        ),
    )
    return res["Test"]["MRR"]

def ensure_nx_datasets(DB, train_path, budgets_triples):
    """Build NX datasets for DB if missing; return map budget->folder path."""
    db_root = KEMENY_NX_DATASETS_ROOT / DB
    # detect which budgets already exist
    ready = {}
    for B in budgets_triples:
        sub = db_root / f"tri_{B}"
        if (sub / "train.txt").exists() and (sub / "removed.txt").exists():
            ready[B] = str(sub)
    missing = [B for B in budgets_triples if B not in ready]
    if missing:
        print(f"[{DB}] building NX Kemeny datasets for budgets (triples): {missing}")
        built = build_kemeny_nx_datasets(DB, RECIPRIOCAL, train_path, missing,
                                         out_root=str(KEMENY_NX_DATASETS_ROOT.parent))
        # build_kemeny_nx_datasets returns db‑scoped paths; normalize to our db_root
        ready.update({B: str(KEMENY_NX_DATASETS_ROOT / DB / f"tri_{B}") for B in missing})
    return ready

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    del_results_csv = RESULTS_ROOT / "all_del_result.csv"
    with del_results_csv.open("w", encoding="utf-8") as f:
        f.write("DB,MODEL,attack,noise_ratio,budget,removed,exp_idx,seed,MRR\n")

    for DB in DBS:
        TRIPLES_PATH = f"./KGs/{DB}/train.txt"
        VALID_PATH = f"./KGs/{DB}/valid.txt"
        TEST_PATH = f"./KGs/{DB}/test.txt"

        train_triples = load_triples(TRIPLES_PATH)
        n_train = len(train_triples)
        budgets_triples = [max(1, int(n_train * p)) for p in PERCENTAGES]
        print(f"\n[DB={DB}] n_train={n_train}, budgets_triples={budgets_triples}")

        # Ensure datasets exist for these budgets
        nx_sets = ensure_nx_datasets(DB, TRIPLES_PATH, budgets_triples)

        for MODEL in MODELS:
            print(f"  [MODEL={MODEL}]")
            for exp_idx, exp_seed in enumerate(EXPERIMENT_SEEDS):
                set_seeds(exp_seed)
                print(f"    [exp={exp_idx}] seed={exp_seed}")

                for noise_ratio, req_budget in zip(PERCENTAGES, budgets_triples):
                    kept_path = Path(nx_sets[req_budget]) / "train.txt"
                    kept_triples = load_triples(str(kept_path))
                    actual_removed = n_train - len(kept_triples)

                    feature_tag = f"kemeny_nx_tri_{req_budget}"
                    mrr = save_deleted_and_eval(
                        original_triples=train_triples,
                        kept_triples=kept_triples,
                        feature_tag=feature_tag,
                        DB=DB,
                        top_k_triples=req_budget,  # record requested; we log actual separately
                        experiment_idx=exp_idx,
                        MODEL=MODEL,
                        experiment_seed=exp_seed,
                        test_path=TEST_PATH,
                        valid_path=VALID_PATH,
                    )
                    print(f"        Kemeny-NX DEL MRR={mrr:.4f} | requested={req_budget} | actual_removed={actual_removed}")

                    with del_results_csv.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{DB},{MODEL},kemeny(del),{noise_ratio},{req_budget},"
                            f"{actual_removed},{exp_idx},{exp_seed},{mrr}\n"
                        )

if __name__ == "__main__":
    main()
