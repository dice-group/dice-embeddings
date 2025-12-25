import os
import shutil
import random
from pathlib import Path

from executer_4del import run_dicee_eval
from utils import set_seeds, load_triples, save_triples
from config import (
    DBS,
    MODELS,
    RECIPRIOCAL,
    PERCENTAGES,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EXPERIMENTS,
    NUM_EPOCHS,
    EMB_DIM,
    LOSS_FN,
    SCORING_TECH,
    OPTIM,
)

# ----------------- roots -----------------
SAVED_DATASETS_ROOT = Path(f"./saved_datasets/{RECIPRIOCAL}/")
SAVED_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

RUNS_ROOT = Path(f"./running_experiments/{RECIPRIOCAL}/")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(f"./final_results/{RECIPRIOCAL}/")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORTS_ROOT = Path(f"./reports/{RECIPRIOCAL}")
REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

# ----------------- seeds -----------------
MASTER_SEED = 12345
_seed_src = random.Random(MASTER_SEED)
EXPERIMENT_SEEDS = [_seed_src.randrange(2**32) for _ in range(NUM_EXPERIMENTS)]

# ----------------- helpers -----------------
def delete_random(triples, k, seed):
    """Pick k triples uniformly at random; return (removed, kept)."""
    rng = random.Random(seed)
    n = len(triples)
    if k <= 0 or n == 0:
        return [], list(triples)
    k = min(k, n)
    idx = list(range(n))
    rng.shuffle(idx)
    pick = set(idx[:k])
    removed = [triples[i] for i in pick]
    kept = [t for i, t in enumerate(triples) if i not in pick]
    return removed, kept

def save_deleted_and_eval(
    original_triples,
    kept_triples,
    feature_tag,
    DB,
    top_k,                   # requested budget
    experiment_idx,
    MODEL,
    experiment_seed,
    test_path,
    valid_path,
):
    """
    Writes a dataset folder with kept train and removed.txt,
    then trains/evals and returns Test MRR.
    """
    out_dir = (
        SAVED_DATASETS_ROOT
        / DB
        / "delete"
        / feature_tag
        / MODEL
        / str(top_k)
        / str(experiment_seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_set = set(original_triples)
    kept_set = set(kept_triples)
    removed = orig_set - kept_set

    # Save kept and removed
    save_triples(list(kept_triples), str(out_dir / "train.txt"))
    (out_dir / "removed.txt").write_text(
        "\n".join("\t".join(x) for x in removed),
        encoding="utf-8",
    )

    # Copy val/test
    shutil.copy2(test_path, str(out_dir / "test.txt"))
    shutil.copy2(valid_path, str(out_dir / "valid.txt"))

    # Train/eval
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
            RUNS_ROOT / f"random_delete/{top_k}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}"
        ),
    )
    return res["Test"]["MRR"]

# ----------------- main -----------------
def main():
    # separate CSV for the random baseline
    rand_results_csv = RESULTS_ROOT / "all_random_result.csv"
    with rand_results_csv.open("w", encoding="utf-8") as f:
        f.write("DB,MODEL,attack,noise_ratio,requested_budget,actual_removed,exp_idx,seed,MRR\n")

    for DB in DBS:
        TRIPLES_PATH = f"./KGs/{DB}/train.txt"
        VALID_PATH = f"./KGs/{DB}/valid.txt"
        TEST_PATH  = f"./KGs/{DB}/test.txt"

        train_triples = load_triples(TRIPLES_PATH)
        n_train = len(train_triples)
        budgets = [max(1, int(n_train * p)) for p in PERCENTAGES]
        print(f"\n[DB={DB}] n_train={n_train}, budgets={budgets}")

        for MODEL in MODELS:
            print(f"  [MODEL={MODEL}]")
            for exp_idx, exp_seed in enumerate(EXPERIMENT_SEEDS):
                set_seeds(exp_seed)
                print(f"    [exp={exp_idx}] seed={exp_seed}")

                for noise_ratio, top_k in zip(PERCENTAGES, budgets):
                    # Random deletion
                    removed, kept = delete_random(train_triples, k=top_k, seed=exp_seed)
                    actual_removed = len(removed)

                    feature_tag = f"random_p{int(noise_ratio * 100):02d}"
                    mrr = save_deleted_and_eval(
                        original_triples=train_triples,
                        kept_triples=kept,
                        feature_tag=feature_tag,
                        DB=DB,
                        top_k=top_k,
                        experiment_idx=exp_idx,
                        MODEL=MODEL,
                        experiment_seed=exp_seed,
                        test_path=TEST_PATH,
                        valid_path=VALID_PATH,
                    )
                    print(
                        f"        RANDOM DEL MRR={mrr:.4f} | requested={top_k} | actual={actual_removed}"
                    )

                    with rand_results_csv.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{DB},{MODEL},random_del,{noise_ratio},{top_k},"
                            f"{actual_removed},{exp_idx},{exp_seed},{mrr}\n"
                        )

if __name__ == "__main__":
    main()
