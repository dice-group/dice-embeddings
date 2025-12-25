import os
import shutil
import random
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from executer_4del import run_dicee_eval
from utils import set_seeds, load_triples, save_triples, visualize_results  
from config import (DBS, 
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
                    OPTIM 
                    )

KEMENY_SCRIPT = "./kg_poison_kemeny_fixed.py"

KEMENY_JITTER = 1e-8  # or 1e-10

SAVED_DATASETS_ROOT = Path(f"./saved_datasets/{RECIPRIOCAL}/")
SAVED_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

RUNS_ROOT = Path(f"./running_experiments/{RECIPRIOCAL}/")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(f"./final_results/{RECIPRIOCAL}/")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORTS_ROOT = Path(f"./reports/{RECIPRIOCAL}")
REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 12345
seed_src = random.Random(MASTER_SEED)
EXPERIMENT_SEEDS = [seed_src.randrange(2 ** 32) for _ in range(NUM_EXPERIMENTS)]

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

    save_triples(list(kept_triples), str(out_dir / "train.txt"))

    (out_dir / "removed.txt").write_text(
        "\n".join(["\t".join(x) for x in removed]),
        encoding="utf-8",
    )

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
            RUNS_ROOT / f"delete/{top_k}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}"
        ),
    )
    return res["Test"]["MRR"]


def save_added_and_eval(
    original_triples,
    added_triples,
    feature_tag,
    DB,
    top_k,
    experiment_idx,
    MODEL,
    experiment_seed,
    test_path,
    valid_path,
):

    out_dir = (
        SAVED_DATASETS_ROOT
        / DB
        / "add"
        / feature_tag
        / MODEL
        / str(top_k)
        / str(experiment_seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    new_train = list(original_triples) + list(added_triples)
    save_triples(new_train, str(out_dir / "train.txt"))

    (out_dir / "added.txt").write_text(
        "\n".join(["\t".join(x) for x in added_triples]),
        encoding="utf-8",
    )

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
            RUNS_ROOT
            / f"add/{top_k}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}"
        ),
    )
    return res["Test"]["MRR"]


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    del_results_csv = RESULTS_ROOT / "all_del_result.csv"

    with del_results_csv.open("w", encoding="utf-8") as f:
        f.write("DB,MODEL,attack,noise_ratio,budget,exp_idx,seed,MRR\n")

    for DB in DBS:
        TRIPLES_PATH = f"./KGs/{DB}/train.txt"
        VALID_PATH = f"./KGs/{DB}/valid.txt"
        TEST_PATH = f"./KGs/{DB}/test.txt"

        train_triples = load_triples(TRIPLES_PATH)
        val_triples = load_triples(VALID_PATH)
        test_triples = load_triples(TEST_PATH)

        n_train = len(train_triples)
        budgets = [max(1, int(n_train * p)) for p in PERCENTAGES]

        print(f"\n[DB={DB}] n_train={n_train}, budgets={budgets}")


        for MODEL in MODELS:
            print(f"  [MODEL={MODEL}]")

            for exp_idx, exp_seed in enumerate(EXPERIMENT_SEEDS):
                set_seeds(exp_seed)
                print(f"    [exp={exp_idx}] seed={exp_seed}")

                exp_work_root = RUNS_ROOT / "attack_work" / DB / MODEL / str(exp_seed)
                exp_work_root.mkdir(parents=True, exist_ok=True)

                for noise_ratio, top_k in zip(PERCENTAGES, budgets):

                    ....

                    with add_results_csv.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{DB},{MODEL},kemeny_add,{noise_ratio},{len(added_k)},"
                            f"{exp_idx},{exp_seed},{mrr_k_add}\n"
                        )


if __name__ == "__main__":
    main()
