import os
import shutil
import random
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

from executer_4del import run_dicee_eval
from utils import set_seeds, load_triples, save_triples  

KEMENY_SCRIPT = "./kemeny_attack.py"

# ---------------------

DBS = [ "UMLS", "KINSHIP" ]
BATCH_SIZE = "256"
LEARNING_RATE = "0.01"

# ---------------------

#DBS = [  "NELL-995-h100", "FB15k-237" ] 
#BATCH_SIZE = "512"
#LEARNING_RATE = "0.02"

# ---------------------

RECIPRIOCAL = "without_recipriocal"

# ---------------------

MODELS = [ "DistMult", "ComplEx", "DeCaL", "Keci", "Pykeen_MuRE", "Pykeen_RotatE" ]
PERCENTAGES = [0.02, 0.04, 0.08]
NUM_EXPERIMENTS = 2

NUM_EPOCHS = "100"
EMB_DIM = "32"
LOSS_FN = "BCELoss"
SCORING_TECH = "KvsAll"
OPTIM = "Adam"

# Kemeny params
KEMENY_R = 1e-14
KEMENY_JITTER = 1e-14

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
        "\n".join("\t".join(x) for x in removed),
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

def run_kemeny_attack(train_path, budget, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    out_removed = work_dir / "kemeny_removed.txt"
    out_poisoned = work_dir / "kemeny_poisoned_train.txt"
    scores_csv = work_dir / "kemeny_scores.csv"

    cmd = [
        "python",
        KEMENY_SCRIPT,
        "--input", str(train_path),
        "--budget", str(budget),
        "--r", str(KEMENY_R),
        "--jitter", str(KEMENY_JITTER),
        "--sep", "\t",
        "--out-removed", str(out_removed),
        "--out-poisoned", str(out_poisoned),
        "--scores-csv", str(scores_csv),
    ]
    subprocess.run(cmd, check=True)

    kept_triples = load_triples(str(out_poisoned))
    return kept_triples


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    # separate result files
    results_del_csv = RESULTS_ROOT / "all_result_delete.csv"
    results_add_csv = RESULTS_ROOT / "all_result_add.csv"

    with results_del_csv.open("w", encoding="utf-8") as f:
        f.write("DB,MODEL,attack,noise_ratio,count,exp_idx,seed,MRR\n")
    with results_add_csv.open("w", encoding="utf-8") as f:
        f.write("DB,MODEL,attack,noise_ratio,count,exp_idx,seed,MRR\n")

    for DB in DBS:
        TRIPLES_PATH = Path(f"./KGs/{DB}/train.txt")
        VALID_PATH = Path(f"./KGs/{DB}/valid.txt")
        TEST_PATH = Path(f"./KGs/{DB}/test.txt")

        train_triples = load_triples(str(TRIPLES_PATH))
        val_triples = load_triples(str(VALID_PATH))
        test_triples = load_triples(str(TEST_PATH))

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
                    print(f"      noise_ratio={noise_ratio:.3f}, budget={top_k}")

                    # ----- DELETION -----
                    kemeny_del_work = exp_work_root / f"kemeny_del_{top_k}"
                    kept_kemeny = run_kemeny_attack(
                        TRIPLES_PATH,
                        top_k,
                        kemeny_del_work,
                    )

                    # actual number of removed triples
                    removed_count = len(train_triples) - len(kept_kemeny)
                    removed_ratio = removed_count / float(n_train)
                    print(
                        f"        deletion: requested={top_k}, "
                        f"actual_removed={removed_count} ({removed_ratio:.4f})"
                    )

                    mrr_k_del = save_deleted_and_eval(
                        original_triples=train_triples,
                        kept_triples=kept_kemeny,
                        feature_tag=f"kemeny_nb_p{int(noise_ratio*100):02d}",
                        DB=DB,
                        top_k=removed_count,  # this isn't used inside, but OK
                        experiment_idx=exp_idx,
                        MODEL=MODEL,
                        experiment_seed=exp_seed,
                        test_path=str(TEST_PATH),
                        valid_path=str(VALID_PATH),
                    )
                    print(f"        Kemeny-no-backup DEL MRR={mrr_k_del:.4f}")

                    # log using ACTUAL removed count, not budget
                    with results_del_csv.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{DB},{MODEL},kemeny_no_backup_del,"
                            f"{noise_ratio},{removed_count},"
                            f"{exp_idx},{exp_seed},{mrr_k_del}\n"
                        )


if __name__ == "__main__":
    main()


