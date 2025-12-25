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


def run_kemeny_attack(train_path: str, budget: int, work_dir: Path, k: int = 2):

    work_dir.mkdir(parents=True, exist_ok=True)
    out_to_delete = work_dir / "kemeny_to_delete.txt"
    out_poisoned = work_dir / "kemeny_poisoned_train.txt"

    cmd = [
        "python",
        KEMENY_SCRIPT,
        "--input", train_path,
        "--budget", str(budget),
        "--k", str(k),
        "--unit", "triple",
        "--gate", "soft",
        "--jitter", str(KEMENY_JITTER),
        "--out", str(out_to_delete),
        "--poisoned-out", str(out_poisoned),
    ]
    subprocess.run(cmd, check=True)

    kept_triples = load_triples(str(out_poisoned))
    return kept_triples



def precompute_kemeny_pair_scores(DB: str, train_path: str):
    """
    Call the Kemeny script once for this DB to get per-pair filtered scores.
    It writes a scores CSV we can reuse across all models / seeds / budgets.

    Requires kg_poison_kemeny_fixed.py to support --scores-csv.
    """
    scores_root = RUNS_ROOT / "kemeny_scores" / DB
    scores_root.mkdir(parents=True, exist_ok=True)
    scores_csv = scores_root / "pair_scores.csv"

    if not scores_csv.exists():
        tmp_out = scores_root / "tmp_to_delete.txt"
        tmp_poisoned = scores_root / "tmp_poisoned.txt"

        cmd = [
            "python",
            KEMENY_SCRIPT,
            "--input", train_path,
            "--budget", "0",         
            "--k", "1",
            "--unit", "pair",
            "--gate", "none",
            "--jitter", str(KEMENY_JITTER),
            "--out", str(tmp_out),
            "--poisoned-out", str(tmp_poisoned),
            "--scores-csv", str(scores_csv),
        ]
        print(f"[Kemeny] Precomputing pair scores for {DB}...")
        subprocess.run(cmd, check=True)

    pair_scores = {}
    with scores_csv.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            head, tail, score_str = parts[0], parts[1], parts[2]
            try:
                score = float(score_str)
            except ValueError:
                continue
            pair_scores[(head, tail)] = score

    return pair_scores


def run_kemeny_addition(
    train_triples,
    pair_scores,
    budget: int,
    seed: int,
    max_neighbors: int = 50,
):

    if budget <= 0:
        return [], list(train_triples)

    node_score = defaultdict(float)
    for (h, t), s in pair_scores.items():
        node_score[h] += s
        node_score[t] += s

    if not node_score:
        return [], list(train_triples)

    existing_pairs = set()
    for h, r, t in train_triples:
        if h == t:
            continue
        a, b = (h, t) if h < t else (t, h)
        existing_pairs.add((a, b))

    nodes = list(node_score.keys())
    nodes_sorted = sorted(nodes, key=lambda u: node_score[u], reverse=True)

    cand_scores = {}   
    for u in nodes_sorted:
        local = 0
        for v in nodes_sorted:
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing_pairs:
                continue
            score = node_score[u] + node_score[v]
            prev = cand_scores.get((a, b))
            if prev is None or score > prev:
                cand_scores[(a, b)] = score
            local += 1
            if local >= max_neighbors:
                break

    if not cand_scores:
        return [], list(train_triples)

    sorted_pairs = sorted(cand_scores.items(), key=lambda kv: kv[1], reverse=True)

    rel_counts = Counter(r for (h, r, t) in train_triples)
    if not rel_counts:
        return [], list(train_triples)

    rels, weights = zip(*rel_counts.items())
    rng = random.Random(seed)
    orig_set = set(train_triples)
    added = []

    for (a, b), _score in sorted_pairs:
        if len(added) >= budget:
            break
        r = rng.choices(rels, weights=weights, k=1)[0]
        if rng.random() < 0.5:
            triple = (a, r, b)
        else:
            triple = (b, r, a)
        if triple in orig_set or triple in added:
            continue
        added.append(triple)

    poisoned = list(train_triples) + added
    return added, poisoned


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
    add_results_csv = RESULTS_ROOT / "all_add_result.csv"

    with del_results_csv.open("w", encoding="utf-8") as f:
        f.write("DB,MODEL,attack,noise_ratio,budget,exp_idx,seed,MRR\n")

    with add_results_csv.open("w", encoding="utf-8") as f:
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

        pair_scores = precompute_kemeny_pair_scores(DB, TRIPLES_PATH)

        for MODEL in MODELS:
            print(f"  [MODEL={MODEL}]")

            for exp_idx, exp_seed in enumerate(EXPERIMENT_SEEDS):
                set_seeds(exp_seed)
                print(f"    [exp={exp_idx}] seed={exp_seed}")

                exp_work_root = RUNS_ROOT / "attack_work" / DB / MODEL / str(exp_seed)
                exp_work_root.mkdir(parents=True, exist_ok=True)

                for noise_ratio, top_k in zip(PERCENTAGES, budgets):
                    """ 
                    print(f"      noise_ratio={noise_ratio:.3f}, budget={top_k}")

                    kemeny_work = exp_work_root / f"kemeny_del_{top_k}"
                    kept_kemeny = run_kemeny_attack(
                        TRIPLES_PATH,
                        top_k,
                        kemeny_work,
                        k=2,
                    )
                    mrr_k_del = save_deleted_and_eval(
                        original_triples=train_triples,
                        kept_triples=kept_kemeny,
                        feature_tag=f"kemeny_del_p{int(noise_ratio*100):02d}",
                        DB=DB,
                        top_k=top_k,
                        experiment_idx=exp_idx,
                        MODEL=MODEL,
                        experiment_seed=exp_seed,
                        test_path=TEST_PATH,
                        valid_path=VALID_PATH,
                    )
                    print(f"        Kemeny-DEL MRR={mrr_k_del:.4f}")

                    with del_results_csv.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{DB},{MODEL},kemeny_del,{noise_ratio},{len(train_triples) - len(kept_kemeny)},"
                            f"{exp_idx},{exp_seed},{mrr_k_del}\n"
                        ) 
                    """

                    added_k, poisoned_k = run_kemeny_addition(
                        train_triples=train_triples,
                        pair_scores=pair_scores,
                        budget=top_k,
                        seed=exp_seed,
                    )
                    mrr_k_add = save_added_and_eval(
                        original_triples=train_triples,
                        added_triples=added_k,
                        feature_tag=f"kemeny_add_p{int(noise_ratio*100):02d}",
                        DB=DB,
                        top_k=top_k,
                        experiment_idx=exp_idx,
                        MODEL=MODEL,
                        experiment_seed=exp_seed,
                        test_path=TEST_PATH,
                        valid_path=VALID_PATH,
                    )
                    print(f"        Kemeny-ADD MRR={mrr_k_add:.4f}")

                    with add_results_csv.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{DB},{MODEL},kemeny_add,{noise_ratio},{len(added_k)},"
                            f"{exp_idx},{exp_seed},{mrr_k_add}\n"
                        )


if __name__ == "__main__":
    main()
