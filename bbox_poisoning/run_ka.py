import os
import shutil
import random
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

from executer_4del import run_dicee_eval
from utils import set_seeds, load_triples, save_triples

KEMENY_SCRIPT = "./ka.py"

DBS = [ "UMLS", "KINSHIP" ]
BATCH_SIZE = "256"
LEARNING_RATE = "0.01"

# ---------------------

#DBS = [  "NELL-995-h100", "FB15k-237", "WN18RR" ] 
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
    count_used,            
    MODEL,
    experiment_seed,
    test_path,
    valid_path,
):
    out_dir = (
        SAVED_DATASETS_ROOT / DB / "delete" / feature_tag / MODEL / str(count_used) / str(experiment_seed)
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
            RUNS_ROOT / f"delete/{count_used}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}"
        ),
    )
    return res["Test"]["MRR"]


def save_added_and_eval(
    original_triples,
    added_triples,
    feature_tag,
    DB,
    count_used,
    MODEL,
    experiment_seed,
    test_path,
    valid_path,
):
    out_dir = (
        SAVED_DATASETS_ROOT / DB / "add" / feature_tag / MODEL / str(count_used) / str(experiment_seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    new_train = list(original_triples) + list(added_triples)
    save_triples(new_train, str(out_dir / "train.txt"))
    (out_dir / "added.txt").write_text(
        "\n".join("\t".join(x) for x in added_triples),
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
            RUNS_ROOT / f"add/{count_used}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}"
        ),
    )
    return res["Test"]["MRR"]


# ------------- Deletion -------------

def run_kemeny_delete(train_path, budget, work_dir):
    
    # Call kemeny_attack.py

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
    return kept_triples, scores_csv


# ------------- Pair scores (for addition) -------------

def precompute_kemeny_pair_scores(DB, train_path):

    # Run deletion script with budget=0 to store pair scores once per DB

    scores_root = RUNS_ROOT / "kemeny_scores" / DB
    scores_root.mkdir(parents=True, exist_ok=True)
    scores_csv = scores_root / "pair_scores.csv"

    if not scores_csv.exists():
        tmp_removed = scores_root / "tmp_removed.txt"
        tmp_poisoned = scores_root / "tmp_poisoned.txt"

        cmd = [
            "python",
            KEMENY_SCRIPT,
            "--input", str(train_path),
            "--budget", "0",
            "--r", str(KEMENY_R),
            "--jitter", str(KEMENY_JITTER),
            "--sep", "\t",
            "--out-removed", str(tmp_removed),
            "--out-poisoned", str(tmp_poisoned),
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
                pair_scores[(head, tail)] = float(score_str)
            except ValueError:
                continue
    return pair_scores


# ------------- Addition (Kemeny-hub) -------------

def run_kemeny_addition(train_triples, pair_scores, budget, seed, max_neighbors=50, rel_mode="freq"):
    """
    Kemeny-guided addition:
    - node_score from pair scores
    - non-edges among high-score nodes ranked by s(u)+s(v)
    - add 'budget' triples (relation chosen by rel_mode)
    """
    if budget <= 0:
        return [], list(train_triples)

    # node scores
    node_score = defaultdict(float)
    for (h, t), s in pair_scores.items():
        node_score[h] += s
        node_score[t] += s
    if not node_score:
        return [], list(train_triples)

    # existing undirected pairs
    existing_pairs = set()
    for h, r, t in train_triples:
        if h == t:
            continue
        a, b = (h, t) if h < t else (t, h)
        existing_pairs.add((a, b))

    nodes = sorted(node_score.keys(), key=lambda u: node_score[u], reverse=True)

    # candidate non-edges
    cand_scores = {}
    for u in nodes:
        local = 0
        for v in nodes:
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing_pairs or (a, b) in cand_scores:
                continue
            cand_scores[(a, b)] = node_score[u] + node_score[v]
            local += 1
            if local >= max_neighbors:
                break

    if not cand_scores:
        return [], list(train_triples)

    ranked_pairs = sorted(cand_scores.items(), key=lambda kv: kv[1], reverse=True)

    rng = random.Random(seed)
    rel_counts = Counter(r for (_, r, _) in train_triples)
    rels, weights = zip(*rel_counts.items()) if rel_counts else ([], [])

    # functional-ish relation prioritization
    if rel_mode == "functional":
        r_head_counts = defaultdict(lambda: defaultdict(int))
        for h, r, t in train_triples:
            r_head_counts[r][h] += 1
        rel_func_score = {r: sum(heads.values()) / max(1, len(heads)) for r, heads in r_head_counts.items()}
        rels_sorted = sorted(rel_func_score, key=lambda rr: rel_func_score[rr])  # lower avg tails per head first

        def pick_relation(u, v):
            for r in rels_sorted:
                return r
            return rng.choices(rels, weights=weights, k=1)[0] if rels else None
    else:
        def pick_relation(u, v):
            return rng.choices(rels, weights=weights, k=1)[0] if rels else None

    orig_set = set(train_triples)
    added = []

    for (a, b), _ in ranked_pairs:
        if len(added) >= budget:
            break
        r = pick_relation(a, b)
        if r is None:
            break
        triple = (a, r, b) if rng.random() < 0.5 else (b, r, a)
        if triple in orig_set or triple in added:
            continue
        added.append(triple)

    return added, list(train_triples) + added


# ------------- MAIN -------------

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    # separate results
    del_csv = RESULTS_ROOT / "deletion_results.csv"
    add_csv = RESULTS_ROOT / "addition_results.csv"
    header = "DB,MODEL,attack,noise_ratio,count,exp_idx,seed,MRR\n"
    for p in [del_csv, add_csv]:
        with p.open("w", encoding="utf-8") as f:
            f.write(header)

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

        # precompute pair scores once per DB (for addition)
        pair_scores = precompute_kemeny_pair_scores(DB, TRIPLES_PATH)

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
                    del_work = exp_work_root / f"kemeny_del_{top_k}"
                    kept_kemeny, _scores_csv = run_kemeny_delete(TRIPLES_PATH, top_k, del_work)
                    removed_count = len(train_triples) - len(kept_kemeny)
                    print(f"        deletion: requested={top_k}, actual_removed={removed_count} "
                          f"ratio={removed_count/n_train:.4f}")

                    mrr_del = save_deleted_and_eval(
                        original_triples=train_triples,
                        kept_triples=kept_kemeny,
                        feature_tag=f"kemeny_topk_p{int(noise_ratio*100):02d}",
                        DB=DB,
                        count_used=removed_count,
                        MODEL=MODEL,
                        experiment_seed=exp_seed,
                        test_path=str(TEST_PATH),
                        valid_path=str(VALID_PATH),
                    )
                    print(f"        DEL MRR={mrr_del:.4f}")

                    with del_csv.open("a", encoding="utf-8") as f:
                        f.write(f"{DB},{MODEL},kemeny_delete,{noise_ratio},{removed_count},"
                                f"{exp_idx},{exp_seed},{mrr_del}\n")

                    # ----- ADDITION -----
                    added_tr, poisoned = run_kemeny_addition(
                        train_triples=train_triples,
                        pair_scores=pair_scores,
                        budget=top_k,
                        seed=exp_seed,
                        max_neighbors=50,
                        rel_mode="functional",  # or "freq"
                    )
                    added_count = len(added_tr)
                    print(f"        addition: requested={top_k}, actual_added={added_count} "
                          f"ratio={added_count/n_train:.4f}")

                    mrr_add = save_added_and_eval(
                        original_triples=train_triples,
                        added_triples=added_tr,
                        feature_tag=f"kemeny_add_p{int(noise_ratio*100):02d}",
                        DB=DB,
                        count_used=added_count,
                        MODEL=MODEL,
                        experiment_seed=exp_seed,
                        test_path=str(TEST_PATH),
                        valid_path=str(VALID_PATH),
                    )
                    print(f"        ADD MRR={mrr_add:.4f}")

                    with add_csv.open("a", encoding="utf-8") as f:
                        f.write(f"{DB},{MODEL},kemeny_add,{noise_ratio},{added_count},"
                                f"{exp_idx},{exp_seed},{mrr_add}\n")


if __name__ == "__main__":
    main()
