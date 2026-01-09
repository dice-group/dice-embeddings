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
from blackbox_attack import (
    score_based_deletion       
)
from config import (DBS, 
                    MODELS, 
                    RECIPRIOCAL, 
                    PERCENTAGES, 
                    BATCH_SIZE, 
                    LEARNING_RATE, 
                    NUM_EXPERIMENTS, 
                    NUM_EPOCHS, 
                    EMB_DIM, 
                    SCORING_TECH, 
                    OPTIM ,
                    QUANTIES
                    )

from typing import List, Tuple, Dict, Optional, Set, Literal
from collections import defaultdict


ORACLE_ROOT = Path(f"./saved_models/{RECIPRIOCAL}/")    
#ORACLE_ROOT.mkdir(parents=True, exist_ok=True)

SAVED_DATASETS_ROOT = Path(f"./saved_perturbed_datasets/{RECIPRIOCAL}/")
SAVED_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)


MASTER_SEED = 12345
seed_src = random.Random(MASTER_SEED)
EXPERIMENT_SEEDS = [seed_src.randrange(2 ** 32) for _ in range(NUM_EXPERIMENTS)]

ORACLE_SEEDS = EXPERIMENT_SEEDS

def save_perturbed_dataset(
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
    q_lo,
    q_hi
):
    out_dir = SAVED_DATASETS_ROOT / DB / "delete" / feature_tag / f"{q_lo}-{q_hi}" / MODEL / str(top_k) / str(experiment_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    perturbed = set(original_triples) - set(kept_triples)
    save_triples(list(kept_triples), str(out_dir / "train.txt"))
    (out_dir / "selected_triples_before_perturbation.txt").write_text("\n".join(["\t".join(x) for x in perturbed]))

    shutil.copy2(test_path, str(out_dir / "test.txt"))
    shutil.copy2(valid_path, str(out_dir / "valid.txt"))

    return "saved!"


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


def perturb_random(triples, k, seed):
    rng = random.Random(seed)
    n = len(triples)
    if k <= 0 or n == 0:
        return list(triples)

    k = min(k, n)

    heads = [h for h, r, t in triples]
    rels  = [r for h, r, t in triples]
    tails = [t for h, r, t in triples]

    idx = list(range(n))
    rng.shuffle(idx)
    pick = set(idx[:k])

    out = []
    for i, (h, r, t) in enumerate(triples):
        if i not in pick:
            out.append((h, r, t))
            continue

        which = rng.randint(0, 2)

        if which == 0:
            new_h = rng.choice(heads)
            while new_h == h and len(set(heads)) > 1:
                new_h = rng.choice(heads)
            out.append((new_h, r, t))

        elif which == 1:
            new_r = rng.choice(rels)
            while new_r == r and len(set(rels)) > 1:
                new_r = rng.choice(rels)
            out.append((h, new_r, t))

        else:
            new_t = rng.choice(tails)
            while new_t == t and len(set(tails)) > 1:
                new_t = rng.choice(tails)
            out.append((h, r, new_t))

    return [], out


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

    for DB in DBS:
        TRIPLES_PATH = f"../KGs/{DB}/train.txt"
        VALID_PATH = f"../KGs/{DB}/valid.txt"
        TEST_PATH = f"../KGs/{DB}/test.txt"

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
                res_score = []


                for top_k in budgets:
                    print(f"\n=== DELETE | {DB} | {MODEL} | oracle_seed={oracle_seed} | budget={top_k} ===")
                    
                    
                    # -------- Random deletion --------
                    print("[Random Delete...")
                    _, kept_rand = perturb_random(train_triples, top_k, seed=exp_seed)

                    save_perturbed_dataset(
                        train_triples, kept_rand, "random", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH, 0, 0
                    )
                    

                    # -------- Score based deletion --------
                    print("Score Delete...")
                    
                    

                    for i, (q_lo, q_hi) in enumerate(QUANTIES):
                        print(f"bin {i}: [{q_lo:.2f}, {q_hi:.2f}]")

                        kept_cl = score_based_deletion(
                            train_triples,
                            model=oracle.model,
                            entity_to_idx=oracle.entity_to_idx,
                            relation_to_idx=oracle.relation_to_idx,
                            budget=top_k,
                            batch_size=1000,
                            device=device,
                            model_name=MODEL,
                            db_name=DB,
                            q1=q_lo,
                            q2=q_hi
                        )

                        save_perturbed_dataset(
                            train_triples, kept_cl, "score", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH, q_lo, q_hi
                        )
                    

if __name__ == "__main__":
    main()
