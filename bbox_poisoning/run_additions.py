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

from typing import List, Tuple, Dict, Optional, Set, Literal
from collections import defaultdict
import networkx as nx

Triple = Tuple[str, str, str]

ORACLE_ROOT = Path(f"./saved_models/{RECIPRIOCAL}/")    
ORACLE_ROOT.mkdir(parents=True, exist_ok=True)

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

ORACLE_SEEDS = EXPERIMENT_SEEDS

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

    out_dir = SAVED_DATASETS_ROOT / DB / "add" / feature_tag / MODEL / str(top_k) / str(experiment_seed)
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
        path_to_store_single_run=str(RUNS_ROOT / f"add/{top_k}/{feature_tag}_{DB}_{MODEL}_{experiment_seed}")
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


import random
import torch

def score_triples(model, triples, entity_to_idx, relation_to_idx, triples_to_idx_with_maps,
                 batch_size=1, device=None):
    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device).eval()

    idx = triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx).to(device)

    logits_list = []
    for s in range(0, idx.size(0), batch_size):
        z = model.forward_triples(idx[s:s + batch_size]).reshape(-1)
        logits_list.append(z.detach().to("cpu"))
    logits = torch.cat(logits_list, dim=0)

    prob = torch.sigmoid(logits)  # (n_triples,)
    return prob


def generate_corrupted_triples(triples, entities, num_corruptions, seed=123):
    """
    Generate corrupted triples by randomly replacing head OR tail.
    Keeps relation fixed.
    """
    rng = random.Random(seed)
    n = len(triples)
    out = []

    for _ in range(num_corruptions):
        h, r, t = triples[rng.randrange(n)]
        if rng.randint(0, 1) == 0:  # corrupt head
            new_h = rng.choice(entities)
            while new_h == h and len(entities) > 1:
                new_h = rng.choice(entities)
            out.append((new_h, r, t))
        else:  # corrupt tail
            new_t = rng.choice(entities)
            while new_t == t and len(entities) > 1:
                new_t = rng.choice(entities)
            out.append((h, r, new_t))

    return out


def add_topk_high_scoring_corruptions(
    model,
    triples,
    entity_to_idx,
    relation_to_idx,
    triples_to_idx_with_maps,
    budget=100,               # k: how many high scoring corruptions to add
    num_corruptions=10000,    # how many corruptions to generate & score
    batch_size=1,
    seed=123,
    device=None,
):
    # entity pool from your mapping
    entities = list(entity_to_idx.keys())

    # 1) generate corrupted triples
    corrupted = generate_corrupted_triples(triples, entities, num_corruptions, seed=seed)

    # 2) score corrupted triples
    dist = score_triples(
        model=model,
        triples=corrupted,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        triples_to_idx_with_maps=triples_to_idx_with_maps,
        batch_size=batch_size,
        device=device,
    )

    # 3) pick top-k by score
    #budget = min(budget, dist.numel())
    #pick_idx = torch.topk(dist, k=budget, largest=False).indices.tolist()


    mu = dist.mean()   
    dist_final = torch.abs(dist - mu)
    pick_idx = torch.topk(dist_final, k=budget, largest=False).indices.tolist()


    



    top_corrupted = [corrupted[i] for i in pick_idx]

    # 4) append to original triples
    final_triples = list(triples) + top_corrupted
    return final_triples, top_corrupted, dist, pick_idx

def triples_to_idx_with_maps(
    triples,
    entity_to_idx,
    relation_to_idx,
):

    idx = torch.empty((len(triples), 3), dtype=torch.long)
    for i, (h, r, t) in enumerate(triples):
        try:
            idx[i, 0] = entity_to_idx[str(h)]
            idx[i, 1] = relation_to_idx[str(r)]
            idx[i, 2] = entity_to_idx[str(t)]
        except KeyError as e:
            raise KeyError(f"Label not found in model maps while indexing {triples[i]}: {e}")
    return idx

def _build_entity_digraph(triples: List[Triple]) -> nx.DiGraph:
    G = nx.DiGraph()
    for h, _, t in triples:
        G.add_edge(h, t)
    return G

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
                #res_add_wb_gax = []
                res_add_wb_fgsm = []
                res_simple_betweenness = []
                res_simple_closeness = []
                res_score_based = []

                for top_k in budgets:
                    
                    print(f"\n=== {DB} | {MODEL} | oracle_seed={oracle_seed} | budget={top_k} ===")
                    # -------- Random baseline --------
                    print("[Random] generating...")
                    _, corrupted = poison_random(train_triples, top_k, "random-one", exp_seed)

                    mrr_random = store_poisoned__adverserial(
                        train_triples, corrupted, "random", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_random.append(f"{mrr_random}")
                    """
                    # -------- Harmonic closeness  --------
                    print("[Harmonic] generating...")
                    add_hc = add_corrupted_by_harmonic_closeness(
                        train_triples, top_k, mode="both", top_k_nodes=500, undirected=True, avoid_existing_edge=True
                    )
                    mrr_hc = store_poisoned__adverserial(
                        train_triples, add_hc, "hc", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_add_hc.append(mrr_hc)

                    # -------- Harmonic closeness --------
                    print("[Betweenness] generating...")
                    add_bw = add_corrupted_by_edge_betweenness(
                        train_triples, top_k, mode="both", top_k_nodes=500, undirected=True, avoid_existing_edge=True
                    )
                    mrr_bw = store_poisoned__adverserial(
                        train_triples, add_bw, "bw", DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_simple_betweenness.append(mrr_bw)
                    """
                    # -------- Whitebox Cent+Loss --------
                    final_triples, added_triples, scores, pick_idx = add_topk_high_scoring_corruptions(
                        model=oracle.model,
                        triples=train_triples,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        triples_to_idx_with_maps=triples_to_idx_with_maps,
                        budget=top_k,
                        num_corruptions=5000,
                        batch_size=100,
                        seed=42,
                        device=device,
                    )
                    
                    mrr_score = store_poisoned__adverserial(
                        train_triples, added_triples, "score",
                        DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_score_based.append(mrr_score)


                    """
                    add_wb_centloss = add_corrupted_by_centrality_and_loss_forward(
                        train_triples,
                        model=oracle.model,
                        entity_to_idx=oracle.entity_to_idx,
                        relation_to_idx=oracle.relation_to_idx,
                        budget=top_k,
                        centrality="harmonic",     
                        undirected=True,
                        mode="both",
                        top_k_nodes=50,
                        avoid_existing_edge=True,
                        restrict_by_relation=False,
                        forbidden=forbidden,
                        batch_size=10000,
                        device=device,
                    )
                    print("corrupted suggestion ended!")
                    mrr_wb_cent = store_poisoned__adverserial(
                        train_triples, add_wb_centloss, "wb_centloss_forward",
                        DB, top_k, exp_idx, MODEL, exp_seed, TEST_PATH, VALID_PATH
                    )
                    res_add_wb_centloss.append(mrr_wb_cent)
                    """
                    """
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
                    """
                    # ----- quick length checks report -----
                    lengths_map = {
                        "Random": len(res_random),
                        "Score-Based": len(res_score_based),
                        #"Cent+Loss": len(res_add_wb_centloss),
                        #"GlobalArgmax": len(res_add_wb_gax),
                        #"FGSM": len(res_add_wb_fgsm),
                        #"Simple Closeness": len(res_add_hc),
                        #"Simple Betweenness": len(res_simple_betweenness),
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
                    ("Score-Based", res_score_based),
                    #("Cent+Loss", res_add_wb_centloss),
                    #("GlobalArgmax", res_add_wb_gax),
                    #("FGSM", res_add_wb_fgsm),
                    #("Simple Closeness", res_add_hc),
                    #("Simple Betweenness", res_simple_betweenness),
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
