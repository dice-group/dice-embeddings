# blackbox_poison_pipeline.py
# Black-box KG poisoning via proxy training + active learning (query budget),
# then addition/deletion selection using the learned proxy.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set, Callable, Literal
from collections import defaultdict
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx

# Your project utilities
from dicee import KGE
from executer import run_dicee_eval
from utils import set_seeds, load_triples, save_triples

Triple  = Tuple[str, str, str]
ScoreFn = Callable[[List[Triple]], torch.Tensor]   # returns probabilities (or logits)

# =============================================================================
# 1) Loading µ-embeddings from CSVs
# =============================================================================

def load_embeddings(entity_csv: str, relation_csv: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    CSV format: first column = name (entity / relation label), remaining columns = float dims.
    Example row: anatomical_abnormality, -0.76, 0.70, ...
    """
    ent_df = pd.read_csv(entity_csv, index_col=0)
    rel_df = pd.read_csv(relation_csv, index_col=0)

    ent_df.index = ent_df.index.map(lambda x: str(x).strip())
    rel_df.index = rel_df.index.map(lambda x: str(x).strip())

    entity_emb   = {name: torch.tensor(row.values, dtype=torch.float32).contiguous()
                    for name, row in ent_df.iterrows()}
    relation_emb = {name: torch.tensor(row.values, dtype=torch.float32).contiguous()
                    for name, row in rel_df.iterrows()}

    e_dims = {v.numel() for v in entity_emb.values()}
    r_dims = {v.numel() for v in relation_emb.values()}
    if len(e_dims) != 1 or len(r_dims) != 1:
        raise ValueError(f"Inconsistent embedding dims. Entities dims={e_dims}, Relations dims={r_dims}")
    return entity_emb, relation_emb

@dataclass
class MuEmbeddings:
    ent_vec: Dict[str, torch.Tensor]   # name -> (d_e,)
    rel_vec: Dict[str, torch.Tensor]   # name -> (d_r,)

    @property
    def d_in(self) -> int:
        e_dim = next(iter(self.ent_vec.values())).numel()
        r_dim = next(iter(self.rel_vec.values())).numel()
        return e_dim + r_dim + e_dim

    def encode_triples(self, triples: List[Triple]) -> torch.Tensor:
        """Return features X: (N, d_in) as [mu(h); mu(r); mu(t)]. Raises if OOV."""
        e_dim = next(iter(self.ent_vec.values())).numel()
        r_dim = next(iter(self.rel_vec.values())).numel()
        X = torch.empty((len(triples), self.d_in), dtype=torch.float32)
        for i, (h, r, t) in enumerate(triples):
            try:
                eh = self.ent_vec[h]; er = self.rel_vec[r]; et = self.ent_vec[t]
            except KeyError as e:
                raise KeyError(f"OOV in µ-embeddings for triple {triples[i]}: {e}")
            X[i, :e_dim]              = eh
            X[i, e_dim:e_dim + r_dim] = er
            X[i, e_dim + r_dim:]      = et
        return X

# =============================================================================
# 2) Oracle scorer (query-only black box)
# =============================================================================

def triples_to_idx_with_oracle(
    triples: List[Triple],
    entity_to_idx: Dict[str, int],
    relation_to_idx: Dict[str, int],
) -> torch.LongTensor:
    idx = torch.empty((len(triples), 3), dtype=torch.long)
    for i, (h, r, t) in enumerate(triples):
        try:
            idx[i, 0] = entity_to_idx[h]
            idx[i, 1] = relation_to_idx[r]
            idx[i, 2] = entity_to_idx[t]
        except KeyError as e:
            raise KeyError(f"OOV label while indexing triple {triples[i]}: {e}")
    return idx

class BudgetedOracle:
    """
    Wraps an oracle to provide a black-box score function with a hard query budget.
    Uses forward_triples -> probabilities. No gradients / no internals used.
    """
    def __init__(self, oracle: KGE, budget: int, logits: bool = False):
        self.oracle    = oracle
        self.remaining = budget
        self.logits    = logits
        self.device    = (next(oracle.model.parameters()).device
                          if any(True for _ in oracle.model.parameters())
                          else torch.device("cpu"))
        self.oracle.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, triples: List[Triple]) -> torch.Tensor:
        if len(triples) > self.remaining:
            raise RuntimeError(f"Query budget exceeded: need {len(triples)}, have {self.remaining} left.")
        self.remaining -= len(triples)
        idx = triples_to_idx_with_oracle(triples, self.oracle.entity_to_idx, self.oracle.relation_to_idx).to(self.device)
        z   = self.oracle.model.forward_triples(idx).reshape(-1).detach().cpu()
        return z if self.logits else torch.sigmoid(z)

# =============================================================================
# 3) Proxy model π + informativeness for active learning
# =============================================================================

class ProxyMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)  # logits

@dataclass
class ActiveConfig:
    q_budget: int
    init_alpha: float = 0.05
    negs_per_pos: int = 1
    n_perturb: int = 8
    eps_perturb: float = 1e-2
    pool_per_iter: int = 4096
    train_epochs_per_iter: int = 1
    batch_size: int = 1024
    lr: float = 1e-3
    device: Optional[torch.device] = None
    logits_from_api: bool = False

def build_digraph(triples: List[Triple]) -> nx.DiGraph:
    G = nx.DiGraph()
    for h, _, t in triples:
        G.add_edge(h, t)
    return G

def neg_sample_uniform(triples: List[Triple], num_negs_per_pos: int = 1) -> List[Triple]:
    """Uniform head/tail corruption negatives within observed vocab (fast & simple)."""
    ents = sorted({x for h, _, t in triples for x in (h, t)})
    S    = set(triples)
    rng  = random.Random(13)
    out  = []
    for (h, r, t) in triples:
        for _ in range(num_negs_per_pos):
            if rng.random() < 0.5:
                h2 = rng.choice(ents)
                if (h2, r, t) not in S:
                    out.append((h2, r, t))
            else:
                t2 = rng.choice(ents)
                if (h, r, t2) not in S:
                    out.append((h, r, t2))
    return out

class GemmaProxy:
    """
    Train proxy π on features [µ(h); µ(r); µ(t)], with labels from a black-box φ,
    spending at most q_budget queries (active learning via parameter-perturbation informativeness).
    """
    def __init__(self, mu: MuEmbeddings, cfg: ActiveConfig):
        self.mu     = mu
        self.cfg    = cfg
        self.device = cfg.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model  = ProxyMLP(mu.d_in).to(self.device)

    def _query_labels(self, query_fn: ScoreFn, triples: List[Triple]) -> List[float]:
        out = query_fn(triples).detach().cpu().view(-1)
        return torch.sigmoid(out).tolist() if self.cfg.logits_from_api else out.tolist()

    @torch.no_grad()
    def _informativeness(self, X: torch.Tensor) -> torch.Tensor:
        """
        i(x) = E[(π_{θ*}(x) - π_{θ}(x))^2], θ* = θ + ε * rand{-1,0,1}
        """
        self.model.eval()
        base   = torch.sigmoid(self.model(X.to(self.device))).cpu()  # (N,)
        params = [p for p in self.model.parameters() if p.requires_grad]
        flat   = torch.cat([p.detach().flatten().cpu() for p in params], 0)
        shapes = [p.shape for p in params]
        sizes  = [p.numel() for p in params]

        vals = torch.zeros_like(base)
        eps  = self.cfg.eps_perturb
        N    = self.cfg.n_perturb
        for _ in range(N):
            noise     = torch.randint(-1, 2, (flat.numel(),), dtype=torch.int8).to(torch.float32) * eps
            flat_pert = flat + noise
            # load perturbed
            pos = 0; k = 0
            with torch.no_grad():
                for p in self.model.parameters():
                    if not p.requires_grad: continue
                    n = sizes[k]
                    p.copy_(flat_pert[pos:pos+n].view(shapes[k]).to(p.device))
                    pos += n; k += 1
                out = torch.sigmoid(self.model(X.to(self.device))).cpu()
            vals += (out - base).pow(2)

        vals /= max(1, N)
        # restore original
        pos = 0; k = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if not p.requires_grad: continue
                n = sizes[k]
                p.copy_(flat[pos:pos+n].view(shapes[k]).to(p.device))
                pos += n; k += 1
        return vals

    def fit_active(self, H_pos: List[Triple], query_fn: ScoreFn, rng: Optional[random.Random] = None) -> None:
        rng  = rng or random.Random(0)
        cfg  = self.cfg
        dev  = self.device

        H_neg    = neg_sample_uniform(H_pos, num_negs_per_pos=cfg.negs_per_pos)
        n_pos    = max(1, int(cfg.init_alpha * len(H_pos)))
        n_neg    = max(1, int(cfg.init_alpha * len(H_neg)))
        seed_pos = rng.sample(H_pos, min(n_pos, len(H_pos)))
        seed_neg = rng.sample(H_neg, min(n_neg, len(H_neg)))
        D        = seed_pos + seed_neg
        y        = self._query_labels(query_fn, D)

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        bce = nn.BCEWithLogitsLoss()
        pool = list(set(H_pos + H_neg) - set(D))
        rng.shuffle(pool)

        while len(D) < cfg.q_budget and pool:
            # train briefly
            self.model.train()
            X = self.mu.encode_triples(D).to(dev)
            Y = torch.tensor(y, dtype=torch.float32, device=dev)
            for _ in range(cfg.train_epochs_per_iter):
                for s in range(0, len(D), cfg.batch_size):
                    xb, yb = X[s:s+cfg.batch_size], Y[s:s+cfg.batch_size]
                    opt.zero_grad(); logits = self.model(xb); loss = bce(logits, yb)
                    loss.backward(); opt.step()

            # pick next by informativeness
            sub  = pool[:cfg.pool_per_iter] if len(pool) > cfg.pool_per_iter else pool
            if not sub: break
            Xsub = self.mu.encode_triples(sub)
            info = self._informativeness(Xsub)
            j    = int(torch.argmax(info).item())
            xstar = sub[j]
            ystar = float(self._query_labels(query_fn, [xstar])[0])

            D.append(xstar); y.append(ystar)
            pool.pop(j)
            if rng.random() < 0.1:
                rng.shuffle(pool)

        # final small finetune
        self.model.train()
        X = self.mu.encode_triples(D).to(dev)
        Y = torch.tensor(y, dtype=torch.float32, device=dev)
        for _ in range(3):
            for s in range(0, len(D), cfg.batch_size):
                xb, yb = X[s:s+cfg.batch_size], Y[s:s+cfg.batch_size]
                opt.zero_grad(); logits = self.model(xb); loss = bce(logits, yb)
                loss.backward(); opt.step()
        self.model.eval()

    @torch.no_grad()
    def score_triples(self, triples: List[Triple], batch_size: int = 8192) -> torch.Tensor:
        probs = []
        for s in range(0, len(triples), batch_size):
            X = self.mu.encode_triples(triples[s:s+batch_size]).to(self.device)
            z = self.model(X); p = torch.sigmoid(z).detach().cpu()
            probs.append(p)
        return torch.cat(probs, 0)

# =============================================================================
# 4) Candidate generation + edit selection with proxy
# =============================================================================

def harmonic_nodes(triples: List[Triple], undirected: bool = True) -> List[str]:
    Gd = build_digraph(triples)
    G  = Gd.to_undirected() if undirected else Gd
    node_h = nx.harmonic_centrality(G)
    return [n for n, _ in sorted(node_h.items(), key=lambda kv: kv[1], reverse=True)]

def propose_add_candidates(
    triples: List[Triple],
    *,
    mode: Literal["corrupt","centrality"] = "corrupt",
    per_pos_negs: int = 20,
    undirected_centrality: bool = True,
    avoid_existing_edge: bool = True,
    restrict_by_relation: bool = False,
) -> List[Triple]:
    S = set(triples)
    ht_set = {(h, t) for h, _, t in triples}
    if mode == "corrupt":
        return neg_sample_uniform(triples, num_negs_per_pos=per_pos_negs)

    ents_sorted = harmonic_nodes(triples, undirected=undirected_centrality)
    heads_by_rel, tails_by_rel = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        heads_by_rel[r].add(h)
        tails_by_rel[r].add(t)

    cands, seen = [], set()
    for h, r, t in triples:
        # head replacements
        for n in ents_sorted[:256]:
            if n == h or n == t: continue
            cand = (n, r, t)
            if cand in S or cand in seen: continue
            if avoid_existing_edge and ((n, t) in ht_set): continue
            if restrict_by_relation and (n not in heads_by_rel[r]): continue
            cands.append(cand); seen.add(cand)
        # tail replacements
        for n in ents_sorted[:256]:
            if n == h or n == t: continue
            cand = (h, r, n)
            if cand in S or cand in seen: continue
            if avoid_existing_edge and ((h, n) in ht_set): continue
            if restrict_by_relation and (n not in tails_by_rel[r]): continue
            cands.append(cand); seen.add(cand)
    return cands

@torch.no_grad()
def select_additions_with_proxy(
    G_train: List[Triple],
    proxy: GemmaProxy,
    *,
    budget: int,
    candidate_mode: Literal["corrupt","centrality"] = "corrupt",
    per_pos_negs: int = 20,
    avoid_existing_edge: bool = True,
) -> List[Triple]:
    cands = propose_add_candidates(
        G_train,
        mode=candidate_mode,
        per_pos_negs=per_pos_negs,
        avoid_existing_edge=avoid_existing_edge,
    )
    if not cands or budget <= 0:
        return []
    P = proxy.score_triples(cands)     # probabilities
    k = min(budget, len(cands))
    idx = torch.topk(-P, k=k, largest=True).indices.tolist()
    return [cands[i] for i in idx]

@torch.no_grad()
def select_deletions_with_proxy(
    G_train: List[Triple],
    proxy: GemmaProxy,
    *,
    budget: int,
) -> List[Triple]:
    if budget <= 0:
        return []
    P = proxy.score_triples(G_train)
    k = min(budget, len(G_train))
    idx = torch.topk(P, k=k, largest=True).indices.tolist()
    return [G_train[i] for i in idx]

# =============================================================================
# 5) Runner: train proxy, craft edits, save splits, evaluate
# =============================================================================

@dataclass
class RunnerConfig:
    DB: str
    MODEL: str
    ORACLE_PATH: str
    ENTITY_CSV: str
    RELATION_CSV: str
    ADD_PCTS: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
    DEL_PCTS: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
    NUM_EPOCHS: str = "100"
    BATCH_SIZE: str = "256"
    LR: str = "0.01"
    EMB_DIM: str = "32"
    LOSS_FN: str = "BCELoss"
    SCORING_TECH: str = "KvsAll"
    OPTIM: str = "Adam"
    SAVED_DATASETS_ROOT: str = "./saved_datasets"
    RUNS_ROOT: str = "./running_experiments"
    RESULTS_ROOT: str = "./final_results/blackbox"

def run_blackbox_poison(cfg: RunnerConfig, master_seed: int = 12345):
    # Load splits
    TR = f"./KGs/{cfg.DB}/train.txt"
    VA = f"./KGs/{cfg.DB}/valid.txt"
    TE = f"./KGs/{cfg.DB}/test.txt"
    train_triples = load_triples(TR)
    val_triples   = load_triples(VA)
    test_triples  = load_triples(TE)

    rng      = random.Random(master_seed)
    exp_seed = rng.randrange(2**32)
    set_seeds(exp_seed)

    QUERY_BUDGET = int(len(train_triples) * 0.1)

    # Load µ-embeddings
    ent_mu, rel_mu = load_embeddings(cfg.ENTITY_CSV, cfg.RELATION_CSV)
    mu = MuEmbeddings(ent_vec=ent_mu, rel_vec=rel_mu)

    # Load black-box oracle (query-only)
    oracle = KGE(path=cfg.ORACLE_PATH)
    bb     = BudgetedOracle(oracle, budget=QUERY_BUDGET, logits=False)

    # Train proxy with active learning

    print(f"Training proxy with q_budget={QUERY_BUDGET}, init_alpha=0.05, negs_per_pos=1")
    proxy = GemmaProxy(
        mu=mu,
        cfg=ActiveConfig(
            q_budget=QUERY_BUDGET,
            init_alpha=0.05,
            negs_per_pos=1,
            n_perturb=8,
            eps_perturb=1e-2,
            pool_per_iter=4096,
            train_epochs_per_iter=1,
            batch_size=256,
            lr=1e-3,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            logits_from_api=False,
        )
    )
    proxy.fit_active(H_pos=train_triples, query_fn=bb)
    print(f"Proxy training done. Queries used: {QUERY_BUDGET - bb.remaining}/{QUERY_BUDGET}")

    # Budgets
    add_ks = [max(1, int(len(train_triples) * p)) for p in cfg.ADD_PCTS]
    del_ks = [max(1, int(len(train_triples) * p)) for p in cfg.DEL_PCTS]

    # Output dirs
    SAVED = Path(cfg.SAVED_DATASETS_ROOT)
    RUNS  = Path(cfg.RUNS_ROOT)
    RES   = Path(cfg.RESULTS_ROOT) / cfg.DB / cfg.MODEL
    RES.mkdir(parents=True, exist_ok=True)

    # ----------------------- ADDITION -----------------------
    add_results = []
    for top_k in add_ks:
        additions = select_additions_with_proxy(
            train_triples, proxy, budget=top_k,
            candidate_mode="corrupt", per_pos_negs=20, avoid_existing_edge=True
        )
        poisoned = train_triples + additions
        random.shuffle(poisoned)

        out_dir = SAVED / cfg.DB / "blackbox" / "add" / cfg.MODEL / str(top_k)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_triples(poisoned, str(out_dir / "train.txt"))
        Path(out_dir / "valid.txt").write_text(Path(VA).read_text())
        Path(out_dir / "test.txt").write_text(Path(TE).read_text())

        res = run_dicee_eval(
            dataset_folder=str(out_dir),
            model=cfg.MODEL,
            num_epochs=str(cfg.NUM_EPOCHS),
            batch_size=str(cfg.BATCH_SIZE),
            learning_rate=str(cfg.LR),
            embedding_dim=str(cfg.EMB_DIM),
            loss_function=str(cfg.LOSS_FN),
            seed=exp_seed,
            scoring_technique=str(cfg.SCORING_TECH),
            optim=str(cfg.OPTIM),
            path_to_store_single_run=str(RUNS / f"blackbox_add_{cfg.DB}_{cfg.MODEL}_{top_k}")
        )
        add_results.append(res["Test"]["MRR"])

    # write ADD report (separate file)
    out_csv_add = RES / f"results-blackbox-add-{cfg.DB}-{cfg.MODEL}-seed-{exp_seed}.csv"
    with open(out_csv_add, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["Add Ratios"] + list(map(str, cfg.ADD_PCTS)))
        w.writerow(["Add #Triples"] + [str(k) for k in add_ks])
        w.writerow(["Add MRR"] + [str(x) for x in add_results])

    # ----------------------- DELETION -----------------------
    del_results = []
    for top_k in del_ks:
        deletions = select_deletions_with_proxy(train_triples, proxy, budget=top_k)
        kept      = [t for t in train_triples if t not in set(deletions)]

        out_dir = SAVED / cfg.DB / "blackbox" / "delete" / cfg.MODEL / str(top_k)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_triples(kept, str(out_dir / "train.txt"))
        (out_dir / "removed.txt").write_text("\n".join(["\t".join(x) for x in deletions]))
        Path(out_dir / "valid.txt").write_text(Path(VA).read_text())
        Path(out_dir / "test.txt").write_text(Path(TE).read_text())

        res = run_dicee_eval(
            dataset_folder=str(out_dir),
            model=cfg.MODEL,
            num_epochs=str(cfg.NUM_EPOCHS),
            batch_size=str(cfg.BATCH_SIZE),
            learning_rate=str(cfg.LR),
            embedding_dim=str(cfg.EMB_DIM),
            loss_function=str(cfg.LOSS_FN),
            seed=exp_seed,
            scoring_technique=str(cfg.SCORING_TECH),
            optim=str(cfg.OPTIM),
            path_to_store_single_run=str(RUNS / f"blackbox_delete_{cfg.DB}_{cfg.MODEL}_{top_k}")
        )
        del_results.append(res["Test"]["MRR"])

    # write DELETE report (separate file)
    out_csv_del = RES / f"results-blackbox-del-{cfg.DB}-{cfg.MODEL}-seed-{exp_seed}.csv"
    with open(out_csv_del, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["Del Ratios"] + list(map(str, cfg.DEL_PCTS)))
        w.writerow(["Del #Triples"] + [str(k) for k in del_ks])
        w.writerow(["Del MRR"] + [str(x) for x in del_results])
