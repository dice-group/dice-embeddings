"""Compare entity-featurization schemes for inductive KGE (CPU-friendly).

Trains the *same* InductiveKGModel under two schemes for the anonymous
entity slots, on the *same* data and seed, so differences are attributable
to the scheme alone:

    learned  – current approach: [Z_i] are rows of a learned nn.Embedding.
    runtime  – no pool: a fresh random vector per distinct entity, per forward
               pass (pure RNI; Abboud et al. 2021). Marginalize at eval (eval_mc).

The task requires *variable binding*: a triple (h, rt, t) is true iff some e
has (h, ra, e) and (e, rb, t). The model must keep the anonymous intermediate
`e` bound across two context triples and the candidate — which is exactly the
property the schemes handle differently. Train/test entities are disjoint
(true inductive) and `rt` edges never appear in context (no leakage).

Run from the repo root (the dir that contains the `ilp/` package):

    python -m ilp.compare_anonymization
    python -m ilp.compare_anonymization --epochs 40 --chains 300
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    InductiveKGDataset,
    KnowledgeGraph,
    NegativeSampler,
    augment_with_inverse,
    build_negative_sampler,
    build_sample,
    collate,
    read_triples,
    two_hop_neighborhood,
    verify_inductive_split,
)
from .model import InductiveKGModel, load_state_dict_compat
from .vocab import build_vocab, inverse_relation


# --------------------------------------------------------------------------- #
# Synthetic 2-hop composition KG (entity-disjoint train/test).
# --------------------------------------------------------------------------- #
def make_world(n_chains, n_distract, prefix, rng):
    """Return (context_triples, positives, entities).

    context: ra/rb/rd edges only (the observed graph).
    positives: the (h, rt, t) links to predict (never in context).
    """
    context, positives, ents = [], [], []
    eid = 0

    def new_ent():
        nonlocal eid
        eid += 1
        return f"{prefix}_e{eid}"

    for _ in range(n_chains):
        h, e, t = new_ent(), new_ent(), new_ent()
        context.append((h, "ra", e))
        context.append((e, "rb", t))
        positives.append((h, "rt", t))
        ents += [h, e, t]
    for _ in range(n_distract):  # distractors → non-trivial neighborhoods + hard negs
        context.append((rng.choice(ents), "rd", rng.choice(ents)))
    return context, positives, ents


class HardNeg(NegativeSampler):
    """2-hop neighbor negatives, filtered against the true rt-tails."""

    def __init__(self, kg, entity_pool, true_tails):
        super().__init__(kg, entity_pool)
        self.true_tails = true_tails

    def __call__(self, anchor, relation, true_tail, rng):
        nb = two_hop_neighborhood(anchor, self.kg)
        ents = {x for s, _, o in nb for x in (s, o)}
        ents.discard(anchor)
        ents -= self.true_tails.get((anchor, relation), set())
        ents.discard(true_tail)
        if ents:
            return rng.choice(list(ents))
        for _ in range(50):  # fallback: uniform, still filtered
            c = rng.choice(self.entity_pool)
            if c != true_tail and c not in self.true_tails.get((anchor, relation), set()):
                return c
        return true_tail


def true_tails_both_dirs(positives):
    """(h, rt)->{t} and (t, rt__inv)->{h} so the flip in the dataset is filtered."""
    from collections import defaultdict
    tt = defaultdict(set)
    for h, r, t in positives:
        tt[(h, r)].add(t)
        tt[(t, r + "__inv")].add(h)
    return tt


# --------------------------------------------------------------------------- #
# Eval: balanced pos/neg binary classification with hard negatives.
# --------------------------------------------------------------------------- #
def auc(scores, labels):
    """Mann–Whitney AUROC (ties ignored; fine for continuous logits)."""
    pos = sum(1 for l in labels if l == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r + 1
    rank_sum = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


@torch.no_grad()
def evaluate(model, positives, kg, vocab, fixed_values, cfg, device, mc):
    model.eval()
    rng = random.Random(123)
    tt = true_tails_both_dirs(positives)
    sampler = HardNeg(kg, kg.entities, tt)
    instances = []
    for (h, r, t) in positives:
        instances.append((h, r, t, 1.0))
        instances.append((h, r, sampler(h, r, t, rng), 0.0))

    batch = collate([
        build_sample(h, r, c, l, kg, vocab, fixed_values,
                     cfg["max_triples"], cfg["z_pool_size"], rng,
                     exclude_triple=(h, r, c) if l == 1.0 else None,
                     subgraph_hops=cfg["subgraph_hops"])
        for (h, r, c, l) in instances
    ])
    # Process in mini-batches: a single 24k-instance batch OOMs on small GPUs,
    # and runtime mode allocates a [B, z_pool, d] random bank on top of that.
    eval_bs = cfg.get("eval_batch_size", 512)
    n = len(instances)
    logits = torch.zeros(n, device=device)
    for start in range(0, n, eval_bs):
        sl = slice(start, start + eval_bs)
        sub = {k: v[sl].to(device) for k, v in batch.items()}
        acc_l = torch.zeros(sub["label"].shape[0], device=device)
        for _ in range(mc):
            acc_l += model(sub["triples"], sub["mask"],
                           sub["target_relation"], sub["target_tail"])
        logits[sl] = acc_l / mc
    labels = batch["label"].tolist()
    scores = logits.tolist()
    acc = sum((s > 0) == (l == 1.0) for s, l in zip(scores, labels)) / len(labels)
    return acc, auc(scores, [int(l) for l in labels])


def z_collapse(model, z_start, z_pool):
    """Mean off-diagonal cosine similarity of the Z embeddings (learned mode)."""
    W = model.embed.weight.data[z_start:z_start + z_pool]
    W = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    sim = W @ W.t()
    n = W.shape[0]
    return (sim.sum() - n) / (n * (n - 1))


# --------------------------------------------------------------------------- #
def build_multimode(mode, vocab, cfg, device):
    """Construct an InductiveKGModel in the given z_mode with standard wiring."""
    z_pool = cfg["z_pool_size"] if mode != "learned" else 0
    return InductiveKGModel(
        vocab_size=len(vocab), x_token_id=vocab["[X]"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_triple_layers=cfg["n_triple_layers"], n_sab=cfg["n_sab"],
        dropout=cfg["dropout"],
        z_mode=mode, z_start=vocab["[Z_0]"], z_pool=z_pool,
    ).to(device)


def train_one(mode, data, vocab, fixed_values, cfg, device, neg_sampler=None,
              ckpt_dir=None):
    ctx_kg, train_pos, test_ctx_kg, test_pos = data
    z_start = vocab["[Z_0]"]
    torch.manual_seed(cfg["seed"])  # identical init across modes
    random.seed(cfg["seed"])

    model = build_multimode(mode, vocab, cfg, device)

    neg = neg_sampler or HardNeg(ctx_kg, ctx_kg.entities, true_tails_both_dirs(train_pos))
    ds = InductiveKGDataset(
        positive_triples=train_pos, kg=ctx_kg, vocab=vocab,
        fixed_values=fixed_values, entity_pool=ctx_kg.entities,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        neg_per_pos=cfg["neg_per_pos"], both_directions=True,
        seed=cfg["seed"], neg_sampler=neg, subgraph_hops=cfg["subgraph_hops"],
    )
    nw = cfg.get("num_workers", 0)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=nw, collate_fn=collate, drop_last=True,
                        persistent_workers=nw > 0,
                        prefetch_factor=4 if nw > 0 else None)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    last = 0.0
    for ep in range(cfg["epochs"]):
        model.train()
        run, nb = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["triples"], batch["mask"],
                           batch["target_relation"], batch["target_tail"])
            loss = loss_fn(logits, batch["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            nb += 1
        last = run / max(1, nb)
        if (ep + 1) % max(1, cfg["epochs"] // 20) == 0:
            print(f"  [{mode:7s}] epoch {ep + 1:>3}/{cfg['epochs']}  loss={last:.4f}",
                  flush=True)

    mc = cfg["eval_mc"] if mode == "runtime" else 1
    acc, au = evaluate(model, test_pos, test_ctx_kg, vocab, fixed_values,
                       cfg, device, mc)
    coll = z_collapse(model, z_start, cfg["z_pool_size"]).item() if mode == "learned" else float("nan")
    if ckpt_dir is not None:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        ckpt = Path(ckpt_dir) / f"{mode}.pt"
        torch.save({"model": model.state_dict(), "mode": mode, "cfg": cfg,
                    "vocab": vocab, "fixed_values": sorted(fixed_values)}, ckpt)
        print(f"  saved checkpoint → {ckpt}")
    return {"mode": mode, "train_loss": last, "test_acc": acc, "test_auc": au,
            "z_cos": coll, "secs": time.time() - t0}


def eval_one(mode, ckpt_dir, data, device):
    """Re-evaluate a saved checkpoint against `data`'s eval graph. No training.

    Uses the checkpoint's own vocab / fixed_values / cfg so the model matches
    exactly how it was trained — only the eval context (test_ctx_kg/test_pos)
    comes from the current run's --obs-file/--test-file.
    """
    _, _, test_ctx_kg, test_pos = data
    ckpt = torch.load(Path(ckpt_dir) / f"{mode}.pt", map_location=device)
    cfg, vocab = ckpt["cfg"], ckpt["vocab"]
    fixed_values = set(ckpt["fixed_values"])
    model = build_multimode(mode, vocab, cfg, device)
    load_state_dict_compat(model, ckpt["model"])
    t0 = time.time()
    mc = cfg["eval_mc"] if mode == "runtime" else 1
    acc, au = evaluate(model, test_pos, test_ctx_kg, vocab, fixed_values,
                       cfg, device, mc)
    z_start = vocab["[Z_0]"]
    coll = z_collapse(model, z_start, cfg["z_pool_size"]).item() if mode == "learned" else float("nan")
    return {"mode": mode, "train_loss": float("nan"), "test_acc": acc,
            "test_auc": au, "z_cos": coll, "secs": time.time() - t0}


def load_synthetic(cfg, args, rng):
    """Synthetic 2-hop composition worlds (entity-disjoint train/test)."""
    train_ctx, train_pos, _ = make_world(args.chains, args.chains, "tr", rng)
    test_ctx, test_pos, _ = make_world(args.test_chains, args.test_chains, "te", rng)
    ctx_kg = KnowledgeGraph(augment_with_inverse(train_ctx))
    test_ctx_kg = KnowledgeGraph(augment_with_inverse(test_ctx))
    # No schema → everything anonymizes; vocab needs [REL_rt] to exist.
    vocab, fixed_values = build_vocab(
        train_ctx + train_pos, z_pool_size=cfg["z_pool_size"],
        type_relation="", subgraph_hops=cfg["subgraph_hops"],
    )
    data = (ctx_kg, train_pos, test_ctx_kg, test_pos)
    return data, vocab, fixed_values, None  # None → train_one builds HardNeg


def load_real(cfg, args, rng):
    """Real inductive KG: train on data_dir/train.txt, eval on a disjoint
    inference graph (obs_file = observed context, test_file = links to predict).

    Vocab (relations + [VAL_*] schema) is built from the *training* graph only.
    Relation tokens for any obs/test relation are appended so they resolve, but
    [VAL_*] promotion never sees the test graph (no schema leakage).
    """
    fmt = args.triple_format
    train_triples = read_triples(Path(args.data_dir) / "train.txt", fmt=fmt)
    vocab, fixed_values = build_vocab(
        train_triples, z_pool_size=cfg["z_pool_size"],
        cardinality_cutoff=0, tail_diversity_cutoff=args.tail_diversity_cutoff,
        type_relation=args.type_relation, subgraph_hops=cfg["subgraph_hops"],
    )
    if args.max_train and len(train_triples) > args.max_train:
        train_pos = rng.sample(train_triples, args.max_train)
    else:
        train_pos = list(train_triples)
    ctx_kg = KnowledgeGraph(augment_with_inverse(train_pos))

    obs_triples = read_triples(args.obs_file, fmt=fmt)
    test_pos = read_triples(args.test_file, fmt=fmt)
    test_ctx_kg = KnowledgeGraph(augment_with_inverse(obs_triples))

    # Relations are shared schema in inductive KGC; make sure obs/test ones resolve.
    eval_rels = {r for _, r, _ in obs_triples} | {r for _, r, _ in test_pos}
    unseen = {r for r in eval_rels if f"[REL_{r}]" not in vocab}
    for r in eval_rels:
        for rr in (r, inverse_relation(r)):
            if f"[REL_{rr}]" not in vocab:
                vocab[f"[REL_{rr}]"] = len(vocab)

    verify_inductive_split(train_pos, test_pos)
    print(f"train graph: {len(train_triples)} triples (using {len(train_pos)})  "
          f"|schema VAL|={len(fixed_values)}")
    print(f"eval: obs={len(obs_triples)}  test={len(test_pos)}  "
          f"unseen test/obs relations (random REL embed)={len(unseen)}")
    # Hard-ish, fast negatives for the (larger) train graph.
    neg = build_negative_sampler("relation_tail_prior", ctx_kg, ctx_kg.entities)
    return (ctx_kg, train_pos, test_ctx_kg, test_pos), vocab, fixed_values, neg


def main():
    print(">>> compare_anonymization starting (stdout is live)", flush=True)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-mc", type=int, default=8, help="MC draws for runtime eval")
    ap.add_argument("--modes", default="learned,runtime")
    ap.add_argument("--z-pool", type=int, default=128, help="Z-slot pool size")
    # Synthetic-task knobs
    ap.add_argument("--chains", type=int, default=250, help="train composition chains")
    ap.add_argument("--test-chains", type=int, default=120)
    # Real-data knobs (set --data-dir to switch from synthetic to a real KG)
    ap.add_argument("--data-dir", default=None,
                    help="Real KG dir with train.txt. If set, uses real data.")
    ap.add_argument("--obs-file", default=None, help="Observed inference graph (eval context).")
    ap.add_argument("--test-file", default=None, help="Test links to predict.")
    ap.add_argument("--max-train", type=int, default=4000, help="Subsample train triples (CPU).")
    ap.add_argument("--triple-format", default="head_relation_tail")
    ap.add_argument("--type-relation", default="rdf:type")
    ap.add_argument("--tail-diversity-cutoff", type=float, default=0.2)
    ap.add_argument("--device", default="auto",
                    help="'auto' (cuda if available), 'cuda', or 'cpu'.")
    ap.add_argument("--eval-batch-size", type=int, default=512,
                    help="Mini-batch size for the eval forward pass.")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Training batch size (bump for GPU).")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers; >0 parallelizes the CPU sample BFS.")
    ap.add_argument("--ckpt-dir", default=None,
                    help="Save per-mode model checkpoints here (train) / load them (--eval-only).")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training: load --ckpt-dir checkpoints and re-evaluate "
                         "against the current --obs-file/--test-file. No retraining.")
    args = ap.parse_args()

    cfg = {
        "d_model": args.d_model, "n_heads": 4, "n_triple_layers": 1, "n_sab": 2,
        "dropout": 0.1, "epochs": args.epochs, "batch_size": args.batch_size, "lr": 3e-4,
        "max_triples": 32, "z_pool_size": args.z_pool, "neg_per_pos": 4,
        "subgraph_hops": 2, "seed": args.seed, "eval_mc": args.eval_mc,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rng = random.Random(args.seed)

    if args.data_dir:
        if not (args.obs_file and args.test_file):
            raise SystemExit("--data-dir requires --obs-file and --test-file.")
        print(f"[real KG] {args.data_dir}")
        data, vocab, fixed_values, train_neg = load_real(cfg, args, rng)
    else:
        print("[synthetic 2-hop composition]")
        data, vocab, fixed_values, train_neg = load_synthetic(cfg, args, rng)
    print(f"vocab={len(vocab)}  train_pos={len(data[1])}  "
          f"test_pos={len(data[3])}  device={device}\n")

    if args.eval_only and not args.ckpt_dir:
        raise SystemExit("--eval-only requires --ckpt-dir.")

    results = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        print(f"== {mode}{' (eval-only)' if args.eval_only else ''} ==")
        if args.eval_only:
            results.append(eval_one(mode, args.ckpt_dir, data, device))
        else:
            results.append(train_one(mode, data, vocab, fixed_values, cfg, device,
                                     neg_sampler=train_neg, ckpt_dir=args.ckpt_dir))
        print()

    print("=" * 70)
    print(f"{'mode':9s}{'train_loss':>12s}{'test_acc':>10s}{'test_auc':>10s}"
          f"{'z_cos':>9s}{'secs':>8s}")
    print("-" * 70)
    for r in results:
        zc = "  n/a" if r["z_cos"] != r["z_cos"] else f"{r['z_cos']:.3f}"
        print(f"{r['mode']:9s}{r['train_loss']:>12.4f}{r['test_acc']:>10.3f}"
              f"{r['test_auc']:>10.3f}{zc:>9s}{r['secs']:>8.1f}")
    print("=" * 70)
    print("test_auc is the headline (inductive, hard negatives). z_cos is the mean")
    print("off-diagonal cosine of the learned Z embeddings: high ⇒ they collapsed.")


if __name__ == "__main__":
    main()
