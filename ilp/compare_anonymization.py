"""Compare entity-featurization schemes for inductive KGE (CPU-friendly).

Trains the *same* InductiveKGModel under three schemes for the anonymous
entity slots, on the *same* synthetic data and seed, so differences are
attributable to the scheme alone:

    learned  – current approach: [Z_i] are rows of a learned nn.Embedding.
    frozen   – a fixed random pool of vectors, assigned per sample, NOT learned.
    runtime  – no pool: a fresh random vector per distinct entity, per forward
               pass (pure RNI; Abboud et al. 2021).

The task requires *variable binding*: a triple (h, rt, t) is true iff some e
has (h, ra, e) and (e, rb, t). The model must keep the anonymous intermediate
`e` bound across two context triples and the candidate — which is exactly the
property the three schemes handle differently. Train/test entities are disjoint
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
from .model import InductiveKGModel
from .vocab import build_vocab, inverse_relation


# --------------------------------------------------------------------------- #
# Model variant: same architecture, three ways to embed the anonymous slots.
# --------------------------------------------------------------------------- #
class MultiModeKGModel(InductiveKGModel):
    """InductiveKGModel where [Z_*] slots use learned / frozen / runtime vectors.

    Only the entity-slot embedding changes; [X], [REL_*], [VAL_*], [HOP_*]
    stay learned exactly as in the base model. `z_start`/`z_pool` describe the
    contiguous block of [Z_0..Z_{z_pool-1}] ids in the vocab.
    """

    def __init__(self, *args, z_mode: str = "learned", z_start: int = 0,
                 z_pool: int = 0, **kw):
        super().__init__(*args, **kw)
        self.z_mode = z_mode
        self.z_start = z_start
        self.z_pool = max(1, z_pool)
        bank = torch.randn(self.z_pool, self.d_model)
        bank = bank / bank.norm(dim=-1, keepdim=True)
        self.register_buffer("z_bank", bank)  # frozen pool

    def _entity_embed(self, ids: torch.Tensor, rand_bank: torch.Tensor | None):
        """Embed token ids, overriding [Z_*] positions per the active scheme.

        `ids` has shape [B, ...] (leading dim must be the batch). `rand_bank`
        is [B, z_pool, d] for runtime mode, else None.
        """
        emb = self.embed(ids)
        if self.z_mode == "learned":
            return emb
        is_z = (ids >= self.z_start) & (ids < self.z_start + self.z_pool)
        z_local = (ids - self.z_start).clamp(0, self.z_pool - 1)
        if self.z_mode == "frozen":
            z_emb = self.z_bank[z_local]
        else:  # runtime
            B, d = ids.shape[0], self.d_model
            flat = z_local.reshape(B, -1)                      # [B, M]
            g = torch.gather(rand_bank, 1, flat.unsqueeze(-1).expand(-1, -1, d))
            z_emb = g.reshape(*ids.shape, d)
        return torch.where(is_z.unsqueeze(-1), z_emb, emb)

    def forward(self, triples, mask, target_relation, target_tail,
                hop_distances=None):
        B, N, _ = triples.shape
        rand_bank = None
        if self.z_mode == "runtime":
            rand_bank = torch.randn(B, self.z_pool, self.d_model,
                                    device=triples.device)
            rand_bank = rand_bank / rand_bank.norm(dim=-1, keepdim=True)

        tok = self._entity_embed(triples, rand_bank) + self.intra_pos
        if hop_distances is not None:
            tok = tok + self.embed(hop_distances)
        tok = self.triple_encoder(tok.view(B * N, 3, -1))
        triple_vec = tok.mean(dim=1).view(B, N, -1)

        x_emb = self.embed(torch.full((B, 1), self.x_token_id,
                                      device=triples.device, dtype=torch.long))
        h = torch.cat([x_emb, triple_vec], dim=1)
        mask_ext = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        h = self.sab_stack(h, src_key_padding_mask=~mask_ext)
        pooled = h[:, 0]

        tr = self.embed(target_relation)
        tt = self._entity_embed(target_tail, rand_bank)
        target_emb = tr + tt
        feats = torch.cat([pooled, target_emb, pooled * target_emb], dim=-1)
        return self.classifier(feats).squeeze(-1)


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
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = torch.zeros(len(instances), device=device)
    for _ in range(mc):
        logits += model(batch["triples"], batch["mask"],
                        batch["target_relation"], batch["target_tail"])
    logits /= mc
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
def train_one(mode, data, vocab, fixed_values, cfg, device, neg_sampler=None):
    ctx_kg, train_pos, test_ctx_kg, test_pos = data
    z_start = vocab["[Z_0]"]
    torch.manual_seed(cfg["seed"])  # identical init across modes
    random.seed(cfg["seed"])

    model = MultiModeKGModel(
        vocab_size=len(vocab), x_token_id=vocab["[X]"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_triple_layers=cfg["n_triple_layers"], n_sab=cfg["n_sab"],
        dropout=cfg["dropout"],
        z_mode=mode, z_start=z_start, z_pool=cfg["z_pool_size"],
    ).to(device)

    neg = neg_sampler or HardNeg(ctx_kg, ctx_kg.entities, true_tails_both_dirs(train_pos))
    ds = InductiveKGDataset(
        positive_triples=train_pos, kg=ctx_kg, vocab=vocab,
        fixed_values=fixed_values, entity_pool=ctx_kg.entities,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        neg_per_pos=cfg["neg_per_pos"], both_directions=True,
        seed=cfg["seed"], neg_sampler=neg, subgraph_hops=cfg["subgraph_hops"],
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)

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
        if (ep + 1) % max(1, cfg["epochs"] // 5) == 0:
            print(f"  [{mode:7s}] epoch {ep + 1:>3}/{cfg['epochs']}  loss={last:.4f}")

    mc = cfg["eval_mc"] if mode == "runtime" else 1
    acc, au = evaluate(model, test_pos, test_ctx_kg, vocab, fixed_values,
                       cfg, device, mc)
    coll = z_collapse(model, z_start, cfg["z_pool_size"]).item() if mode == "learned" else float("nan")
    return {"mode": mode, "train_loss": last, "test_acc": acc, "test_auc": au,
            "z_cos": coll, "secs": time.time() - t0}


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-mc", type=int, default=8, help="MC draws for runtime eval")
    ap.add_argument("--modes", default="learned,frozen,runtime")
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
    args = ap.parse_args()

    cfg = {
        "d_model": args.d_model, "n_heads": 4, "n_triple_layers": 1, "n_sab": 2,
        "dropout": 0.1, "epochs": args.epochs, "batch_size": 64, "lr": 3e-4,
        "max_triples": 32, "z_pool_size": args.z_pool, "neg_per_pos": 4,
        "subgraph_hops": 2, "seed": args.seed, "eval_mc": args.eval_mc,
    }
    device = torch.device("cpu")
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
          f"test_pos={len(data[3])}  device=cpu\n")

    results = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        print(f"== {mode} ==")
        results.append(train_one(mode, data, vocab, fixed_values, cfg, device,
                                 neg_sampler=train_neg))
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
