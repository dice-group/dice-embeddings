"""
Graph Prior-Fitted Network (GraphPFN) for in-context link prediction.

Model
-----
GraphPFN is a **Prior-Fitted Network (PFN)** for knowledge graph link
prediction.  A PFN is meta-trained over a prior so that, at inference time,
it performs in-context prediction purely through a single forward pass —
without any gradient update on the target graph.

**Problem statement.**
Given a knowledge graph G = (E, R, T) with entity set E, relation set R,
and observed triples T ⊆ E × R × E, and a query triple (h_q, r_q, t_q),
predict whether the triple is true:

    s(h_q, r_q, t_q | T_ctx)  =  σ( f_θ(T_ctx, (h_q, r_q, t_q)) )

where T_ctx ⊆ T is a context window of S observed triples, σ is the
sigmoid function, and f_θ is the learned scoring network.  At evaluation
time, all candidate tails are scored and the target is ranked accordingly.

**Embeddings.**
Entity and relation tokens are represented using the frozen pre-trained
SentenceTransformer **all-MiniLM-L6-v2** (384-dimensional, no gradients).
No learnable embedding tables are maintained::

    e  →  ST(e)  ∈ ℝ^{384}   (frozen, from all-MiniLM-L6-v2)
    r  →  ST(r)  ∈ ℝ^{384}   (frozen, from all-MiniLM-L6-v2)

Two learned linear projections bring these down to the model working
dimension d::

    entity_proj   :  ℝ^{384} → ℝ^d
    relation_proj :  ℝ^{384} → ℝ^d

**Triple encoder.**
A two-layer MLP with GELU activations encodes any triple (h, r, t) —
whether a support triple or the complete query triple — into a single
token, preserving the distinct roles of head, relation, and tail:

    τ(h, r, t)  =  MLP_enc( [proj_e(ST(h)) ; proj_r(ST(r)) ; proj_e(ST(t))] )  ∈ ℝ^d

    MLP_enc :  ℝ^{3d}  →  ℝ^{2d}  →  ℝ^d
               Linear, GELU, Linear

**In-context sequence.**
The S support tokens and the query token are concatenated into a sequence
of length S + 1 and passed through a pre-norm Transformer encoder
(norm_first=True, L layers, H heads):

    Z  =  LayerNorm-drop( [τ(sup_1) ; … ; τ(sup_S) ; τ(query)] )
    Z' =  TransformerEncoder(Z)                              ∈ ℝ^{(S+1) × d}

The last token Z'_{S+1} aggregates information from all support triples
via full bidirectional self-attention.

**Prediction head (binary scorer).**
A two-layer MLP (``score_head``) maps the aggregated query token to a
scalar logit:

    f_θ(T_ctx, q)  =  MLP_score( Z'_{S+1} )                 ∈ ℝ

    MLP_score :  ℝ^d  →  ℝ^d  →  ℝ^1
                 Linear, GELU, Linear

    P(triple is true | T_ctx)  =  σ( f_θ(T_ctx, q) )

**Training objective.**
The model is meta-trained on tasks sampled from :class:`RichSubgraphPrior`
using binary cross-entropy loss:

    L  =  -  𝔼_{(T_ctx, q, y) ~ P}  [ y · log σ(f) + (1-y) · log(1-σ(f)) ]

Each episode uses **entity-centric sampling**: a focal entity ``e`` is
chosen at random, its 1-hop neighbourhood (all triples mentioning ``e``)
is collected, one triple from the neighbourhood is selected as the query,
and the remainder (up to ``context_size``) serves as the support context.
The query is either kept as-is (y=1) or corrupted by replacing the tail
with a random entity (y=0).  Because entity/relation identity is carried
by frozen SentenceTransformer embeddings, **no ID re-randomisation is
needed** — the model learns structural patterns directly from semantics.

Optimisation uses AdamW with cosine learning-rate annealing and gradient
clipping (‖g‖ ≤ 1).

**Ranking at evaluation time.**
For each test query (h, r, t*) the model scores (h, r, t_i) for every
entity t_i in the training vocabulary and ranks t* by its score.  This
removes the constraint that the answer must appear in the context.

Command-line usage
-------------------
The examples below use ``KGs/Countries-S1/train.txt``, which ships with
this repository.  Each line is ``head<TAB>relation<TAB>tail``, e.g.::

    western_africa  locatedin  africa
    slovakia        neighbor   ukraine
    belize          locatedin  americas

Train a model and save it::

    python graph_pfn.py train --epochs 10000 --save model.pt

Infer with an explicit support set (real triples from Countries-S1)::

    python graph_pfn.py infer --model model.pt \
        --query slovakia neighbor \
        --support slovakia,neighbor,ukraine slovakia,neighbor,hungary \
                  slovakia,neighbor,austria slovakia,neighbor,czechia \
        --k 5

Infer by sampling the support automatically from the training file::

    python graph_pfn.py infer --model model.pt \
        --query slovakia neighbor \
        --data KGs/Countries-S1/train.txt --context-size 32 --k 5

Score a fully-specified triple (10 passes, context size 32)::

    python graph_pfn.py score --model model.pt \
        --data KGs/Countries-S1/train.txt \
        --triple slovakia neighbor austria \
        --n 10 --context-size 32

    # Output:
    #   Pass    P(true)
    #   -----  ---------
    #       1     0.8213
    #       2     0.7954
    #      ...
    #   Average P(true) over 10 passes: 0.8021  (std=0.0187)

The data file must have one triple per line (tab- or space-separated
string tokens)::

    # KGs/Countries-S1/train.txt (excerpt)
    western_africa  locatedin  africa
    slovakia        neighbor   ukraine
    niger           neighbor   chad

Quick-start: inference (Python API)
-------------------------------------
After meta-training, use a :class:`TriplePFN` model to predict the missing
tail entity given a small support set of ``(head, relation, tail)`` **string**
triples and a query ``(head_str, relation_str, ?)``:

.. code-block:: python

    from graph_pfn import train, infer

    # 1. Meta-train the model (or load a checkpoint)
    model = train(num_epochs=3000, kg_dir="KGs/")

    # 2. Build support context as string triples — no ID mapping needed
    support = [
        ("slovakia", "neighbor", "ukraine"),
        ("slovakia", "neighbor", "hungary"),
        ("slovakia", "neighbor", "austria"),
        ("slovakia", "neighbor", "czechia"),
    ]

    # 3. Run inference — returns (entity_string, logit) pairs
    top_k = infer(model, "slovakia", "neighbor", support, k=3)
    for entity, score in top_k:
        print(f"{entity:<20}  score={score:.4f}")
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pfn_dataset import PFNDataset, RichSubgraphPrior, _load_real_triples, build_dataset
from pfn_inference import evaluate, infer, score_triple
from pfn_model import TriplePFN


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    num_epochs: int = 1000,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    kg_dir: str = "KGs/",
    context_size: int = 32,
    num_episodes: int = 100_000,
    dataset_cache_dir: str = ".pfn_cache",
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    embed_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    seed: int = 42,
    warmup_ratio: float = 0.05,
    eval_train_file: Optional[str] = None,
    eval_test_file: Optional[str] = None,
) -> TriplePFN:
    """Meta-train a GraphPFN model on entity-centric episodes from real KGs.

    Loads all ``train.txt`` files under *kg_dir*, generates pre-computed
    episodes via :class:`RichSubgraphPrior`, and trains the model using
    AdamW with cosine annealing and gradient clipping.

    Parameters
    ----------
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for DataLoader (default: 1024).
    learning_rate : float
        Peak learning rate for AdamW after warmup (default: 1e-4).
    kg_dir : str
        Root directory containing KGs (each with a ``train.txt`` file).
    context_size : int
        Number of support triples per episode.
    num_episodes : int
        Total number of episodes to pre-generate.
    dataset_cache_dir : str
        Directory for pre-computed episode cache.  Re-used as-is on
        subsequent runs if the ``dataset_meta.json`` sidecar is present.
    save_path : str or None
        If given, save the final model to this path.  If the file already
        exists it is loaded and training resumes from that checkpoint.
    device : torch.device or None
        Training device. Defaults to CUDA if available, else CPU.
    embed_dim : int
        Model embedding dimension (default: 512).
    num_heads : int
        Number of transformer attention heads (default: 8).
    num_layers : int
        Number of transformer encoder layers (default: 6).
    dropout : float
        Dropout rate (default: 0.1).
    seed : int
        Random seed for Python, NumPy, and PyTorch (default: 42).
    warmup_ratio : float
        Fraction of total training steps used for linear LR warmup (default: 0.05).
    eval_train_file : str or None
        Path to a ``train.txt`` file used as the in-context support during
        post-training evaluation.  Required together with *eval_test_file*.
    eval_test_file : str or None
        Path to a ``test.txt`` file to evaluate link-prediction metrics
        (MRR, MR, Hits@1/3/10) after training finishes.

    Returns
    -------
    TriplePFN
        The trained model (on CPU for serialization).

    Examples
    --------
    >>> model = train(num_epochs=3000, kg_dir="KGs/")
    >>> torch.save(model.state_dict(), "model.pt")
    """
    _set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset: reuse cache if already generated ────────────────────────────
    meta_path = os.path.join(dataset_cache_dir, "dataset_meta.json")
    if os.path.isfile(meta_path):
        print(f"Found existing dataset cache at {dataset_cache_dir!r} — skipping generation.")
    else:
        print(f"Loading KGs from {kg_dir!r}...")
        kg_pools = _load_real_triples(kg_dir)
        if not kg_pools:
            raise ValueError(f"No KGs found in {kg_dir!r}")
        print("Initialising RichSubgraphPrior...")
        prior = RichSubgraphPrior(kg_pools=kg_pools)
        print(f"Pre-generating {num_episodes:,} episodes to {dataset_cache_dir!r}...")
        build_dataset(prior, num_episodes, context_size, dataset_cache_dir)

    dataset = PFNDataset(dataset_cache_dir, expected_context_size=context_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    num_samples = len(dataset)
    batches_per_epoch = len(dataloader)
    print(
        f"Dataset: {num_samples:,} episodes  |  "
        f"Batch size: {batch_size}  |  "
        f"Batches per epoch: {batches_per_epoch:,}  |  "
        f"Total mini-batch updates: {batches_per_epoch * num_epochs:,}"
    )

    # ── Model: resume from checkpoint if it exists ───────────────────────────
    model = TriplePFN(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    if save_path and os.path.isfile(save_path):
        print(f"Resuming from checkpoint {save_path!r}...")
        ckpt = torch.load(save_path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    else:
        print("Starting training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Per-step linear warmup → cosine decay scheduler.
    total_steps = num_epochs * len(dataloader)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    loss_fn = nn.BCEWithLogitsLoss()

    print(
        f"Training on {device} for {num_epochs} epochs  "
        f"(seed={seed}, lr={learning_rate}, warmup={warmup_steps} steps, total={total_steps} steps)..."
    )

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        ema_loss: Optional[float] = None  # exponential moving average of per-batch loss

        for batch_idx, (support, query, label) in enumerate(dataloader):
            support = support.to(device)
            query = query.to(device)
            label = label.to(device)

            logits = model(support, query)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1

            # EMA loss (α=0.98) gives a smooth per-batch signal, unlike running average
            ema_loss = batch_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * batch_loss

            print_every = max(1, len(dataloader) // 10)
            if (batch_idx + 1) % print_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch + 1:>5d} | Batch {batch_idx + 1:>5d} / {len(dataloader)}"
                    f" | BCE Loss: {batch_loss:.6f} | EMA Loss: {ema_loss:.6f} | LR: {current_lr:.2e}"
                )

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"  Epoch {epoch + 1:>5d} / {num_epochs}  |  Avg Loss: {avg_loss:.6f}  |  EMA Loss: {ema_loss:.6f}")

    if save_path:
        model.cpu()
        torch.save(
            {
                "hparams": {
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
                "state_dict": model.state_dict(),
            },
            save_path,
        )
        print(f"Model saved to {save_path!r}")

    # ── Post-training evaluation ────────────────────────────────────────────
    if eval_train_file and eval_test_file:
        print("\n" + "=" * 60)
        print("Link-Prediction Evaluation")
        print("=" * 60)
        model.to(device)
        model.eval()
        metrics = evaluate(
            model,
            train_file=eval_train_file,
            test_file=eval_test_file,
            device=device,
            context_size=context_size,
        )
        model.cpu()
        print(f"  MRR:     {metrics['MRR']:.4f}")
        print(f"  MR:      {metrics['MR']:.1f}")
        print(f"  Hits@1:  {metrics['Hits@1']:.4f}")
        print(f"  Hits@3:  {metrics['Hits@3']:.4f}")
        print(f"  Hits@10: {metrics['Hits@10']:.4f}")

    return model.cpu()


# ---------------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for training, inference, and scoring."""
    parser = argparse.ArgumentParser(
        description="Graph Prior-Fitted Network (GraphPFN) for link prediction."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run.")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Meta-train a GraphPFN model.")
    train_parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of full passes over the pre-generated episode dataset."
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Number of episodes per mini-batch. Larger values make better use of GPU parallelism."
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Peak learning rate reached after the linear warmup phase (AdamW, default: 1e-4)."
    )
    train_parser.add_argument(
        "--kg-dir", type=str, default="KGs/",
        help="Root directory that contains one or more KG sub-folders, each with a train.txt file."
    )
    train_parser.add_argument(
        "--context-size", type=int, default=64,
        help=(
            "Number of *support triples* shown to the model per episode. "
            "Each support triple is a (head, relation, tail) string tuple drawn from the "
            "1-hop neighbourhood of a focal entity. The query triple (whose label is "
            "predicted) is NOT counted — the transformer input is context_size+1 tokens long."
        ),
    )
    train_parser.add_argument(
        "--num-episodes", type=int, default=5000,
        help=(
            "Total number of training episodes to pre-generate and cache to disk. "
            "Each episode is one (support, query, label) sample: a set of context_size "
            "real triples around a randomly chosen focal entity, plus one query triple "
            "that is either real (label=1) or tail-corrupted (label=0). "
            "This becomes the fixed dataset size; with --batch-size B and --epochs E "
            "there are ceil(num_episodes / B) * E total mini-batch updates."
        ),
    )
    train_parser.add_argument(
        "--embed-dim", type=int, default=512,
        help=(
            "Working embedding dimension d of the model. "
            "Frozen SentenceTransformer embeddings (384-dim) are projected to this size "
            "before being fed to the Transformer encoder."
        ),
    )
    train_parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of self-attention heads in each Transformer encoder layer. Must divide --embed-dim."
    )
    train_parser.add_argument(
        "--num-layers", type=int, default=6,
        help="Number of stacked Transformer encoder layers."
    )
    train_parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability applied inside the Transformer encoder and the score head."
    )
    train_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Python, NumPy, and PyTorch (for reproducibility)."
    )
    train_parser.add_argument(
        "--warmup-ratio", type=float, default=0.05,
        help=(
            "Fraction of total training steps used for linear LR warmup. "
            "E.g. 0.05 means the LR ramps from 0 to --lr over the first 5%% of steps, "
            "then follows a cosine decay back to 0."
        ),
    )
    train_parser.add_argument(
        "--save", type=str, default=None,
        help="File path for saving the trained model (.pt). If the file already exists, training resumes from it."
    )
    train_parser.add_argument(
        "--eval-train", type=str, default=None, metavar="TRAIN_FILE",
        help=(
            "Path to train.txt used as the in-context support for post-training evaluation. "
            "Requires --eval-test. Example: KGs/UMLS/train.txt"
        ),
    )
    train_parser.add_argument(
        "--eval-test", type=str, default=None, metavar="TEST_FILE",
        help=(
            "Path to test.txt to compute link-prediction metrics (MRR, MR, Hits@1/3/10) "
            "after training finishes. Requires --eval-train. Example: KGs/UMLS/test.txt"
        ),
    )

    # Infer subcommand
    infer_parser = subparsers.add_parser(
        "infer",
        help="Predict top-k tail entities for a (head, relation, ?) query.",
        description=(
            "Load a trained GraphPFN model, use a KG file as in-context support, "
            "and rank all candidate entities for a given (head, relation) query.\n\n"
            "Example:\n"
            "  python graph_pfn.py infer \\\n"
            "      --model model.pt \\\n"
            "      --train-file KGs/Countries-S1/train.txt \\\n"
            "      --head slovakia --relation neighbor --k 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer_parser.add_argument(
        "--model", type=str, required=True,
        help="Path to a saved model checkpoint (.pt).",
    )
    infer_parser.add_argument(
        "--train-file", type=str, required=True,
        metavar="TRAIN_TXT",
        help="Path to train.txt whose triples are used as in-context support.",
    )
    infer_parser.add_argument(
        "--head", type=str, required=True,
        help="Head entity string token for the query (e.g. 'slovakia').",
    )
    infer_parser.add_argument(
        "--relation", type=str, required=True,
        help="Relation string token for the query (e.g. 'neighbor').",
    )
    infer_parser.add_argument(
        "--k", type=int, default=10,
        help="Number of top-k tail predictions to display (default: 10).",
    )
    infer_parser.add_argument(
        "--support-size", type=int, default=None, metavar="N",
        help=(
            "Use only the first N triples from train.txt as support. "
            "Set this to the context_size used during training to test memorisation "
            "(e.g. --support-size 64). Defaults to all triples."
        ),
    )
    infer_parser.add_argument(
        "--show-support", action="store_true",
        help="Print the support triples before scoring so you can pick a query that is in-context.",
    )

    # Score subcommand
    score_parser = subparsers.add_parser("score", help="Score a triple.")
    score_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model."
    )
    score_parser.add_argument(
        "--triple", type=str, nargs=3, required=True, metavar=("HEAD", "RELATION", "TAIL"),
        help="Triple to score (h, r, t)."
    )
    score_parser.add_argument(
        "--data", type=str, required=True, help="Data file for context sampling."
    )
    score_parser.add_argument(
        "--n", type=int, default=10, help="Number of passes."
    )
    score_parser.add_argument(
        "--context-size", type=int, default=32, help="Context size per pass."
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            kg_dir=args.kg_dir,
            context_size=args.context_size,
            num_episodes=args.num_episodes,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seed=args.seed,
            warmup_ratio=args.warmup_ratio,
            save_path=args.save,
            eval_train_file=args.eval_train,
            eval_test_file=args.eval_test,
        )
    elif args.command == "infer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.model, map_location=device)
        if isinstance(ckpt, dict) and "hparams" in ckpt:
            model = TriplePFN(**ckpt["hparams"])
            model.load_state_dict(ckpt["state_dict"])
        else:
            model = TriplePFN()
            model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        # Load triples from the train file as in-context support.
        support: List[Tuple[str, str, str]] = []
        with open(args.train_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 3:
                    support.append((parts[0], parts[1], parts[2]))

        if args.support_size is not None:
            support = support[:args.support_size]
            print(f"Support: first {len(support)} triples from '{args.train_file}'")
        else:
            print(f"Support loaded: {len(support):,} triples from '{args.train_file}'")

        if args.show_support:
            print(f"\n{'─'*52}")
            print(f"  {'#':<5}  {'Head':<20}  {'Relation':<15}  Tail")
            print(f"  {'─'*5}  {'─'*20}  {'─'*15}  {'─'*20}")
            for i, (h, r, t) in enumerate(support, start=1):
                print(f"  {i:<5}  {h:<20}  {r:<15}  {t}")
            print(f"{'─'*52}\n")

        results = infer(model, args.head, args.relation, support, k=args.k, device=device)

        print(f"\nTop-{args.k} predictions for ({args.head}, {args.relation}, ?):")
        print(f"  {'Entity':<30}  Log P(true)")
        print(f"  {'-'*30}  -----------")
        for rank, (entity, score) in enumerate(results, start=1):
            log_p_true = torch.nn.functional.logsigmoid(torch.tensor(score)).item()
            print(f"  {rank}. {entity:<28}  {log_p_true:>11.4f}")

    elif args.command == "score":
        ckpt = torch.load(args.model, map_location="cpu")
        if isinstance(ckpt, dict) and "hparams" in ckpt:
            model = TriplePFN(**ckpt["hparams"])
            model.load_state_dict(ckpt["state_dict"])
        else:
            model = TriplePFN()
            model.load_state_dict(ckpt)
        model.eval()

        h, r, t = args.triple
        prob = score_triple(
            model,
            h, r, t,
            args.data,
            n=args.n,
            context_size=args.context_size,
        )
        print(f"P({h} {r} {t}) = {prob:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

