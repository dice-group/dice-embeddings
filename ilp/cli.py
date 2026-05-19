"""Unified CLI: `python -m ilp {train,infer,score}`.

Designed to mirror the `pfn` interface so this model is a drop-in replacement.
The CLI flow produces a single self-contained .pt bundle (model + vocab +
config) that `infer` / `score` load without any sidecar files.

For the reproducible-experiment workflow (YAML configs, run dirs with
intermediate checkpoints) use `python -m ilp.train --config ...`
and `python -m ilp.eval --run ...` instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dataset import KnowledgeGraph, augment_with_inverse, read_triples
from .eval import score_candidates
from .model import InductiveKGModel
from .train import train_model

# Sensible defaults for the CLI flow. Anything not overridden by a flag
# falls back to these — chosen to work out of the box on small KGs
# (Countries-S1 scale). The YAML configs under configs/ remain the source
# of truth for reproducible experiments.
DEFAULT_CFG: dict = {
    # Model
    "d_model": 128, "n_heads": 8, "n_triple_layers": 2, "n_sab": 4, "dropout": 0.1,
    # Optimization
    "batch_size": 64, "lr": 1.0e-4, "weight_decay": 1.0e-2,
    "warmup_steps": 200, "max_steps": 2000, "grad_clip": 1.0,
    # Sampling
    "max_triples": 64, "z_pool_size": 256,
    "cardinality_cutoff": 100, "neg_samples_per_pos": 4,
    # Data
    "triple_format": "head_relation_tail", "type_relation": "",
    # Runtime
    "val_every": 1_000_000,  # effectively disabled — bundle mode saves once at the end
    "seed": 0, "num_workers": 2, "device": "cuda",
}


def _build_model(cfg: dict, vocab: dict, device: torch.device) -> InductiveKGModel:
    model = InductiveKGModel(
        vocab_size=len(vocab),
        x_token_id=vocab["[X]"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_triple_layers=cfg["n_triple_layers"],
        n_sab=cfg["n_sab"],
        dropout=cfg["dropout"],
    ).to(device)
    return model


def _load_bundle(model_path: str) -> tuple[InductiveKGModel, dict, set[str], dict, torch.device]:
    bundle = torch.load(model_path, map_location="cpu")
    cfg = bundle["cfg"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    vocab = bundle["vocab"]
    model = _build_model(cfg, vocab, device)
    model.load_state_dict(bundle["model"])
    model.eval()
    return model, vocab, set(bundle["fixed_values"]), cfg, device


def cmd_train(args: argparse.Namespace) -> None:
    cfg = dict(DEFAULT_CFG)
    cfg["data_dir"] = str(Path(args.kg_dir))
    cfg["epochs"] = args.epochs
    if args.triple_format:
        cfg["triple_format"] = args.triple_format
    if args.device:
        cfg["device"] = args.device
    train_model(cfg, save_path=Path(args.save))


def cmd_infer(args: argparse.Namespace) -> None:
    model, vocab, fixed_values, cfg, device = _load_bundle(args.model)
    triples = read_triples(args.train_file, fmt=cfg.get("triple_format", "head_relation_tail"))
    kg = KnowledgeGraph(augment_with_inverse(triples))
    candidates = sorted(kg.entities)
    scores = score_candidates(
        model, args.head, args.relation, candidates, kg, vocab, fixed_values,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        batch_size=128, device=device,
        collapse_z=cfg.get("collapse_z", False),
    )
    topk = sorted(scores.items(), key=lambda kv: -kv[1])[: args.k]
    for tail, s in topk:
        print(f"{s:.4f}\t{tail}")


def cmd_score(args: argparse.Namespace) -> None:
    head, relation, tail = args.triple
    model, vocab, fixed_values, cfg, device = _load_bundle(args.model)
    triples = read_triples(args.data, fmt=cfg.get("triple_format", "head_relation_tail"))
    kg = KnowledgeGraph(augment_with_inverse(triples))
    scores = score_candidates(
        model, head, relation, [tail], kg, vocab, fixed_values,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        batch_size=1, device=device,
        collapse_z=cfg.get("collapse_z", False),
    )
    print(f"{scores[tail]:.4f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ilp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train a model and save a single-file bundle.")
    t.add_argument("--kg-dir", required=True,
                   help="Directory containing train.txt (and optionally valid.txt, test.txt).")
    t.add_argument("--epochs", type=int, required=True)
    t.add_argument("--save", required=True, help="Output .pt path for the bundled model.")
    t.add_argument("--triple-format", default=None,
                   help="Source file column order (see TRIPLE_FORMATS). "
                        f"Default: {DEFAULT_CFG['triple_format']}.")
    t.add_argument("--device", default=None, help="cuda | cpu (auto-falls-back if cuda missing).")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer", help="Top-k tail predictions for (head, relation, ?).")
    i.add_argument("--model", required=True)
    i.add_argument("--train-file", required=True,
                   help="Triples file used to build the KG context for subgraph extraction.")
    i.add_argument("--head", required=True)
    i.add_argument("--relation", required=True)
    i.add_argument("--k", type=int, default=10)
    i.set_defaults(func=cmd_infer)

    s = sub.add_parser("score", help="Score a single (head, relation, tail) triple.")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True,
                   help="Triples file used to build the KG context for subgraph extraction.")
    s.add_argument("--triple", nargs=3, metavar=("HEAD", "RELATION", "TAIL"), required=True)
    s.set_defaults(func=cmd_score)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
