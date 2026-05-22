"""Unified CLI: `python -m ilp {train,infer,score,eval}`.

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
import yaml

from .dataset import KnowledgeGraph, augment_with_inverse, read_triples
from .eval import evaluate, filter_known_relations, score_candidates
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
    # cardinality_cutoff=0 disables the cardinality-based fallback heuristic;
    # schema membership should come from `type_relation` (explicit). Raise it
    # if you want low-cardinality relation ranges auto-promoted to [VAL_*].
    "cardinality_cutoff": 0, "neg_samples_per_pos": 4,
    # Negative sampling. "uniform" | "relation_tail_prior" | "two_hop" |
    # {mixture: {uniform: 0.5, two_hop: 0.5}}. Override via --set, e.g.:
    #   --set 'neg_sampler={mixture: {uniform: 0.5, relation_tail_prior: 0.25, two_hop: 0.25}}'
    "neg_sampler": "uniform",
    # Subgraph depth and hop-distance-token feature. With the flag off the
    # model behaves identically to a vocab built without hop tokens; with it
    # on, each entity gets a token encoding its BFS distance to the anchor.
    "subgraph_hops": 2,
    "use_hop_distance_tokens": False,
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


def _apply_overrides(cfg: dict, overrides: list[str] | None) -> None:
    """Apply `--set key=value` overrides. Values are YAML-parsed so numbers,
    booleans, and null work as expected (e.g. `--set lr=1e-3 --set collapse_z=true`).
    Unknown keys raise — typos shouldn't silently no-op."""
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in cfg:
            raise SystemExit(
                f"--set: unknown config key {key!r}. Known keys: {sorted(cfg)}"
            )
        cfg[key] = yaml.safe_load(raw)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = dict(DEFAULT_CFG)
    cfg["data_dir"] = str(Path(args.kg_dir))
    cfg["epochs"] = args.epochs
    if args.triple_format:
        cfg["triple_format"] = args.triple_format
    if args.device:
        cfg["device"] = args.device
    _apply_overrides(cfg, args.set)
    train_model(cfg, save_path=Path(args.save))


def cmd_infer(args: argparse.Namespace) -> None:
    model, vocab, fixed_values, cfg, device = _load_bundle(args.model)
    triples = read_triples(args.train_file, fmt=cfg.get("triple_format", "head_relation_tail"))
    triples, _ = filter_known_relations(triples, vocab)
    kg = KnowledgeGraph(augment_with_inverse(triples))
    candidates = sorted(kg.entities)
    scores = score_candidates(
        model, args.head, args.relation, candidates, kg, vocab, fixed_values,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        batch_size=128, device=device,
        collapse_z=cfg.get("collapse_z", False),
        subgraph_hops=cfg.get("subgraph_hops", 2),
        use_hop_distance_tokens=cfg.get("use_hop_distance_tokens", False),
    )
    topk = sorted(scores.items(), key=lambda kv: -kv[1])[: args.k]
    for tail, s in topk:
        print(f"{s:.4f}\t{tail}")


def cmd_score(args: argparse.Namespace) -> None:
    head, relation, tail = args.triple
    model, vocab, fixed_values, cfg, device = _load_bundle(args.model)
    triples = read_triples(args.data, fmt=cfg.get("triple_format", "head_relation_tail"))
    triples, _ = filter_known_relations(triples, vocab)
    kg = KnowledgeGraph(augment_with_inverse(triples))
    scores = score_candidates(
        model, head, relation, [tail], kg, vocab, fixed_values,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        batch_size=1, device=device,
        collapse_z=cfg.get("collapse_z", False),
        subgraph_hops=cfg.get("subgraph_hops", 2),
        use_hop_distance_tokens=cfg.get("use_hop_distance_tokens", False),
    )
    print(f"{scores[tail]:.4f}")


def cmd_eval(args: argparse.Namespace) -> None:
    model, vocab, fixed_values, cfg, device = _load_bundle(args.model)
    fmt = cfg.get("triple_format", "head_relation_tail")
    kg_dir = Path(args.kg_dir)

    test_path = Path(args.test_file) if args.test_file else kg_dir / "test.txt"
    test = read_triples(test_path, fmt=fmt)

    known: list = list(test)
    train_path = kg_dir / "train.txt"
    if train_path.exists():
        known += read_triples(train_path, fmt=fmt)
    valid_path = kg_dir / "valid.txt"
    if valid_path.exists():
        known += read_triples(valid_path, fmt=fmt)

    # Subgraph context: by default the test split itself (legacy behavior).
    # For GraIL-style inductive eval, pass --obs-file pointing at the observed
    # inductive graph so queries are scored against a richer context — e.g.
    #   --obs-file KGs/WN18RR_v1_ind/train.txt --test-file .../test.txt
    if args.obs_file:
        obs_triples = read_triples(Path(args.obs_file), fmt=fmt)
        # Include the queries themselves so test-internal facts also enter
        # the subgraph; the test triple of each query is excluded inside
        # score_candidates via exclude_triple=(h, r, t).
        obs_triples = obs_triples + test
    else:
        obs_triples = test
    obs_for_graph, dropped_graph = filter_known_relations(obs_triples, vocab)
    if dropped_graph:
        print(f"[eval] removed {dropped_graph} observation-KG triples with unseen relations "
              f"before building subgraph context.")
    test_kg = KnowledgeGraph(augment_with_inverse(obs_for_graph))
    results = evaluate(
        model, test, test_kg, vocab, fixed_values, test_kg.entities,
        known_triples=known,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        device=device, collapse_z=cfg.get("collapse_z", False),
        subgraph_hops=cfg.get("subgraph_hops", 2),
        use_hop_distance_tokens=cfg.get("use_hop_distance_tokens", False),
    )
    for split in ("tail", "head", "avg"):
        m = results[split]
        if split == "avg":
            print(f"{split:5s}  MRR={m['MRR']:.4f}  H@1={m['Hits@1']:.4f}  "
                  f"H@3={m['Hits@3']:.4f}  H@10={m['Hits@10']:.4f}")
        else:
            print(f"{split:5s}  MRR={m['MRR']:.4f}  H@1={m['Hits@1']:.4f}  "
                  f"H@3={m['Hits@3']:.4f}  H@10={m['Hits@10']:.4f}  n={m['n']}")
            if "reachable" in m:
                rm = m["reachable"]; um = m["unreachable"]
                print(f"  reachable    MRR={rm['MRR']:.4f}  H@1={rm['Hits@1']:.4f}  "
                      f"H@10={rm['Hits@10']:.4f}  n={rm['n']}")
                print(f"  unreachable  MRR={um['MRR']:.4f}  H@1={um['Hits@1']:.4f}  "
                      f"H@10={um['Hits@10']:.4f}  n={um['n']}")
    if results.get("n_skipped"):
        print(f"(skipped {results['n_skipped']} test triples with unseen relations)")

    if getattr(args, "json_out", None):
        import json
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[eval] wrote full results to {out_path}")

    if getattr(args, "per_relation", False):
        _print_per_relation(results)


def _print_per_relation(results: dict, top_n: int = 30) -> None:
    """Print per-relation MRR table including inverse rows.

    For each relation r, the inverse row r__inv reuses head's metrics as tail
    and vice versa — predicting the tail of (t, r__inv, h) is the same problem
    as predicting the head of (h, r, t). The visual flip makes the per-relation
    asymmetry obvious: functional relations have one direction near 1 and the
    other near 0; many-to-many relations stay roughly balanced.
    """
    tail_br = results["tail"].get("by_relation", {})
    head_br = results["head"].get("by_relation", {})
    rels = sorted(
        set(tail_br) | set(head_br),
        key=lambda r: -tail_br.get(r, head_br.get(r, {})).get("n", 0),
    )[:top_n]
    print(f"\n--- per-relation MRR (top {len(rels)} by support) ---")
    print(f"{'relation':40s}  {'tail_MRR':>8s}  {'head_MRR':>8s}  {'n':>5s}")
    for r in rels:
        t_mrr = tail_br.get(r, {}).get("MRR", float("nan"))
        h_mrr = head_br.get(r, {}).get("MRR", float("nan"))
        n = tail_br.get(r, head_br.get(r, {})).get("n", 0)
        print(f"{r:40s}  {t_mrr:8.4f}  {h_mrr:8.4f}  {n:5d}")
        print(f"{(r + '__inv'):40s}  {h_mrr:8.4f}  {t_mrr:8.4f}  {n:5d}")


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
    t.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                   help="Override any config key (repeatable). Value is YAML-parsed, "
                        "e.g. --set cardinality_cutoff=0 --set lr=1e-3 --set collapse_z=true.")
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

    e = sub.add_parser("eval", help="Filtered MRR / Hits@K on a held-out split.")
    e.add_argument("--model", required=True, help="Bundled .pt produced by `ilp train`.")
    e.add_argument("--kg-dir", required=True,
                   help="Directory containing test.txt (and optionally train.txt, valid.txt "
                        "to build the filter set).")
    e.add_argument("--test-file", default=None,
                   help="Override path to the eval split (default: {kg-dir}/test.txt).")
    e.add_argument("--obs-file", default=None,
                   help="Observed-graph triples for subgraph context (GraIL-style "
                        "inductive eval). If omitted, the test split is used as context.")
    e.add_argument("--json-out", default=None,
                   help="If set, dump the full results dict (incl. per-relation MRR) as JSON.")
    e.add_argument("--per-relation", action="store_true",
                   help="After eval, print a per-relation MRR table with each relation's "
                        "inverse row alongside it (tail/head columns swap on the inverse).")
    e.set_defaults(func=cmd_eval)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
