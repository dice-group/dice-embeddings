"""CLI-first training for InductiveKGModel (spec §6).

Run with plain CLI flags, a YAML config, or both (flags override the config,
which overrides the built-in defaults):

    # pure CLI — trains for 100 epochs by default, then auto-evaluates test.txt
    python -m ilp.train --data-dir KGs/kg_transductive --save runs/kg.pt

    # YAML config
    python -m ilp.train --config ilp/configs/kg_transductive.yaml

    # config + targeted overrides
    python -m ilp.train --config ilp/configs/kg_transductive.yaml --epochs 50 --lr 3e-4

    # inductive: train on graph A, eval on a disjoint inference graph whose
    # train.txt is the observed context and test.txt holds the queries
    python -m ilp.train --data-dir KGs/kg_inductive --save runs/kg_ind.pt \\
        --obs-file  KGs/kg_inductive_ind/train.txt \\
        --test-file KGs/kg_inductive_ind/test.txt \\
        --filter-file KGs/kg_inductive_ind/valid.txt
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import (
    InductiveKGDataset,
    KnowledgeGraph,
    augment_with_inverse,
    build_negative_sampler,
    collate,
    read_triples,
    verify_inductive_split,
)
from .eval import load_eval_inputs, print_eval_results, run_eval
from .model import build_model, resolve_device
from .vocab import build_vocab, format_vocab_summary, load_vocab, save_vocab

# Canonical defaults. A YAML --config overrides these; explicit CLI flags
# override both. Chosen to work out of the box on small/medium KGs.
DEFAULT_CFG: dict = {
    # Model
    "d_model": 128, "n_heads": 8, "n_triple_layers": 2, "n_sab": 4, "dropout": 0.1,
    # Optimization (epoch-based; max_steps is derived at runtime)
    "epochs": 100, "batch_size": 128, "lr": 1.0e-4, "weight_decay": 1.0e-2,
    "warmup_steps": 1000, "grad_clip": 1.0,
    # Mixed precision: bf16 autocast on CUDA (no GradScaler needed). Default on;
    # silently no-ops on CPU or hardware without bf16 support.
    "amp": True,
    # Sampling
    "max_triples": 128, "z_pool_size": 300,
    "cardinality_cutoff": 0, "tail_diversity_cutoff": 0.0,
    "neg_samples_per_pos": 4, "neg_sampler": "uniform",
    "subgraph_hops": 2, "use_hop_distance_tokens": False, "collapse_z": False,
    "dual_subgraph": False,
    # Data
    "triple_format": "head_relation_tail", "type_relation": "",
    # Runtime / checkpointing
    "val_every": 1_000_000, "run_name": "run", "seed": 0, "num_workers": 2,
    "device": "cuda",
}

# cfg keys settable from the CLI (data_dir has no default, handled separately).
CFG_KEYS = set(DEFAULT_CFG) | {"data_dir"}


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def train_model(
    cfg: dict,
    *,
    save_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    eval_after: bool = True,
    eval_test_files: list[str | Path] | None = None,
    eval_obs_files: list[str | Path] | None = None,
    eval_filter_files: list[str | Path] | None = None,
    eval_context_with_test: bool = False,
    eval_per_relation: bool = False,
) -> dict:
    """Train a model from a config dict, then (optionally) evaluate on test.

    `cfg` is mutated in place to record derived values (epochs → max_steps),
    so the snapshot stored in the bundle reflects what was actually used.

    save_path:
        Write a single-file bundle {model, vocab, fixed_values, cfg, step}.
        Fully self-contained and loadable by `ilp.eval` / `ilp.predict`.
    run_dir:
        Also write `vocab.json`, `config.yaml`, intermediate `model_step{N}.pt`
        checkpoints, and `model_final.pt` here (reproducible-experiment flow).

    eval_after:
        After training, run filtered MRR/Hits@K and print the metrics. The
        query files default to [{data_dir}/test.txt] and the context to those
        same queries (transductive self-context) unless `eval_obs_files` is
        given (inductive: a disjoint observed graph); `eval_filter_files`
        defaults to the data_dir's train.txt/valid.txt. Skipped if the resolved
        files don't all exist.
    """
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    data_dir = Path(cfg["data_dir"])
    fmt = cfg.get("triple_format", "head_relation_tail")
    train_triples = read_triples(data_dir / "train.txt", fmt=fmt)
    test_path = data_dir / "test.txt"
    if test_path.exists():
        verify_inductive_split(train_triples, read_triples(test_path, fmt=fmt))

    run_dir = Path(run_dir) if run_dir is not None else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = run_dir / "vocab.json"
        if vocab_path.exists():
            vocab, fixed_values = load_vocab(vocab_path)
        else:
            vocab, fixed_values = build_vocab(
                train_triples,
                z_pool_size=cfg["z_pool_size"],
                cardinality_cutoff=cfg["cardinality_cutoff"],
                tail_diversity_cutoff=cfg.get("tail_diversity_cutoff", 0.0),
                type_relation=cfg["type_relation"],
                subgraph_hops=cfg.get("subgraph_hops", 2),
            )
            save_vocab(vocab, fixed_values, vocab_path)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Run dir: {run_dir}")
    else:
        vocab, fixed_values = build_vocab(
            train_triples,
            z_pool_size=cfg["z_pool_size"],
            cardinality_cutoff=cfg["cardinality_cutoff"],
            tail_diversity_cutoff=cfg.get("tail_diversity_cutoff", 0.0),
            type_relation=cfg["type_relation"],
            subgraph_hops=cfg.get("subgraph_hops", 2),
        )
    print(format_vocab_summary(
        train_triples, fixed_values,
        type_relation=cfg["type_relation"],
        cardinality_cutoff=cfg["cardinality_cutoff"],
        tail_diversity_cutoff=cfg.get("tail_diversity_cutoff", 0.0),
    ))
    print(f"Vocab size: {len(vocab)} (|fixed_values|={len(fixed_values)})")

    train_kg = KnowledgeGraph(augment_with_inverse(train_triples))
    neg_sampler = build_negative_sampler(cfg.get("neg_sampler"), train_kg, train_kg.entities)
    train_ds = InductiveKGDataset(
        positive_triples=train_triples,
        kg=train_kg,
        vocab=vocab,
        fixed_values=fixed_values,
        entity_pool=train_kg.entities,
        max_triples=cfg["max_triples"],
        z_pool=cfg["z_pool_size"],
        neg_per_pos=cfg["neg_samples_per_pos"],
        both_directions=True,
        collapse_z=cfg.get("collapse_z", False),
        neg_sampler=neg_sampler,
        subgraph_hops=cfg.get("subgraph_hops", 2),
        use_hop_distance_tokens=cfg.get("use_hop_distance_tokens", False),
        dual_subgraph=cfg.get("dual_subgraph", False),
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
        drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    # Epochs are the unit. Resolve into steps once we know the loader length,
    # then record everything back into cfg for the saved snapshot.
    epochs = cfg["epochs"]
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(cfg.get("warmup_steps", 1000), max(1, total_steps // 10))
    cfg["steps_per_epoch"] = steps_per_epoch
    cfg["max_steps"] = total_steps
    cfg["warmup_steps"] = warmup_steps
    print(f"epochs={epochs} × steps_per_epoch={steps_per_epoch} "
          f"→ {total_steps} steps (warmup={warmup_steps})")

    device = resolve_device(cfg)
    model = build_model(cfg, vocab, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: warmup_cosine(s, warmup_steps, total_steps)
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # bf16 autocast: ~1.5-2x on Ampere+; bf16's fp32 exponent range means no
    # loss scaling is required, so backward/clip/step stay in their fp32 path.
    use_amp = (
        cfg.get("amp", True)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    if cfg.get("amp", True) and not use_amp:
        print("[amp] requested but unavailable (needs CUDA + bf16); running fp32.")
    elif use_amp:
        print("[amp] bf16 autocast enabled.")

    step = 0
    use_hop = cfg.get("use_hop_distance_tokens", False)
    use_dual = cfg.get("dual_subgraph", False)
    epoch_bar = tqdm(range(epochs), desc="epochs", unit="ep", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        running = 0.0
        n_batches = 0
        batch_bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", unit="batch",
                         leave=False, dynamic_ncols=True)
        for batch in batch_bar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(
                    batch["triples"], batch["mask"],
                    batch["target_relation"], batch["target_tail"],
                    hop_distances=batch.get("hop_distances") if use_hop else None,
                    cand_triples=batch.get("cand_triples") if use_dual else None,
                    cand_mask=batch.get("cand_mask") if use_dual else None,
                    cand_hop_distances=batch.get("cand_hop_distances") if (use_dual and use_hop) else None,
                )
                loss = loss_fn(logits, batch["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()

            running += loss.item()
            n_batches += 1
            step += 1
            batch_bar.set_postfix(loss=f"{running / n_batches:.4f}",
                                  lr=f"{sched.get_last_lr()[0]:.2e}")

            if run_dir is not None and step % cfg["val_every"] == 0:
                ckpt = run_dir / f"model_step{step}.pt"
                torch.save({"model": model.state_dict(), "step": step, "cfg": cfg}, ckpt)
                batch_bar.write(f"saved {ckpt}")
        batch_bar.close()
        epoch_bar.set_postfix(avg_loss=f"{running / max(1, n_batches):.4f}")
    epoch_bar.close()

    bundle = {
        "model": model.state_dict(),
        "vocab": vocab,
        "fixed_values": sorted(fixed_values),
        "cfg": cfg,
        "step": step,
    }
    if run_dir is not None:
        torch.save(bundle, run_dir / "model_final.pt")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, save_path)
        print(f"saved {save_path}")
    print("training done.")

    if eval_after:
        # Resolve explicit file lists. Default: queries = {data_dir}/test.txt,
        # context = those same queries (transductive self-context), filter =
        # data_dir's train.txt/valid.txt. eval_obs_files overrides the context
        # with a disjoint observed graph (inductive eval).
        test_files = [Path(p) for p in eval_test_files] if eval_test_files else [data_dir / "test.txt"]
        obs_files = [Path(p) for p in eval_obs_files] if eval_obs_files else list(test_files)
        if eval_filter_files is not None:
            filter_files = [Path(p) for p in eval_filter_files]
        else:
            filter_files = [data_dir / n for n in ("train.txt", "valid.txt")
                            if (data_dir / n).exists()]
        missing = [p for p in test_files + obs_files if not p.exists()]
        if not missing:
            print("\n" + "=" * 60 + "\nEvaluation (filtered MRR / Hits@K)\n" + "=" * 60)
            model.eval()
            test, known, context = load_eval_inputs(
                fmt, obs_files, test_files,
                include_test_in_context=eval_context_with_test,
                filter_files=filter_files,
            )
            results = run_eval(
                model, vocab, fixed_values, cfg, device,
                test_triples=test, known_triples=known, context_triples=context,
            )
            print_eval_results(results, per_relation=eval_per_relation)
        else:
            print(f"[eval] missing eval file(s) {missing}; skipping auto-eval.")

    return bundle


def resolve_cfg(args: argparse.Namespace) -> dict:
    """Layer config sources: DEFAULT_CFG → --config YAML → explicit CLI flags.

    Only flags actually passed appear in `args` (defaults are SUPPRESS-ed),
    so unset flags never clobber YAML/default values.
    """
    cfg = dict(DEFAULT_CFG)
    if getattr(args, "config", None):
        cfg.update(load_config(args.config))
    for key in CFG_KEYS:
        if key in vars(args):
            cfg[key] = getattr(args, key)
    if "data_dir" not in cfg or not cfg["data_dir"]:
        raise SystemExit("data_dir is required: pass --data-dir or set it in --config.")
    return cfg


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ilp.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=None, help="YAML config file (base values).")
    ap.add_argument("--data-dir", dest="data_dir", default=argparse.SUPPRESS,
                    help="Directory with train.txt (and optionally valid.txt/test.txt).")
    # Output targets
    ap.add_argument("--save", default=None, help="Write a self-contained .pt bundle here.")
    ap.add_argument("--run-dir", default=None,
                    help="Write vocab.json/config.yaml/checkpoints here. "
                         "If neither --save nor --run-dir is given, defaults to checkpoints/{run_name}.")
    # Auto-eval
    ap.add_argument("--no-eval", dest="eval_after", action="store_false",
                    help="Skip the automatic post-training evaluation.")
    ap.add_argument("--test-file", nargs="+", default=None, metavar="FILE",
                    help="Query file(s) for auto-eval (default: {data-dir}/test.txt).")
    ap.add_argument("--obs-file", nargs="+", default=None, metavar="FILE",
                    help="Observed-graph file(s) for subgraph context at eval (inductive: a "
                         "disjoint graph). Default: the query files themselves (transductive self-context).")
    ap.add_argument("--filter-file", nargs="*", default=None, metavar="FILE",
                    help="Extra filter-only file(s) for the known set at eval "
                         "(default: {data-dir}/train.txt and valid.txt).")
    ap.add_argument("--context-with-test", action="store_true",
                    help="Fold the query (test) triples into the eval context graph.")
    ap.add_argument("--per-relation", action="store_true",
                    help="Print a per-relation MRR table after evaluation.")

    g = ap.add_argument_group("hyperparameters (override config/defaults)")

    def opt(flag, dest, typ, help):
        g.add_argument(flag, dest=dest, type=typ, default=argparse.SUPPRESS,
                       help=f"{help} (default: {DEFAULT_CFG[dest]})")

    opt("--d-model", "d_model", int, "Model dimension")
    opt("--n-heads", "n_heads", int, "Attention heads")
    opt("--n-triple-layers", "n_triple_layers", int, "Per-triple encoder layers")
    opt("--n-sab", "n_sab", int, "Set-attention blocks")
    opt("--dropout", "dropout", float, "Dropout")
    opt("--epochs", "epochs", int, "Training epochs")
    opt("--batch-size", "batch_size", int, "Batch size")
    opt("--lr", "lr", float, "Peak learning rate")
    opt("--weight-decay", "weight_decay", float, "AdamW weight decay")
    opt("--warmup-steps", "warmup_steps", int, "Warmup steps (capped to total/10)")
    opt("--grad-clip", "grad_clip", float, "Gradient clip norm")
    opt("--max-triples", "max_triples", int, "Max subgraph triples per sample")
    opt("--z-pool-size", "z_pool_size", int, "Anonymization Z-pool size")
    opt("--cardinality-cutoff", "cardinality_cutoff", int, "Distinct-tail count cutoff for [VAL_*] promotion")
    opt("--tail-diversity-cutoff", "tail_diversity_cutoff", float,
        "Promote a relation's tails to [VAL_*] when distinct_tails/triples < this (e.g. 0.2)")
    opt("--neg-samples-per-pos", "neg_samples_per_pos", int, "Negatives per positive")
    opt("--neg-sampler", "neg_sampler", str, "uniform | relation_tail_prior | two_hop")
    opt("--subgraph-hops", "subgraph_hops", int, "BFS depth for subgraph extraction")
    opt("--triple-format", "triple_format", str, "Source column order (see TRIPLE_FORMATS)")
    opt("--type-relation", "type_relation", str, "Relation whose tails become [VAL_*] schema tokens")
    opt("--val-every", "val_every", int, "Checkpoint every N steps (run-dir mode)")
    opt("--seed", "seed", int, "Random seed")
    opt("--num-workers", "num_workers", int, "DataLoader workers")
    opt("--device", "device", str, "cuda | cpu (auto-falls back to cpu)")
    opt("--run-name", "run_name", str, "Run name (default run dir = checkpoints/{run_name})")
    g.add_argument("--use-hop-distance-tokens", dest="use_hop_distance_tokens",
                   action="store_true", default=argparse.SUPPRESS,
                   help="Add per-entity BFS-distance tokens.")
    g.add_argument("--collapse-z", dest="collapse_z", action="store_true",
                   default=argparse.SUPPRESS, help="Collapse Z pool on exhaustion (z_pool=1 ablation).")
    g.add_argument("--no-amp", dest="amp", action="store_false",
                   default=argparse.SUPPRESS,
                   help="Disable bf16 autocast (default: on when CUDA+bf16 available).")
    g.add_argument("--dual-subgraph", dest="dual_subgraph", action="store_true",
                   default=argparse.SUPPRESS,
                   help="Anchor in both entities: represent the candidate by its own "
                        "k-hop subgraph (candidate tower). Eval precomputes one vector per entity.")
    return ap


def main():
    args = build_parser().parse_args()
    cfg = resolve_cfg(args)

    save_path = Path(args.save) if args.save else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif save_path is None:
        run_dir = Path("checkpoints") / cfg["run_name"]
    else:
        run_dir = None

    train_model(
        cfg,
        save_path=save_path,
        run_dir=run_dir,
        eval_after=args.eval_after,
        eval_test_files=args.test_file,
        eval_obs_files=args.obs_file,
        eval_filter_files=args.filter_file,
        eval_context_with_test=args.context_with_test,
        eval_per_relation=args.per_relation,
    )


if __name__ == "__main__":
    main()
