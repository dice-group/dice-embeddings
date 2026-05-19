"""Training loop (spec §6).

Usage (YAML, reproducible-experiment flow):
    python -m ilp.train --config ilp/configs/default.yaml

Usage (CLI flow): see ilp.cli — train_model() is the shared core.
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
    collate,
    read_triples,
    verify_inductive_split,
)
from .model import InductiveKGModel
from .vocab import build_vocab, load_vocab, save_vocab


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
) -> dict:
    """Train a model from a config dict. Returns the saved bundle.

    `cfg` is mutated in place to record any derived values (e.g. epochs →
    max_steps), so the snapshot stored in the bundle reflects what was
    actually used.

    save_path:
        If given, write a single-file bundle {model, vocab, fixed_values,
        cfg, step} to this path. The bundle is fully self-contained and
        loadable by `ilp.cli` for inference.
    run_dir:
        If given, also write `vocab.json`, `config.yaml`, and intermediate
        `model_step{N}.pt` checkpoints there (the reproducible-experiment
        workflow). `model_final.pt` is written here too.

    At least one of `save_path` / `run_dir` should be provided.
    """
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    data_dir = Path(cfg["data_dir"])
    fmt = cfg.get("triple_format", "head_tail_relation")
    train_triples = read_triples(data_dir / "train.txt", fmt=fmt)
    test_path = data_dir / "test.txt"
    if test_path.exists():
        test_triples = read_triples(test_path, fmt=fmt)
        verify_inductive_split(train_triples, test_triples)

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
                type_relation=cfg["type_relation"],
            )
            save_vocab(vocab, fixed_values, vocab_path)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Run dir: {run_dir}")
    else:
        vocab, fixed_values = build_vocab(
            train_triples,
            z_pool_size=cfg["z_pool_size"],
            cardinality_cutoff=cfg["cardinality_cutoff"],
            type_relation=cfg["type_relation"],
        )
    print(f"Vocab size: {len(vocab)} (|fixed_values|={len(fixed_values)})")

    train_kg = KnowledgeGraph(augment_with_inverse(train_triples))

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

    # Epochs are the CLI's natural unit. Resolve into max_steps once we
    # know dataset size, then record both back into cfg for the snapshot.
    if "epochs" in cfg:
        steps_per_epoch = max(1, len(train_triples) // cfg["batch_size"])
        cfg["max_steps"] = cfg["epochs"] * steps_per_epoch
        cfg["warmup_steps"] = min(cfg.get("warmup_steps", 1000), max(1, cfg["max_steps"] // 10))
        print(f"epochs={cfg['epochs']} × steps_per_epoch={steps_per_epoch} "
              f"→ max_steps={cfg['max_steps']} (warmup={cfg['warmup_steps']})")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = InductiveKGModel(
        vocab_size=len(vocab),
        x_token_id=vocab["[X]"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_triple_layers=cfg["n_triple_layers"],
        n_sab=cfg["n_sab"],
        dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: warmup_cosine(s, cfg["warmup_steps"], cfg["max_steps"])
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    step = 0
    running = 0.0
    log_every = 100
    model.train()
    pbar = tqdm(total=cfg["max_steps"], desc="train", unit="step", dynamic_ncols=True)
    while step < cfg["max_steps"]:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(
                batch["triples"], batch["mask"],
                batch["target_relation"], batch["target_tail"],
            )
            loss = loss_fn(logits, batch["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
            pbar.update(1)

            if step % log_every == 0:
                lr = sched.get_last_lr()[0]
                pbar.set_postfix(loss=f"{running / log_every:.4f}", lr=f"{lr:.2e}")
                running = 0.0

            if run_dir is not None and step % cfg["val_every"] == 0:
                ckpt = run_dir / f"model_step{step}.pt"
                torch.save({"model": model.state_dict(), "step": step, "cfg": cfg}, ckpt)
                pbar.write(f"saved {ckpt}")

            if step >= cfg["max_steps"]:
                break
    pbar.close()

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
    print("done.")
    return bundle


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = Path("checkpoints") / cfg["run_name"]
    train_model(cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()
