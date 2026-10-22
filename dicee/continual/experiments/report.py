"""Unified per-snapshot experimental report for continual KGE runs.

Reads the artifacts produced by ``run_experiment.py`` for a single experiment
(``config.json``, ``sequence/sequence_metadata.json``, ``initial_metrics.json``,
``all_metrics.json``) and prints/saves a structured, snapshot-by-snapshot
comparison that answers, for every update step:

  1. How many entities/relations were added (and removed -- always 0 under
     the current additions-only/cumulative protocol; deletion is out of
     scope for this milestone, see Section 15).
  2. What is the performance of the base/oracle model (retrain-from-scratch
     on the full cumulative graph) at this snapshot.
  3. What is the performance of each continual distillation method
     (finetune / uniform / chebyshev, or whichever subset was run).
  4. What is the training runtime of each method (and the oracle) at this
     snapshot.
  5. How did training progress within the snapshot (loss decay: first,
     min, last epoch loss, plus the full per-epoch curve for later
     plotting/CSV export).

This report is generated automatically at the end of every
``run_experiment.py`` run (printed to stdout, plus optionally saved as JSON
and/or a per-epoch-loss CSV via that script's ``--save_report_json`` /
``--save_losses_csv`` flags). It can also be regenerated standalone for any
past experiment directory::

    from dicee.continual.experiments.report import build_report, render_report_text
    report = build_report(exp_dir)
    print(render_report_text(report))
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional


ORACLE_KEY = "retrain_from_scratch"


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_report(exp_dir: str) -> Dict[str, Any]:
    """Assemble a structured report dict for a single experiment directory.

    Parameters
    ----------
    exp_dir:
        Path to an experiment output directory, e.g.
        ``Experiments/continual/umls_mixed/umls_mixed`` (the directory that
        directly contains ``config.json``, ``all_metrics.json``, etc.).
    """
    config_path = os.path.join(exp_dir, "config.json")
    seq_meta_path = os.path.join(exp_dir, "sequence", "sequence_metadata.json")
    initial_metrics_path = os.path.join(exp_dir, "initial_metrics.json")
    all_metrics_path = os.path.join(exp_dir, "all_metrics.json")

    config = _load_json(config_path) if os.path.exists(config_path) else {}
    seq_meta = _load_json(seq_meta_path) if os.path.exists(seq_meta_path) else {"steps": []}
    initial_metrics = _load_json(initial_metrics_path) if os.path.exists(initial_metrics_path) else {}
    all_metrics = _load_json(all_metrics_path) if os.path.exists(all_metrics_path) else {}

    steps_meta = {s["step"]: s for s in seq_meta.get("steps", [])}
    method_names = [m for m in all_metrics.keys() if m != ORACLE_KEY]
    has_oracle = ORACLE_KEY in all_metrics
    num_t0_triples = seq_meta.get("num_t0_triples")

    num_steps = max(
        [s["step"] for s in seq_meta.get("steps", [])]
        + [r["step"] for rs in all_metrics.values() for r in rs],
        default=0,
    )

    snapshots: List[Dict[str, Any]] = []

    # Snapshot 0 (T0): no churn (it's the base), only base-model performance.
    snapshots.append(
        {
            "step": 0,
            "entities_added": None,
            "entities_removed": 0,
            "relations_added": None,
            "relations_removed": 0,
            "num_active_entities": None,
            "num_active_relations": None,
            "num_added_triples": num_t0_triples,
            "num_cumulative_triples": num_t0_triples,
            "methods": {
                "base_model": {
                    "test_MRR": initial_metrics.get("MRR"),
                    "test_H@1": initial_metrics.get("H@1"),
                    "test_H@3": initial_metrics.get("H@3"),
                    "test_H@10": initial_metrics.get("H@10"),
                    "training_time_seconds": initial_metrics.get("training_time_seconds"),
                    "runtime_seconds": initial_metrics.get("runtime_seconds"),
                    "losses": initial_metrics.get("losses"),
                }
            },
        }
    )

    for step in range(1, num_steps + 1):
        meta = steps_meta.get(step, {})
        entry: Dict[str, Any] = {
            "step": step,
            "entities_added": meta.get("num_new_entities"),
            "entities_removed": meta.get("num_removed_entities", 0),
            "relations_added": meta.get("num_new_relations"),
            "relations_removed": meta.get("num_removed_relations", 0),
            "num_active_entities": meta.get("num_active_entities"),
            "num_active_relations": meta.get("num_active_relations"),
            "num_added_triples": meta.get("num_added_triples"),
            "num_cumulative_triples": meta.get("num_cumulative_triples"),
            "density": meta.get("density"),
            "methods": {},
        }

        for method_name, results in all_metrics.items():
            match = next((r for r in results if r.get("step") == step), None)
            if match is None:
                continue
            losses = match.get("losses")
            loss_summary = None
            if losses:
                loss_summary = {
                    "first_epoch_loss": losses[0],
                    "last_epoch_loss": losses[-1],
                    "min_epoch_loss": min(losses),
                    "num_epochs": len(losses),
                }
            entry["methods"][method_name] = {
                "test_MRR": match.get("test_MRR"),
                "test_H@1": match.get("test_H@1"),
                "test_H@3": match.get("test_H@3"),
                "test_H@10": match.get("test_H@10"),
                "training_time_seconds": match.get("training_time_seconds"),
                "test_runtime_seconds": match.get("test_runtime_seconds"),
                "MRR_new": match.get("MRR_new"),
                "MRR_old": match.get("MRR_old"),
                "Delta_MRR_stable": match.get("Delta_MRR_stable"),
                "ScoreDrift": match.get("ScoreDrift"),
                "loss_summary": loss_summary,
                "losses": losses,
            }

        snapshots.append(entry)

    return {
        "exp_dir": exp_dir,
        "config": config,
        "has_oracle": has_oracle,
        "method_names": method_names,
        "snapshots": snapshots,
    }


def render_report_text(report: Dict[str, Any]) -> str:
    """Render the structured report as a human-readable text table."""
    lines: List[str] = []
    cfg = report["config"]
    lines.append("=" * 100)
    lines.append(f"Continual KGE Experiment Report: {cfg.get('experiment_id', report['exp_dir'])}")
    lines.append("=" * 100)
    lines.append(
        f"model={cfg.get('model', 'DistMult')}  embedding_dim={cfg.get('embedding_dim')}  "
        f"dataset={cfg.get('dataset_dir')}  scenario={cfg.get('scenario')}  seed={cfg.get('seed')}"
    )
    lines.append(
        f"epochs_initial={cfg.get('epochs_initial')}  epochs_per_step={cfg.get('epochs_per_step')}  "
        f"methods={cfg.get('methods')}  oracle={'yes' if report['has_oracle'] else 'no'}"
    )
    lines.append("")

    for snap in report["snapshots"]:
        step = snap["step"]
        if step == 0:
            lines.append(f"--- Snapshot T0 (initial training, base model) ---")
            lines.append(
                f"  Triples: {snap.get('num_cumulative_triples')} available at T0 "
                f"(initial_fraction of full train split)"
            )
            base = snap["methods"].get("base_model", {})
            lines.append(
                f"  base_model: MRR={_fmt(base.get('test_MRR'))} H@1={_fmt(base.get('test_H@1'))} "
                f"H@3={_fmt(base.get('test_H@3'))} H@10={_fmt(base.get('test_H@10'))} "
                f"train_time={_fmt(base.get('training_time_seconds'), 3)}s"
            )
            losses = base.get("losses")
            if losses:
                lines.append(
                    f"    loss decay: first={_fmt(losses[0])} min={_fmt(min(losses))} "
                    f"last={_fmt(losses[-1])} (over {len(losses)} epochs)"
                )
            lines.append("")
            continue

        lines.append(f"--- Snapshot T{step} ---")
        lines.append(
            f"  Entities: +{snap['entities_added']} added, {snap['entities_removed']} removed "
            f"(active={snap['num_active_entities']})"
        )
        lines.append(
            f"  Relations: +{snap['relations_added']} added, {snap['relations_removed']} removed "
            f"(active={snap['num_active_relations']})"
        )
        lines.append(
            f"  Triples: +{snap.get('num_added_triples')} added this step, "
            f"{snap.get('num_cumulative_triples')} total available at T{step}  "
            f"density={_fmt(snap.get('density'))}"
        )
        lines.append(
            f"  {'method':<22}{'MRR':>8}{'H@1':>8}{'H@3':>8}{'H@10':>8}"
            f"{'train_s':>10}{'loss(first->last)':>22}"
        )
        # Oracle first, then each distillation method, for easy visual comparison.
        ordered_methods = ([ORACLE_KEY] if ORACLE_KEY in snap["methods"] else []) + [
            m for m in snap["methods"] if m != ORACLE_KEY
        ]
        for method_name in ordered_methods:
            m = snap["methods"][method_name]
            ls = m.get("loss_summary")
            loss_str = f"{_fmt(ls['first_epoch_loss'])} -> {_fmt(ls['last_epoch_loss'])}" if ls else "-"
            label = "retrain" if method_name == ORACLE_KEY else method_name
            lines.append(
                f"  {label:<22}{_fmt(m.get('test_MRR')):>8}{_fmt(m.get('test_H@1')):>8}"
                f"{_fmt(m.get('test_H@3')):>8}{_fmt(m.get('test_H@10')):>8}"
                f"{_fmt(m.get('training_time_seconds'), 3):>10}{loss_str:>22}"
            )
        lines.append("")

    return "\n".join(lines)


def save_losses_csv(report: Dict[str, Any], csv_path: str) -> None:
    """Export per-epoch loss curves (one row per epoch per method per step) to CSV."""
    rows = []
    for snap in report["snapshots"]:
        step = snap["step"]
        for method_name, m in snap["methods"].items():
            losses = m.get("losses")
            if not losses:
                continue
            for epoch_idx, loss_val in enumerate(losses):
                rows.append(
                    {"step": step, "method": method_name, "epoch": epoch_idx, "loss": loss_val}
                )
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "method", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def save_report_json(report: Dict[str, Any], json_path: str) -> None:
    """Save the structured report dict as JSON."""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
