"""Preliminary large-scale CKGE experiment sweep.

Runs the controlled continual-KGE experiment across both datasets (UMLS,
KINSHIP), all three update scenarios (existing_entities, new_entities,
mixed), with 5 snapshots (``num_steps=5``), the retrain-from-scratch
oracle plus finetune/uniform/chebyshev baselines, repeated over multiple
seeds for variance estimation, and a full ``beta`` x ``cheb_K`` grid sweep
for the Chebyshev method (``cheb_K`` is the number of Chebyshev propagation
hops/order -- swept from 1 to ``num_steps``, since with 5 snapshots there is
no reason to propagate influence further than the number of update steps
that have actually happened so far).

Each (dataset, scenario, seed) combination is run:
  1. Once as a "base" run with ``methods=["finetune", "uniform"]`` and the
     retrain-from-scratch oracle enabled (beta/cheb_K are irrelevant to
     these methods).
  2. Once per (beta, cheb_K) pair in the grid ``--betas`` x ``--cheb_Ks``,
     with ``methods=["chebyshev"]`` and the oracle disabled (already
     captured by the base run), so the oracle/finetune/uniform are never
     needlessly recomputed once per grid cell.

All per-step, per-method results (including the base model at T0) are
flattened into a single long-format CSV with one row per
(dataset, scenario, seed, beta, cheb_K, step, method), ready for
groupby/std analysis across seeds.

Usage
-----
    python -m dicee.continual.experiments.run_preliminary_sweep

    # Customize:
    python -m dicee.continual.experiments.run_preliminary_sweep \
        --datasets KGs/UMLS KGs/KINSHIP --scenarios existing_entities new_entities mixed \
        --num_steps 5 --seeds 0 1 2 3 4 --betas 0.05 0.5 1 2 5 10 --cheb_Ks 1 2 3 4 5 \
        --output_csv ckge_prelimary_results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from dicee.continual.experiments.run_experiment import DEFAULT_CONFIG, run_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASETS = ["KGs/UMLS", "KGs/KINSHIP"]
DEFAULT_SCENARIOS = ["existing_entities", "new_entities", "mixed"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_BETAS = [0.05, 0.1, 0.5, 1, 2]  # Chebyshev beta sweep for preliminary experiments
DEFAULT_CHEB_KS = [1, 2, 3, 4, 5]  # Chebyshev propagation order sweep (1..num_steps)

CSV_FIELDS = [
    "dataset", "scenario", "seed", "beta", "cheb_K", "step", "method",
    "num_cumulative_triples", "num_added_triples", "num_new_entities", "num_new_relations",
    "test_MRR", "test_H@1", "test_H@3", "test_H@10",
    "training_time_seconds",
    "first_epoch_loss", "last_epoch_loss", "min_epoch_loss", "num_epochs",
]


def _dataset_name(dataset_dir: str) -> str:
    return Path(dataset_dir).name


def _flatten_report(report: Dict, dataset: str, scenario: str, seed: int, beta: Optional[float],
                     cheb_K: Optional[int] = None) -> List[Dict]:
    """Flatten a single `report.py`-style structured report into long-format rows."""
    rows: List[Dict] = []
    for snap in report["snapshots"]:
        step = snap["step"]
        common = {
            "dataset": dataset,
            "scenario": scenario,
            "seed": seed,
            "beta": beta,
            "cheb_K": cheb_K,
            "step": step,
            "num_cumulative_triples": snap.get("num_cumulative_triples"),
            "num_added_triples": snap.get("num_added_triples"),
            "num_new_entities": snap.get("entities_added"),
            "num_new_relations": snap.get("relations_added"),
        }
        for method_name, m in snap["methods"].items():
            ls = m.get("loss_summary") or {}
            rows.append({
                **common,
                "method": method_name,
                "test_MRR": m.get("test_MRR"),
                "test_H@1": m.get("test_H@1"),
                "test_H@3": m.get("test_H@3"),
                "test_H@10": m.get("test_H@10"),
                "training_time_seconds": m.get("training_time_seconds"),
                "first_epoch_loss": ls.get("first_epoch_loss"),
                "last_epoch_loss": ls.get("last_epoch_loss"),
                "min_epoch_loss": ls.get("min_epoch_loss"),
                "num_epochs": ls.get("num_epochs"),
            })
    return rows


def _make_config(
    dataset_dir: str, scenario: str, seed: int, num_steps: int,
    experiment_id: str, output_dir: str, methods: List[str], include_oracle: bool,
    beta: Optional[float] = None, cheb_K: Optional[int] = None, force_regenerate: bool = False,
) -> Dict:
    config = dict(DEFAULT_CONFIG)
    config.update({
        "experiment_id": experiment_id,
        "output_dir": output_dir,
        "dataset_dir": dataset_dir,
        "scenario": scenario,
        "seed": seed,
        "num_steps": num_steps,
        "methods": methods,
        "include_oracle": include_oracle,
        "save_report_json": None,
        "save_losses_csv": None,
        "force_regenerate_sequence": force_regenerate,
    })
    if beta is not None:
        config["beta"] = beta
    if cheb_K is not None:
        config["cheb_K"] = cheb_K
    return config


def run_sweep(
    datasets: List[str] = DEFAULT_DATASETS,
    scenarios: List[str] = DEFAULT_SCENARIOS,
    seeds: List[int] = DEFAULT_SEEDS,
    betas: List[float] = DEFAULT_BETAS,
    cheb_Ks: List[int] = DEFAULT_CHEB_KS,
    num_steps: int = 5,
    output_root: str = "Experiments/continual_prelim",
    output_csv: str = "ckge_prelimary_results.csv",
    force_regenerate: bool = False,
) -> str:
    all_rows: List[Dict] = []
    t_sweep_start = time.time()
    total_runs = len(datasets) * len(scenarios) * len(seeds) * (1 + len(betas) * len(cheb_Ks))
    run_idx = 0

    for dataset_dir in datasets:
        dname = _dataset_name(dataset_dir)
        for scenario in scenarios:
            for seed in seeds:
                # --- 1. Base run: finetune + uniform + oracle (beta-independent) ---
                run_idx += 1
                base_id = f"{dname}_{scenario}_seed{seed}_base"
                logger.info("[%s/%s] BASE run: %s / %s / seed=%s", run_idx, total_runs, dname, scenario, seed)
                base_config = _make_config(
                    dataset_dir, scenario, seed, num_steps,
                    experiment_id=base_id, output_dir=output_root,
                    methods=["finetune", "uniform"], include_oracle=True,
                    force_regenerate=force_regenerate,
                )
                t0 = time.time()
                base_result = run_experiment(base_config)
                logger.info("  done in %.1fs", time.time() - t0)
                all_rows.extend(_flatten_report(base_result["report"], dname, scenario, seed, beta=None))

                # --- 2. Chebyshev runs, one per (beta, cheb_K) grid cell (oracle already captured above) ---
                # Only force-regenerate once (on the base run above); the materialized snapshots are
                # shared/cached on disk per (dataset, scenario, seed), so every chebyshev config in
                # this grid reuses what the base run just (re)built.
                for beta in betas:
                    for cheb_K in cheb_Ks:
                        run_idx += 1
                        cheb_id = f"{dname}_{scenario}_seed{seed}_cheb_beta{beta}_K{cheb_K}"
                        logger.info("[%s/%s] CHEBYSHEV run: %s / %s / seed=%s / beta=%s / cheb_K=%s",
                                    run_idx, total_runs, dname, scenario, seed, beta, cheb_K)
                        cheb_config = _make_config(
                            dataset_dir, scenario, seed, num_steps,
                            experiment_id=cheb_id, output_dir=output_root,
                            methods=["chebyshev"], include_oracle=False, beta=beta, cheb_K=cheb_K,
                        )
                        t0 = time.time()
                        cheb_result = run_experiment(cheb_config)
                        logger.info("  done in %.1fs", time.time() - t0)
                        all_rows.extend(_flatten_report(cheb_result["report"], dname, scenario, seed,
                                                         beta=beta, cheb_K=cheb_K))

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Sweep complete: %s runs, %s rows written to %s (total %.1fs)",
                total_runs, len(all_rows), output_csv, time.time() - t_sweep_start)
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Run the preliminary CKGE experiment sweep (UMLS + KINSHIP, 3 scenarios, "
                    "5 seeds, beta sweep for Chebyshev) and aggregate results into one CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                         choices=["existing_entities", "new_entities", "mixed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS)
    parser.add_argument("--cheb_Ks", nargs="+", type=int, default=DEFAULT_CHEB_KS,
                         help="Chebyshev propagation order (hops) sweep, typically 1..num_steps.")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--output_root", default="Experiments/continual_prelim")
    parser.add_argument("--output_csv", default="ckge_prelimary_results.csv")
    parser.add_argument("--force_regenerate", action="store_true",
                         help="Rebuild materialized dataset snapshots from scratch even if already "
                              "cached on disk (use this if a cached snapshot folder was built with a "
                              "different --num_steps/--seed and is now stale/incomplete).")
    args = parser.parse_args()

    run_sweep(
        datasets=args.datasets, scenarios=args.scenarios, seeds=args.seeds, betas=args.betas,
        cheb_Ks=args.cheb_Ks, num_steps=args.num_steps, output_root=args.output_root, output_csv=args.output_csv,
        force_regenerate=args.force_regenerate,
    )


if __name__ == "__main__":
    main()
