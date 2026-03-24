#!/usr/bin/env python3
"""
Comprehensive experiment script for TrinitE model validation.

Set 1 — Alpha ablation
    TrinitE with alpha fixed in [0, 0.1, 0.5, 0.9, 1.0] and one run
    with alpha as a learnable parameter, evaluated on UMLS and KINSHIP.
    Goal: understand how alpha affects Train / Val / Test MRR.

Set 2 — Baseline comparison
    Best alpha (by Val MRR) is chosen per dataset, then TrinitE is
    compared against TransE, DistMult, ComplEx, Keci, DeCaL.

Usage:
    python examples/trinite_experiments.py

    # Override number of epochs or embedding dim:
    python examples/trinite_experiments.py --num_epochs 200 --embedding_dim 30
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Experiment-wide defaults
# ---------------------------------------------------------------------------
DATASETS        = ["KGs/UMLS", "KGs/KINSHIP"]
ALPHA_VALUES    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9, 1.0]   # fixed; None ⇒ learnable
BASELINE_MODELS = ["TransE", "DistMult", "ComplEx", "DualE","Keci", "DeCaL"]
STORAGE_ROOT    = "Experiments/TrinitE_Experiments"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: str,
    model: str,
    num_epochs: int,
    embedding_dim: int,
    run_dir: str,
    extra_args: list[str] | None = None,
) -> dict:
    """
    Execute a single ``dicee`` training run and return the report dict.

    Parameters
    ----------
    dataset      : path to the KG folder, e.g. 'KGs/UMLS'
    model        : model name, e.g. 'TrinitE'
    num_epochs   : training epochs
    embedding_dim: embedding dimensionality
    run_dir      : directory where results are stored
    extra_args   : additional CLI flags (e.g. ['--alpha', '0.5'])
    """
    dataset_name = Path(dataset).name
    label        = Path(run_dir).name

    cmd = [
        sys.executable, "-m", "dicee",
        "--dataset_dir",          dataset,
        "--model",                model,
        "--trainer",              "PL",
        "--scoring_technique",    "KvsAll",
        "--eval_model",           "train_val_test",
        "--num_epochs",           str(num_epochs),
        "--embedding_dim",        str(embedding_dim),
        "--path_to_store_single_run", run_dir,
    ]
    if extra_args:
        cmd += extra_args

    print(f"\n{'='*65}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Model   : {model}  |  label: {label}")
    if extra_args:
        print(f"  Extra   : {' '.join(extra_args)}")
    print(f"{'='*65}")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[WARNING] Run exited with code {result.returncode}")

    report_path = os.path.join(run_dir, "report.json")
    if os.path.exists(report_path):
        with open(report_path) as fh:
            return json.load(fh)
    print(f"[WARNING] report.json not found at {report_path}")
    return {}


def mrr(report: dict, split: str) -> float:
    """Return MRR for 'Train', 'Val', or 'Test' split, or NaN if missing."""
    return report.get(split, {}).get("MRR", math.nan)


def fmt(v: float) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "  N/A "


def print_table(title: str, headers: list[str], rows: list[list]) -> None:
    col_widths = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print(f"\n{'='*65}")
    print(f"  {title}")
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)
    for row in rows:
        print(row_fmt.format(*[str(c) for c in row]))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TrinitE comprehensive experiments")
    parser.add_argument("--num_epochs",    type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=60,
                        help="Must be divisible by 3 for TrinitE (default: 60)")
    cli = parser.parse_args()

    num_epochs    = cli.num_epochs
    embedding_dim = cli.embedding_dim

    if embedding_dim % 3 != 0:
        sys.exit(f"[ERROR] embedding_dim must be divisible by 3 for TrinitE, got {embedding_dim}")

    os.makedirs(STORAGE_ROOT, exist_ok=True)

    # -----------------------------------------------------------------------
    # SET 1 — Alpha ablation
    # # -----------------------------------------------------------------------
    print("\n" + "#" * 65)
    print("  SET 1: Alpha Ablation — TrinitE on UMLS and KINSHIP")
    print("#" * 65)

    # alpha_label → list of alpha CLI flags (empty list = learnable)
    alpha_configs: list[tuple[str, list[str]]] = [
        (str(a), ["--alpha", str(a)]) for a in ALPHA_VALUES
    ] + [("learnable", [])]

    # {dataset_name: {alpha_label: report}}
    set1_results: dict[str, dict[str, dict]] = {}

    for dataset in DATASETS:
        ds_name = Path(dataset).name
        set1_results[ds_name] = {}

        for alpha_label, extra in alpha_configs:
            run_dir = os.path.join(
                STORAGE_ROOT, ds_name, f"TrinitE_alpha_{alpha_label}"
            )
            report = run_experiment(
                dataset=dataset,
                model="TrinitE",
                num_epochs=num_epochs,
                embedding_dim=embedding_dim,
                run_dir=run_dir,
                extra_args=extra or None,
            )
            set1_results[ds_name][alpha_label] = report

    # Print Set 1 tables
    alpha_labels = [str(a) for a in ALPHA_VALUES] + ["learnable"]
    for ds_name, res in set1_results.items():
        rows = [
            [al, fmt(mrr(res[al], "Train")), fmt(mrr(res[al], "Val")), fmt(mrr(res[al], "Test"))]
            for al in alpha_labels
        ]
        print_table(
            f"TrinitE Alpha Ablation — {ds_name}",
            ["Alpha", "Train MRR", "Val MRR", "Test MRR"],
            rows,
        )

    # -----------------------------------------------------------------------
    # Pick best alpha per dataset (by Val MRR)
    # -----------------------------------------------------------------------
    best_alpha: dict[str, str] = {}
    print("\n  Best alpha by Val MRR:")
    for ds_name, res in set1_results.items():
        best = max(alpha_labels, key=lambda al: mrr(res.get(al, {}), "Val"))
        best_alpha[ds_name] = best
        print(f"    {ds_name}: alpha = {best}  (Val MRR = {fmt(mrr(res[best], 'Val'))})")

    # -----------------------------------------------------------------------
    # SET 2 — Baseline comparison
    # -----------------------------------------------------------------------
    print("\n" + "#" * 65)
    print("  SET 2: Baseline Comparison (best alpha vs baselines)")
    print("#" * 65)

    # {dataset_name: {model_label: report}}
    set2_results: dict[str, dict[str, dict]] = {}

    for dataset in DATASETS:
        ds_name = Path(dataset).name
        set2_results[ds_name] = {}
        ba = best_alpha[ds_name]

        # TrinitE with best alpha
        trinite_extra = ["--alpha", ba] if ba != "learnable" else None
        run_dir = os.path.join(
            STORAGE_ROOT, ds_name, f"TrinitE_best_alpha_{ba}"
        )
        set2_results[ds_name]["TrinitE"] = run_experiment(
            dataset=dataset,
            model="TrinitE",
            num_epochs=num_epochs,
            embedding_dim=embedding_dim,
            run_dir=run_dir,
            extra_args=trinite_extra,
        )

        # Baseline models (use same embedding_dim; all baselines support
        # dimensions divisible by 3 since 27 is divisible by 2 only for
        # ComplEx / DistMult — raise to nearest even if needed)
        baseline_dim = embedding_dim if embedding_dim % 2 == 0 else embedding_dim + 1
        for model in BASELINE_MODELS:
            run_dir = os.path.join(STORAGE_ROOT, ds_name, f"{model}_baseline")
            set2_results[ds_name][model] = run_experiment(
                dataset=dataset,
                model=model,
                num_epochs=num_epochs,
                embedding_dim=baseline_dim,
                run_dir=run_dir,
            )

    # Print Set 2 tables
    all_models = ["TrinitE"] + BASELINE_MODELS
    for ds_name, res in set2_results.items():
        rows = [
            [m, fmt(mrr(res[m], "Train")), fmt(mrr(res[m], "Val")), fmt(mrr(res[m], "Test"))]
            for m in all_models
        ]
        print_table(
            f"Baseline Comparison — {ds_name}  (TrinitE alpha={best_alpha[ds_name]})",
            ["Model", "Train MRR", "Val MRR", "Test MRR"],
            rows,
        )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "#" * 65)
    print("  SUMMARY")
    print("#" * 65)
    for ds_name in [Path(d).name for d in DATASETS]:
        ba = best_alpha.get(ds_name, "?")
        trinite_test = fmt(mrr(set2_results.get(ds_name, {}).get("TrinitE", {}), "Test"))
        print(f"  {ds_name}: best alpha = {ba}, TrinitE Test MRR = {trinite_test}")
        for m in BASELINE_MODELS:
            bm_test = fmt(mrr(set2_results.get(ds_name, {}).get(m, {}), "Test"))
            print(f"    vs {m:<12}: Test MRR = {bm_test}")
    print()
    print("All experiment results stored under:", STORAGE_ROOT)


if __name__ == "__main__":
    main()
