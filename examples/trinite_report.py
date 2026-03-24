#!/usr/bin/env python3
"""
Report and visualise TrinitE experiment results.

Reads from  Experiments/TrinitE_Experiments/<DATASET>/<RUN>/eval_report.json

Produces
--------
1. One plot per dataset — alpha (x-axis) vs MRR (y-axis), with separate
   lines for Train / Val / Test.  The "learnable" alpha run is shown as a
   horizontal dashed reference line.

2. One LaTeX-style table per dataset — baseline models vs TrinitE (best
   alpha), showing H@1 / H@3 / H@10 / MRR for the Test split.

All figures are saved to  Experiments/TrinitE_Experiments/figures/
"""

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration – mirrors the directory names produced by trinite_experiments.py
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("Experiments/TrinitE_Experiments")
FIGURES_DIR  = RESULTS_ROOT / "figures"
DATASETS     = ["UMLS", "KINSHIP"]

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9, 1.0]   # fixed-alpha runs

BASELINES = ["TransE", "DistMult", "ComplEx", "DeCaL", "DualE", "Keci"]

METRICS   = ["H@1", "H@3", "H@10", "MRR"]
SPLITS    = ["Train", "Val", "Test"]

# Colour / style choices
SPLIT_STYLES = {
    "Train": dict(color="#2196F3", marker="o", linestyle="-",  label="Train"),
    "Val":   dict(color="#FF9800", marker="s", linestyle="--", label="Val"),
    "Test":  dict(color="#4CAF50", marker="^", linestyle="-.", label="Test"),
}

# Bar colours for grouped baseline chart
BAR_COLORS = {"Train": "#2196F3", "Val": "#FF9800", "Test": "#4CAF50"}

# Global matplotlib defaults — research-paper quality
plt.rcParams.update({
    "font.size":          16,
    "axes.titlesize":     20,
    "axes.labelsize":     18,
    "xtick.labelsize":    15,
    "ytick.labelsize":    15,
    "legend.fontsize":    14,
    "legend.title_fontsize": 14,
    "figure.titlesize":   22,
    "lines.linewidth":    2.5,
    "lines.markersize":   9,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_report(path: Path) -> dict:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def get(report: dict, split: str, metric: str) -> float:
    return report.get(split, {}).get(metric, math.nan)


def run_dir(dataset: str, name: str) -> Path:
    return RESULTS_ROOT / dataset / name


def alpha_run_name(alpha) -> str:
    return f"TrinitE_alpha_{alpha}"


def best_alpha_for_dataset(dataset: str) -> tuple[float | str, dict]:
    """Return (alpha_label, report) with highest Val MRR among fixed-alpha runs."""
    best_label, best_report, best_mrr = None, {}, -math.inf
    for a in ALPHA_VALUES:
        label  = alpha_run_name(a)
        report = load_report(run_dir(dataset, label) / "eval_report.json")
        v = get(report, "Val", "MRR")
        if not math.isnan(v) and v > best_mrr:
            best_mrr   = v
            best_label = a
            best_report = report
    return best_label, best_report


# ---------------------------------------------------------------------------
# Plot 1 — Alpha ablation (one per dataset)
# ---------------------------------------------------------------------------

def make_alpha_plot() -> None:
    """One figure per dataset — alpha (x-axis) vs MRR (y-axis), Train/Val/Test lines."""
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(8, 6))

        x_vals: list[float] = []
        split_mrrs: dict[str, list[float]] = {s: [] for s in SPLITS}

        for a in ALPHA_VALUES:
            report = load_report(run_dir(dataset, alpha_run_name(a)) / "eval_report.json")
            if report:
                x_vals.append(a)
                for s in SPLITS:
                    split_mrrs[s].append(get(report, s, "MRR"))

        # Fixed-alpha lines
        for split, mrrs in split_mrrs.items():
            style = SPLIT_STYLES[split]
            ax.plot(x_vals, mrrs, **style, zorder=3)

        # Learnable-alpha horizontal reference lines
        learnable = load_report(run_dir(dataset, "TrinitE_alpha_learnable") / "eval_report.json")
        if learnable:
            for split in SPLITS:
                v = get(learnable, split, "MRR")
                if not math.isnan(v):
                    ax.axhline(
                        y=v,
                        color=SPLIT_STYLES[split]["color"],
                        linestyle=":",
                        linewidth=2.0,
                        alpha=0.80,
                        zorder=2,
                    )
                    ax.annotate(
                        f"learnable {split}: {v:.3f}",
                        xy=(x_vals[-1], v),
                        xytext=(8, 4),
                        textcoords="offset points",
                        fontsize=13,
                        color=SPLIT_STYLES[split]["color"],
                        va="bottom",
                    )

        ax.set_title(f"TrinitE — Effect of α on MRR ({dataset})", fontweight="bold", pad=12)
        ax.set_xlabel("Alpha (α)")
        ax.set_ylabel("MRR")
        ax.set_xticks(ALPHA_VALUES)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_facecolor("#FAFAFA")
        fig.tight_layout()

        stem = f"alpha_ablation_mrr_{dataset}"
        for ext in ("pdf", "png"):
            out = FIGURES_DIR / f"{stem}.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=200)
            print(f"  Saved: {out}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Table — Baseline comparison (per dataset)
# ---------------------------------------------------------------------------

def build_comparison_table(dataset: str) -> None:
    best_label, _ = best_alpha_for_dataset(dataset)

    # TrinitE with best alpha — prefer the dedicated "best_alpha" run if it
    # exists, otherwise fall back to the ablation run with that alpha.
    trinite_dir = run_dir(dataset, f"TrinitE_best_alpha_{best_label}")
    if not (trinite_dir / "eval_report.json").exists():
        trinite_dir = run_dir(dataset, alpha_run_name(best_label))
    trinite_report = load_report(trinite_dir / "eval_report.json")

    rows: list[dict] = []

    # Baselines
    for model in BASELINES:
        report = load_report(run_dir(dataset, f"{model}_baseline") / "eval_report.json")
        if report:
            rows.append({"model": model, "report": report})

    # TrinitE (best alpha)
    rows.append({"model": f"TrinitE (α={best_label})", "report": trinite_report})

    # ---- Print table -------------------------------------------------------
    col_w  = 20
    metric_w = 8

    header_model   = "Model".ljust(col_w)
    header_metrics = "  ".join(m.rjust(metric_w) for m in METRICS)
    separator      = "-" * (col_w + 2 + len(header_metrics))

    print(f"\n{'='*65}")
    print(f"  Baseline Comparison — {dataset} (Test split)")
    print(f"  Best TrinitE α = {best_label}")
    print(f"{'='*65}")
    print(f"{header_model}  {header_metrics}")
    print(separator)

    for row in rows:
        name   = row["model"].ljust(col_w)
        values = "  ".join(
            f"{get(row['report'], 'Test', m):.4f}".rjust(metric_w)
            if not math.isnan(get(row["report"], "Test", m))
            else "   N/A  "
            for m in METRICS
        )
        marker = "  ◀ TrinitE" if "TrinitE" in row["model"] else ""
        print(f"{name}  {values}{marker}")

    print(separator)


# ---------------------------------------------------------------------------
# Plot 2 — Baseline grouped bar-chart (Train / Val / Test MRR)
# ---------------------------------------------------------------------------

def make_baseline_plot() -> None:
    """One figure per dataset — grouped bars (Train/Val/Test) per model."""
    for dataset in DATASETS:
        best_label, _ = best_alpha_for_dataset(dataset)

        # Collect model names and MRR values per split
        model_names: list[str] = []
        split_vals: dict[str, list[float]] = {s: [] for s in SPLITS}

        for model in BASELINES:
            report = load_report(run_dir(dataset, f"{model}_baseline") / "eval_report.json")
            if not report:
                continue
            model_names.append(model)
            for s in SPLITS:
                split_vals[s].append(get(report, s, "MRR"))

        # TrinitE best alpha
        trinite_dir = run_dir(dataset, f"TrinitE_best_alpha_{best_label}")
        if not (trinite_dir / "eval_report.json").exists():
            trinite_dir = run_dir(dataset, alpha_run_name(best_label))
        report = load_report(trinite_dir / "eval_report.json")
        if report:
            model_names.append(f"TrinitE (α={best_label})")
            for s in SPLITS:
                split_vals[s].append(get(report, s, "MRR"))

        n_models = len(model_names)
        n_splits = len(SPLITS)
        bar_w    = 0.22
        x        = np.arange(n_models)
        offsets  = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * bar_w

        fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 7))

        for offset, split in zip(offsets, SPLITS):
            vals  = split_vals[split]
            is_trinite = ["TrinitE" in m for m in model_names]
            edge_colors = ["#B71C1C" if t else "white" for t in is_trinite]
            lwidths     = [1.8 if t else 0.6 for t in is_trinite]
            bars = ax.bar(
                x + offset, vals,
                width=bar_w,
                color=BAR_COLORS[split],
                edgecolor=edge_colors,
                linewidth=lwidths,
                label=split,
                zorder=3,
            )
            # Value labels
            for bar, val in zip(bars, vals):
                if not math.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.004,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=11, rotation=90,
                    )

        all_vals = [v for lst in split_vals.values() for v in lst if not math.isnan(v)]
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylabel("MRR")
        ax.set_title(
            f"Baseline Comparison — MRR ({dataset})\n"
            f"Best TrinitE α = {best_label}",
            fontweight="bold", pad=12,
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_ylim(0, max(all_vals) * 1.18 if all_vals else 1.0)
        ax.legend(title="Split", loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_facecolor("#FAFAFA")
        fig.tight_layout()

        stem = f"baseline_comparison_mrr_{dataset}"
        for ext in ("pdf", "png"):
            out = FIGURES_DIR / f"{stem}.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=200)
            print(f"  Saved: {out}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 65)
    print("  TrinitE Results Report")
    print("#" * 65)

    # ---- Alpha ablation tables (all splits) --------------------------------
    for dataset in DATASETS:
        print(f"\n{'='*65}")
        print(f"  Alpha Ablation — {dataset}")
        print(f"{'='*65}")
        header = f"{'Alpha':<12}" + "".join(
            f"  {'Train':>8}  {'Val':>8}  {'Test':>8}"
        )
        print(header)
        print("-" * len(header))
        for a in ALPHA_VALUES:
            report = load_report(run_dir(dataset, alpha_run_name(a)) / "eval_report.json")
            tr  = f"{get(report, 'Train', 'MRR'):.4f}" if report else "  N/A  "
            val = f"{get(report,   'Val', 'MRR'):.4f}" if report else "  N/A  "
            tst = f"{get(report,  'Test', 'MRR'):.4f}" if report else "  N/A  "
            print(f"{str(a):<12}  {tr:>8}  {val:>8}  {tst:>8}")
        # Learnable run
        report = load_report(run_dir(dataset, "TrinitE_alpha_learnable") / "eval_report.json")
        tr  = f"{get(report, 'Train', 'MRR'):.4f}" if report else "  N/A  "
        val = f"{get(report,   'Val', 'MRR'):.4f}" if report else "  N/A  "
        tst = f"{get(report,  'Test', 'MRR'):.4f}" if report else "  N/A  "
        print(f"{'learnable':<12}  {tr:>8}  {val:>8}  {tst:>8}")

    # ---- Baseline comparison tables ----------------------------------------
    for dataset in DATASETS:
        build_comparison_table(dataset)

    # ---- Figures ------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  Generating figures …")
    make_alpha_plot()
    make_baseline_plot()

    print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
