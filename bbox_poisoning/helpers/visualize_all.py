#!/usr/bin/env python3
"""
Compute per-ratio mean/std across multiple runs for specified stochastic methods,
keep deterministic methods from run 0 only, and generate box plots for each
stochastic method across triple injection ratios.

- Accepts both "wide" CSVs (first column = method, remaining columns = ratios like '0.01', '0.05', ...)
  and "long" CSVs with columns similar to: Method, TripleInjectionRatio, Value (metric).
- Uses matplotlib (no seaborn), one chart per figure, no explicit colors.
- No nested functions (per user preference).

Usage:
    python compute_stats_and_boxplots.py --folder /path/to/csvs \
        --stochastic-methods Random Low_Scores High_Closeness High_Gradients \
        --output-dir /path/to/out

Defaults:
    folder       = current working directory
    output_dir   = ./outputs
    stochastic   = ["Random", "Low_Scores", "High_Closeness", "High_Gradients"]
"""

import os
import re
import argparse
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Helpers (module scope) -----------------------------

def list_csv_files(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(".csv"):
            files.append(os.path.join(folder, name))
    files.sort()
    return files


def extract_run_index_from_filename(filename: str) -> Optional[int]:
    # Matches "-0.csv", "_0.csv" or any digits just before .csv
    m = re.search(r"[-_](\d+)\.csv$", os.path.basename(filename))
    if m:
        return int(m.group(1))
    return None


def is_numeric_like_string(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def parse_ratio_value(v) -> float:
    """
    Parse a ratio given as '0.05', '5%', '5', etc. into [0, 1].
    If the numeric value is >1, treat it as a percentage and divide by 100.
    Returns np.nan on failure.
    """
    if pd.isna(v):
        return np.nan
    try:
        text = str(v).strip().replace("%", "")
        x = float(text)
    except Exception:
        return np.nan
    if x > 1.0:
        x = x / 100.0
    return x


def detect_wide_format(df: pd.DataFrame) -> bool:
    # "Wide" if there are 2+ columns whose names look numeric-ish ratios
    numeric_name_cols = [c for c in df.columns if is_numeric_like_string(str(c).strip().replace("%", ""))]
    return len(numeric_name_cols) >= 2


def get_method_col_for_wide(df: pd.DataFrame) -> str:
    # Heuristic: the first non-numeric-name column is the method column
    for c in df.columns:
        if not is_numeric_like_string(str(c).strip().replace("%", "")):
            return c
    # Fallback: first column
    return df.columns[0]


def melt_wide_to_long(df: pd.DataFrame, method_col: Optional[str] = None) -> pd.DataFrame:
    """
    Convert wide format into long with standardized columns:
    Method | TripleInjectionRatio | Value
    """
    dfc = df.copy()
    if method_col is None:
        method_col = get_method_col_for_wide(dfc)
    ratio_cols = [c for c in dfc.columns if c != method_col]

    long_df = dfc.melt(id_vars=[method_col], value_vars=ratio_cols,
                       var_name="TripleInjectionRatio", value_name="Value")

    long_df["TripleInjectionRatio"] = long_df["TripleInjectionRatio"].map(parse_ratio_value)
    long_df.rename(columns={method_col: "Method"}, inplace=True)
    long_df["Method"] = long_df["Method"].astype(str).str.strip()
    return long_df[["Method", "TripleInjectionRatio", "Value"]]


def standardize_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to standardize an already-long dataframe into columns:
    Method | TripleInjectionRatio | Value
    """
    dfc = df.copy()
    # Guess columns
    method_candidates = [c for c in dfc.columns if any(k in c.lower() for k in ["method", "strategy", "attack", "technique"])]
    ratio_candidates = [c for c in dfc.columns if "ratio" in c.lower()]
    metric_candidates = [c for c in dfc.columns if c not in method_candidates + ratio_candidates]

    method_col = method_candidates[0] if method_candidates else dfc.columns[0]
    ratio_col = ratio_candidates[0] if ratio_candidates else dfc.columns[1]

    metric_col = None
    preferred_metrics = ["TestMRR", "MRR", "ValidMRR", "Score", "Accuracy", "MeanRank", "MR"]
    for cand in preferred_metrics:
        if cand in dfc.columns and pd.api.types.is_numeric_dtype(dfc[cand]):
            metric_col = cand
            break
    if metric_col is None:
        # fallback: first numeric column not already used
        for c in metric_candidates:
            if pd.api.types.is_numeric_dtype(dfc[c]):
                metric_col = c
                break
        if metric_col is None and metric_candidates:
            metric_col = metric_candidates[0]

    dfc = dfc.rename(columns={method_col: "Method", ratio_col: "TripleInjectionRatio", metric_col: "Value"})
    dfc["Method"] = dfc["Method"].astype(str).str.strip()
    dfc["TripleInjectionRatio"] = dfc["TripleInjectionRatio"].map(parse_ratio_value)
    return dfc[["Method", "TripleInjectionRatio", "Value"]]


def load_dataframe_from_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        # Fallback to semicolon if needed
        return pd.read_csv(path, sep=";")


def load_all_results_long(folder: str) -> pd.DataFrame:
    """
    Load all CSVs from folder and return a single long-format dataframe with columns:
    Method | TripleInjectionRatio | Value | __source_file | __run_index
    """
    files = list_csv_files(folder)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = []
    for f in files:
        df = load_dataframe_from_csv(f)
        if detect_wide_format(df):
            long_df = melt_wide_to_long(df)
        else:
            long_df = standardize_long(df)

        long_df = long_df.copy()
        long_df["__source_file"] = os.path.basename(f)
        long_df["__run_index"] = extract_run_index_from_filename(f)
        frames.append(long_df)

    combined_long = pd.concat(frames, ignore_index=True)
    return combined_long


def compute_stats_long(
    long_df: pd.DataFrame,
    stochastic_methods: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two dataframes:
      - agg_stoch: Method, TripleInjectionRatio, N, Value_mean, Value_std, Deterministic=False
      - agg_det:   Method, TripleInjectionRatio, N, Value, Deterministic=True   (from run 0 only)
    """
    df = long_df.copy()
    df = df[df["TripleInjectionRatio"].notna() & df["Value"].notna()]

    is_stochastic = df["Method"].isin(stochastic_methods)

    # Stochastic -> mean/std over runs
    stoch_df = df[is_stochastic].copy()
    agg_stoch = (
        stoch_df.groupby(["Method", "TripleInjectionRatio"])["Value"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "N", "mean": "Value_mean", "std": "Value_std"})
    )
    if not agg_stoch.empty:
        agg_stoch["Deterministic"] = False

    # Deterministic -> only from run 0
    det_df = df[~is_stochastic & (df["__run_index"] == 0)].copy()
    agg_det = (
        det_df.groupby(["Method", "TripleInjectionRatio"])["Value"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "N", "mean": "Value"})
    )
    if not agg_det.empty:
        agg_det["Deterministic"] = True

    return agg_stoch, agg_det


def save_summary_tables(
    agg_stoch: pd.DataFrame,
    agg_det: pd.DataFrame,
    out_dir: str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    stoch_path = os.path.join(out_dir, "stochastic_stats.csv")
    det_path = os.path.join(out_dir, "deterministic_values.csv")

    if not agg_stoch.empty:
        agg_stoch.sort_values(["Method", "TripleInjectionRatio"], inplace=True)
        agg_stoch.to_csv(stoch_path, index=False)
    else:
        pd.DataFrame(columns=["Method", "TripleInjectionRatio", "N", "Value_mean", "Value_std", "Deterministic"]).to_csv(stoch_path, index=False)

    if not agg_det.empty:
        agg_det.sort_values(["Method", "TripleInjectionRatio"], inplace=True)
        agg_det.to_csv(det_path, index=False)
    else:
        pd.DataFrame(columns=["Method", "TripleInjectionRatio", "N", "Value", "Deterministic"]).to_csv(det_path, index=False)

    return stoch_path, det_path



def make_grouped_boxplot_all_methods(
    long_df: pd.DataFrame,
    include_methods: Optional[List[str]],
    save_path: str,
    ylabel: str = "Value"
) -> str:
    """
    Create ONE figure that shows grouped box plots:
      - X-axis: Triple Injection Ratio
      - For each ratio: one box per method (across runs)
      - Methods are offset around each ratio center.
    Returns the saved file path.
    """
    df = long_df.copy()
    df = df[df["TripleInjectionRatio"].notna() & df["Value"].notna()]

    if include_methods is None or len(include_methods) == 0:
        methods = sorted(df["Method"].unique().tolist())
    else:
        methods = include_methods

    ratios = sorted(df["TripleInjectionRatio"].unique().tolist())
    if len(ratios) == 0 or len(methods) == 0:
        raise ValueError("No data to plot. Check your inputs.")

    # Positions: for each ratio index i, place k boxes (k = len(methods)) around i
    k = len(methods)
    centers = list(range(len(ratios)))
    group_width = 0.8  # total width allotted per ratio
    if k > 0:
        step = group_width / max(k, 1)
    else:
        step = group_width
    box_width = step * 0.8

    plt.figure()
    for j, method in enumerate(methods):
        data_for_method = []
        positions = []
        for i, r in enumerate(ratios):
            sub = df[(df["Method"] == method) & (df["TripleInjectionRatio"] == r)]["Value"].dropna().values
            data_for_method.append(sub)
            # position: center + offset
            offset = (j - (k - 1) / 2.0) * step
            positions.append(centers[i] + offset)

        plt.boxplot(
            data_for_method,
            positions=positions,
            widths=box_width,
            showmeans=True
        )

    # X ticks at ratio centers
    plt.xticks(centers, [f"{r:.2f}" for r in ratios])
    plt.xlabel("Triple Injection Ratio")
    plt.ylabel(ylabel)
    plt.title("Grouped Box Plots by Triple Injection Ratio and Method")

    # Build a legend with default styles (no explicit colors)
    handles = []
    labels = []
    for method in methods:
        h, = plt.plot([], [], label=method)  # dummy handle
        handles.append(h)
        labels.append(method)
    plt.legend(handles, labels, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path



def make_atomic_boxplots_per_method_ratio(
    long_df: pd.DataFrame,
    include_methods: Optional[List[str]],
    save_dir: str,
    ylabel: str = "Value"
) -> int:
    """
    Create individual box plots for every (method, ratio) pair.
    Each figure contains a single box showing the distribution across runs.
    Returns the number of figures saved.
    """
    df = long_df.copy()
    df = df[df["TripleInjectionRatio"].notna() & df["Value"].notna()]

    if include_methods is None or len(include_methods) == 0:
        methods = sorted(df["Method"].unique().tolist())
    else:
        methods = include_methods

    ratios = sorted(df["TripleInjectionRatio"].unique().tolist())
    if len(ratios) == 0 or len(methods) == 0:
        return 0

    out_dir = os.path.join(save_dir, "atomic")  # nested folder
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for method in methods:
        for r in ratios:
            vals = df[(df["Method"] == method) & (df["TripleInjectionRatio"] == r)]["Value"].dropna().values
            if vals.size == 0:
                continue
            plt.figure()
            plt.boxplot([vals], labels=[f"{r:.2f}"], showmeans=True)
            plt.xlabel("Triple Injection Ratio")
            plt.ylabel(ylabel)
            plt.title(f"{method} — ratio {r:.2f}")
            plt.tight_layout()
            safe_m = method.replace(" ", "_")
            out_path = os.path.join(out_dir, f"box_{safe_m}_ratio_{r:.2f}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            count += 1

    return count


def make_boxplots_for_methods(
    long_df: pd.DataFrame,
    stochastic_methods: List[str],
    save_dir: str,
    ylabel: str = "Value",
) -> List[str]:
    """
    Generate one box plot per method, with one box per triple injection ratio.
    Returns a list of saved plot file paths.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = long_df.copy()
    df = df[df["TripleInjectionRatio"].notna() & df["Value"].notna()]

    plot_paths: List[str] = []

    for method in stochastic_methods:
        sub = df[df["Method"] == method].copy()
        if sub.empty:
            continue

        ratios = sorted(sub["TripleInjectionRatio"].unique())
        data = [sub[sub["TripleInjectionRatio"] == r]["Value"].dropna().values for r in ratios]

        plt.figure()
        plt.boxplot(data, labels=[f"{r:.2f}" for r in ratios], showmeans=True)
        plt.title(f"{method} — {ylabel} by Triple Injection Ratio")
        plt.xlabel("Triple Injection Ratio")
        plt.ylabel(ylabel)
        plt.tight_layout()

        safe_method = method.replace(" ", "_")
        out_path = os.path.join(save_dir, f"boxplot_{safe_method}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        plot_paths.append(out_path)

    return plot_paths


def main(folder: str,
         stochastic_methods: Optional[List[str]] = None,
         output_dir: str = "outputs",
         plots_subdir: str = "plots",
         grouped_filename: str = "grouped_boxplots_all_methods.png",
         include_methods_for_grouped: Optional[List[str]] = None,
         also_make_atomic: bool = True) -> Dict[str, str]:
    if stochastic_methods is None:
        stochastic_methods = ["Random", "Low_Scores", "High_Closeness", "High_Gradients"]

    long_df = load_all_results_long(folder)
    agg_stoch, agg_det = compute_stats_long(long_df, stochastic_methods)

    os.makedirs(output_dir, exist_ok=True)
    stoch_path, det_path = save_summary_tables(agg_stoch, agg_det, output_dir)

    plots_dir = os.path.join(output_dir, plots_subdir)
    plot_paths = make_boxplots_for_methods(long_df, stochastic_methods, plots_dir)

    grouped_path = os.path.join(plots_dir, grouped_filename)
    make_grouped_boxplot_all_methods(long_df, include_methods_for_grouped, grouped_path)

    atomic_count = 0
    if also_make_atomic:
        atomic_count = make_atomic_boxplots_per_method_ratio(long_df, include_methods_for_grouped, plots_dir)

    summary_info = {
        "stochastic_stats_csv": os.path.abspath(stoch_path),
        "deterministic_values_csv": os.path.abspath(det_path),
        "num_plots_generated": str(len(plot_paths)),
        "plots_dir": os.path.abspath(plots_dir),
        "grouped_boxplot": os.path.abspath(grouped_path),
        "atomic_plots_dir": os.path.abspath(os.path.join(plots_dir, "atomic")),
        "atomic_plots_count": str(atomic_count),
    }
    return summary_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute stats and plot boxplots for KGE experiment runs.")
    parser.add_argument("--folder", type=str, default=".", help="Folder containing CSV files for multiple runs.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save CSV summaries and plots.")
    parser.add_argument("--stochastic-methods", nargs="*", default=["Random", "Low_Scores", "High_Closeness", "High_Gradients"],
                        help="Methods to average across runs (mean/std). Others are treated as deterministic from run 0.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    info = main(args.folder, args.stochastic_methods, args.output_dir)
    print("Stochastic stats CSV:", info["stochastic_stats_csv"])
    print("Deterministic values CSV:", info["deterministic_values_csv"])
    print("Plots dir:", info["plots_dir"])
    print("Num plots generated:", info["num_plots_generated"])