#!/usr/bin/env python3
"""
Scans a 'room' folder with model subfolders:
  ["DistMult","ComplEx","Keci","Pykeen_MuRE","Pykeen_RotatE","DeCaL"]
Each model has 'add' and 'delete' subfolders containing CSVs whose filenames
END with either 831769172.csv or 2430986565.csv.

From each CSV:
- First row is header; first column is methods (renamed to 'method' if needed).
- Extract ratios 0.02, 0.04, 0.08 (accepts columns named 0.02/0.04/0.08 or close variants).
- Detect baseline row whose method contains 'random' (case-insensitive; falls back to rand/uniform/baseline).

For each model × condition(add/delete) × ratio × method:
- Compute paired diffs across matching CSV files: (method - random).
- One-sided sign test (H1: method < random), report p-values at alpha=0.05.
- Also report BH-FDR q-values for reference.

Outputs (written into the provided room root):
- stat_tests_results.csv
- extracted_values_long.csv
"""

import argparse
import os
import re
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

MODEL_DIRS = ["DistMult", "ComplEx", "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL"]
CONDITIONS = ["add", "delete"]
RATIOS = [0.02, 0.04, 0.08]
SUFFIX_IDS_DEFAULT = ["831769172", "2430986565"]  # must be filename-ending before .csv
ALPHA_DEFAULT = 0.05


def list_target_csvs(model_path: str, condition: str, suffix_ids: List[str]) -> List[str]:
    target_dir = os.path.join(model_path, condition)
    if not os.path.isdir(target_dir):
        return []
    out = []
    for fn in os.listdir(target_dir):
        full = os.path.join(target_dir, fn)
        if not os.path.isfile(full) or not fn.lower().endswith(".csv"):
            continue
        for sid in suffix_ids:
            # strict: filename must end with this numeric id before ".csv"
            if re.search(rf"{re.escape(sid)}\.csv$", fn):
                out.append(full)
                break
    return sorted(out)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure the first column is called 'method'
    first = df.columns[0]
    if str(first).strip().lower() != "method":
        df = df.rename(columns={first: "method"})
    # Strip whitespace in headers
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_baseline_index(df: pd.DataFrame) -> Optional[int]:
    if "method" not in df.columns:
        return None
    methods = df["method"].astype(str).str.strip()
    low = methods.str.casefold()
    # Preferred: contains 'random'
    hit = low.str.contains("random")
    if hit.any():
        return int(np.where(hit)[0][0])
    # Fallback variants
    variants = {"rand", "uniform", "baseline"}
    hit2 = low.isin(variants)
    if hit2.any():
        return int(np.where(hit2)[0][0])
    return None


def find_ratio_column(df: pd.DataFrame, ratio: float) -> Optional[str]:
    label = f"{ratio:.2f}"
    # Exact string match first
    if label in df.columns:
        return label
    for c in df.columns:
        cc = str(c).strip()
        if cc == label:
            return c
        # Numeric-parse column header
        try:
            if abs(float(cc) - ratio) < 1e-12:
                return c
        except Exception:
            pass
        # Soft match like "ratio=0.02"
        if label in cc:
            return c
    return None


def one_sided_sign_test_negatives(diffs: List[float]) -> Tuple[Optional[float], int]:
    """One-sided sign test: H0 median(diff)=0 vs H1: more negatives (diff<0)."""
    nz = [d for d in diffs if not pd.isna(d) and d != 0]
    n = len(nz)
    if n == 0:
        return (None, 0)
    k = sum(1 for d in nz if d < 0)  # count negatives
    # p = P[X >= k] for X ~ Binomial(n, 0.5)
    # (upper-tail since we test for "at least this many negatives")
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return (p, n)


def bh_fdr(pvals: List[Optional[float]]) -> List[Optional[float]]:
    """Benjamini–Hochberg FDR. None p-values propagate as None."""
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not indexed:
        return [None] * len(pvals)
    indexed.sort(key=lambda x: x[1])  # ascending p
    m = len(indexed)
    qvals = [None] * len(pvals)
    prev = 1.0
    for rank, (i, p) in enumerate(indexed, start=1):
        q = p * m / rank
        if q > prev:
            q = prev
        prev = q
        qvals[i] = min(q, 1.0)
    return qvals


def main():
    ap = argparse.ArgumentParser(description="Analyze model CSVs vs random baseline.")
    ap.add_argument("room_root", help="Path to the room root folder (contains model subfolders).")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT, help="Significance level for p-values (default: 0.05).")
    ap.add_argument("--suffix-ids", nargs="*", default=SUFFIX_IDS_DEFAULT,
                    help="Filename-ending numeric IDs (before .csv) to include. Default: 831769172 2430986565")
    args = ap.parse_args()

    room_root = args.room_root
    alpha = args.alpha
    suffix_ids = args.suffix_ids

    if not os.path.isdir(room_root):
        raise SystemExit(f"Room root not found or not a directory: {room_root}")

    # Detect which model directories actually exist
    models = [m for m in MODEL_DIRS if os.path.isdir(os.path.join(room_root, m))]
    if not models:
        print("Warning: No expected model folders found in the given room root.")
        print("Expected any of:", MODEL_DIRS)

    raw_records = []
    diff_map: Dict[Tuple[str, str, str, str], List[float]] = {}

    for model in models:
        model_path = os.path.join(room_root, model)
        for cond in CONDITIONS:
            csvs = list_target_csvs(model_path, cond, suffix_ids)
            if not csvs:
                print(f"Note: no matching CSVs found for {model}/{cond}.")
                continue
            for fpath in csvs:
                try:
                    df = pd.read_csv(fpath, header=0)
                except Exception as e:
                    print(f"Skipped unreadable CSV: {fpath} ({e})")
                    continue
                df = normalize_columns(df)
                if "method" not in df.columns:
                    print(f"Skipped (no 'method' column): {fpath}")
                    continue

                base_idx = detect_baseline_index(df)
                if base_idx is None:
                    print(f"Baseline 'random' not found in: {fpath}")
                for r in RATIOS:
                    col = find_ratio_column(df, r)
                    if col is None:
                        print(f"Ratio {r:.2f} not found in: {fpath}")
                        continue
                    vals = pd.to_numeric(df[col], errors="coerce")
                    methods = df["method"].astype(str)
                    # Raw extraction
                    for mname, v in zip(methods, vals):
                        raw_records.append({
                            "model": model,
                            "condition": cond,
                            "file": os.path.basename(fpath),
                            "method": mname,
                            "ratio": f"{r:.2f}",
                            "value": v
                        })
                    # Build diffs vs baseline
                    if base_idx is None:
                        continue
                    base_val = vals.iloc[base_idx]
                    if pd.isna(base_val):
                        print(f"Baseline NaN for ratio {r:.2f} in: {fpath}")
                        continue
                    for i, mname in enumerate(methods):
                        if i == base_idx:
                            continue
                        mv = vals.iloc[i]
                        if pd.isna(mv):
                            continue
                        key = (model, cond, f"{r:.2f}", mname)
                        diff_map.setdefault(key, []).append(float(mv) - float(base_val))

    # Build stats
    stats_rows = []
    for key, diffs in diff_map.items():
        model, cond, rlab, method = key
        diffs = [d for d in diffs if not pd.isna(d)]
        if not diffs:
            continue
        pval, n_nonzero = one_sided_sign_test_negatives(diffs)
        mean_diff = float(np.mean(diffs)) if diffs else float("nan")
        median_diff = float(np.median(diffs)) if diffs else float("nan")
        stats_rows.append({
            "model": model,
            "condition": cond,
            "ratio": rlab,
            "method": method,
            "n_files": len(diffs),
            "n_nonzero_pairs": n_nonzero,
            "mean_diff_method_minus_random": mean_diff,
            "median_diff_method_minus_random": median_diff,
            "p_value_sign_test_one_sided": pval,
            "alpha": alpha,
            "significant_p": bool(pval is not None and pval < alpha),
            "lower_than_random": bool(median_diff < 0),
        })

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        qvals = bh_fdr(stats_df["p_value_sign_test_one_sided"].tolist())
        stats_df["q_value_bh"] = qvals
        stats_df["significant_bh_0.05"] = stats_df["q_value_bh"].apply(lambda q: bool(q is not None and q <= 0.05))
        stats_df["lower_and_significant"] = stats_df.apply(
            lambda r: bool(r["significant_p"] and r["median_diff_method_minus_random"] < 0), axis=1
        )
        stats_df["lower_and_significant_bh"] = stats_df.apply(
            lambda r: bool(r["significant_bh_0.05"] and r["median_diff_method_minus_random"] < 0), axis=1
        )

    raw_df = pd.DataFrame(raw_records)

    out_results = os.path.join(room_root, "stat_tests_results.csv")
    out_raw = os.path.join(room_root, "extracted_values_long.csv")
    stats_df.to_csv(out_results, index=False)
    raw_df.to_csv(out_raw, index=False)

    # Console summary
    print(f"\nWrote: {out_results}  ({len(stats_df)} rows)" if not stats_df.empty else f"\nWrote: {out_results} (EMPTY)")
    print(f"Wrote: {out_raw}  ({len(raw_df)} rows)" if not raw_df.empty else f"Wrote: {out_raw} (EMPTY)")
    if not stats_df.empty:
        print("\nExample rows (results):")
        print(stats_df.head(10).to_string(index=False))
    if not raw_df.empty:
        print("\nExample rows (raw values):")
        print(raw_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
