#!/usr/bin/env python3
"""
Create summary tables (rows=methods, columns=ratios) that show, for each ratio,
how often a method's value is lower than the 'random' baseline across ALL MODELS.

Folder layout (under ROOM_ROOT):
  ROOM_ROOT/
    DistMult/
      add/*.csv
      delete/*.csv
    ComplEx/
      add/*.csv
      delete/*.csv
    Keci/ ...
    Pykeen_MuRE/ ...
    Pykeen_RotatE/ ...
    DeCaL/ ...

We only read CSVs whose filenames END with either:
  - 831769172.csv
  - 2430986565.csv

CSV format assumptions:
  - First row is header
  - First column = methods (we rename to 'method' if needed)
  - Ratio columns include 0.02, 0.04, 0.08 (accepts string/float headers or variants like 'ratio=0.02')
  - There is a baseline row whose method contains 'random' (case-insensitive). Fallbacks: 'rand', 'uniform', 'baseline'

Output:
  - summary_add.csv
  - summary_delete.csv
  - summary_all.csv  (add+delete pooled)

Each cell contains: "lower_count/total (percentage%)"
where:
  - lower_count = number of files (across all models) where method_value < random_value
  - total = number of files where both method_value and random_value exist for that ratio
"""

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict

MODEL_DIRS = ["DistMult", "ComplEx", "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL"]
CONDITIONS = ["add", "delete"]
SUFFIX_IDS = ["831769172", "2430986565"]
RATIOS = [0.02, 0.04, 0.08]
RATIO_LABELS = [f"{r:.2f}" for r in RATIOS]


# ------------------------------- helpers ------------------------------------ #

def list_target_csvs(model_path: str, condition: str) -> List[str]:
    """List CSV files under model_path/condition that end with one of SUFFIX_IDS."""
    target_dir = os.path.join(model_path, condition)
    if not os.path.isdir(target_dir):
        return []
    out = []
    for fn in os.listdir(target_dir):
        if not fn.lower().endswith(".csv"):
            continue
        for sid in SUFFIX_IDS:
            if re.search(rf"{re.escape(sid)}\.csv$", fn):
                out.append(os.path.join(target_dir, fn))
                break
    return sorted(out)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure first column is 'method', trim headers, and strip method strings."""
    df = df.copy()
    if df.shape[1] == 0:
        return df
    first = df.columns[0]
    if str(first).strip().lower() != "method":
        df = df.rename(columns={first: "method"})
    df.columns = [str(c).strip() for c in df.columns]
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).str.strip()
    return df


def detect_baseline_index(df: pd.DataFrame) -> Optional[int]:
    """Return the row index of the baseline (contains 'random'); fallback variants."""
    if "method" not in df.columns:
        return None
    low = df["method"].astype(str).str.casefold()
    hit = low.str.contains("random")
    if hit.any():
        return int(np.where(hit)[0][0])
    variants = {"rand", "uniform", "baseline"}
    hit2 = low.isin(variants)
    if hit2.any():
        return int(np.where(hit2)[0][0])
    return None


def find_ratio_column(df: pd.DataFrame, ratio: float) -> Optional[str]:
    """Find the column corresponding to the given ratio."""
    label = f"{ratio:.2f}"
    if label in df.columns:
        return label
    for c in df.columns:
        cc = str(c).strip()
        if cc == label:
            return c
        try:
            if abs(float(cc) - ratio) < 1e-12:
                return c
        except Exception:
            pass
        if label in cc:  # e.g., "ratio=0.02"
            return c
    return None


# ------------------------------- core logic --------------------------------- #

def collect_counts(room_root: str) -> Dict[str, Dict[Tuple[str, str], List[int]]]:
    """
    Iterate all model folders and both conditions, look at matched CSVs,
    and return counts keyed by condition.

    Returns:
      counts_by_cond: dict like
        {
          "add":    { (method, ratio_label): [lower_count, total] , ... },
          "delete": { (method, ratio_label): [lower_count, total] , ... }
        }
    """
    counts_by_cond: Dict[str, Dict[Tuple[str, str], List[int]]] = {
        "add": defaultdict(lambda: [0, 0]),
        "delete": defaultdict(lambda: [0, 0]),
    }

    models_found = [m for m in MODEL_DIRS if os.path.isdir(os.path.join(room_root, m))]
    if not models_found:
        raise SystemExit(
            f"No expected model folders found under: {room_root}\n"
            f"Expected any of: {MODEL_DIRS}"
        )

    for model in models_found:
        model_path = os.path.join(room_root, model)
        for cond in CONDITIONS:
            files = list_target_csvs(model_path, cond)
            if not files:
                continue
            for fpath in files:
                try:
                    df = pd.read_csv(fpath, header=0)
                except Exception as e:
                    print(f"[WARN] Skipping unreadable CSV: {fpath} ({e})")
                    continue
                df = normalize_columns(df)
                if "method" not in df.columns:
                    print(f"[WARN] No 'method' column: {fpath} — skipping")
                    continue

                base_idx = detect_baseline_index(df)
                if base_idx is None:
                    print(f"[WARN] Baseline 'random' not found: {fpath} — skipping")
                    continue

                for r in RATIOS:
                    col = find_ratio_column(df, r)
                    if col is None:
                        print(f"[WARN] Ratio {r:.2f} not found in {fpath}")
                        continue

                    vals = pd.to_numeric(df[col], errors="coerce")
                    base_val = vals.iloc[base_idx]
                    if pd.isna(base_val):
                        print(f"[WARN] Baseline value NaN for ratio {r:.2f} in {fpath}")
                        continue

                    for i, method in enumerate(df["method"]):
                        if i == base_idx:
                            continue
                        mv = vals.iloc[i]
                        if pd.isna(mv):
                            continue
                        key = (str(method).strip(), f"{r:.2f}")
                        counts_by_cond[cond][key][1] += 1  # total
                        if float(mv) < float(base_val):
                            counts_by_cond[cond][key][0] += 1  # lower_count

    return counts_by_cond


def build_summary_table(counts_for_cond: Dict[Tuple[str, str], List[int]]) -> pd.DataFrame:
    """
    Build a pretty summary pivot: rows=method, cols=ratios, cell="count/total (pct%)".
    """
    # Collect all methods and ensure fixed ratio column order
    methods = sorted({m for (m, _ratio) in counts_for_cond.keys()})
    data = []
    for method in methods:
        row = {"method": method}
        for rlab in RATIO_LABELS:
            cnt, tot = counts_for_cond.get((method, rlab), [0, 0])
            if tot > 0:
                pct = 100.0 * cnt / tot
                row[rlab] = f"{cnt}/{tot} ({pct:.1f}%)"
            else:
                row[rlab] = "0/0 (NA)"
        data.append(row)
    df = pd.DataFrame(data)
    # Ensure columns order: method, 0.02, 0.04, 0.08
    return df[["method"] + RATIO_LABELS]


def save_summary_tables(room_root: str,
                        counts_by_cond: Dict[str, Dict[Tuple[str, str], List[int]]],
                        out_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Save summary_add.csv, summary_delete.csv, and summary_all.csv.
    Returns dict of written paths.
    """
    if out_dir is None:
        out_dir = room_root
    os.makedirs(out_dir, exist_ok=True)

    # Per-condition tables
    add_tbl = build_summary_table(counts_by_cond["add"])
    del_tbl = build_summary_table(counts_by_cond["delete"])

    # Combined (add + delete pooled)
    pooled = defaultdict(lambda: [0, 0])
    for key, v in counts_by_cond["add"].items():
        pooled[key][0] += v[0]
        pooled[key][1] += v[1]
    for key, v in counts_by_cond["delete"].items():
        pooled[key][0] += v[0]
        pooled[key][1] += v[1]
    all_tbl = build_summary_table(pooled)

    paths = {}
    paths["summary_add"] = os.path.join(out_dir, "summary_add.csv")
    paths["summary_delete"] = os.path.join(out_dir, "summary_delete.csv")
    paths["summary_all"] = os.path.join(out_dir, "summary_all.csv")

    add_tbl.to_csv(paths["summary_add"], index=False)
    del_tbl.to_csv(paths["summary_delete"], index=False)
    all_tbl.to_csv(paths["summary_all"], index=False)

    return paths


# --------------------------------- CLI -------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Summarize how often each method < random, across all models.")
    ap.add_argument("room_root", help="Path to the room folder that contains model subfolders.")
    ap.add_argument("--out-dir", default=None,
                    help="Directory to write summary CSVs (default: room_root).")
    args = ap.parse_args()

    room_root = args.room_root
    if not os.path.isdir(room_root):
        raise SystemExit(f"Provided room_root is not a directory: {room_root}")

    counts_by_cond = collect_counts(room_root)
    paths = save_summary_tables(room_root, counts_by_cond, args.out_dir)

    print("\nWrote summary tables:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
