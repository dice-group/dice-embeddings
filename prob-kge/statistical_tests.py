import pandas as pd
import glob
import numpy as np
from scipy import stats

from config import DBS, MODELS, RECIPRIOCAL, PERCENTAGES

EPS = 1e-12

for DB in DBS:
    for MODEL in MODELS:
        files = sorted(glob.glob(f"./final_results/{RECIPRIOCAL}/{DB}/{MODEL}/delete/*.csv"))

        if not files:
            print(f"DB: {DB} MODEL: {MODEL} -> no CSV files found, skipping.")
            continue

        diffs = []          # raw a-b in [0,1] scale
        pp_diffs = []       # percentage points
        rel_changes = []    # relative % change vs b (Random)
        sym_changes = []    # symmetric % difference (signed)

        for f in files:
            df = pd.read_csv(f)

            if "Deletion Ratios" not in df.columns:
                print(f"Missing 'Deletion Ratios' in {f}, skipping file.")
                continue

            df["Deletion Ratios"] = df["Deletion Ratios"].astype(str).str.strip()

            rd_df = df[df["Deletion Ratios"] == "Random"]
            cl_df = df[df["Deletion Ratios"] == "Score"]

            if rd_df.empty or cl_df.empty:
                print(f"No Random/Score row in {f}, skipping file.")
                continue

            rd = rd_df.iloc[0]
            cl = cl_df.iloc[0]

            cols = [c for c in df.columns if c != "Deletion Ratios"]

            for col in cols:
                a = pd.to_numeric(cl[col], errors="coerce")  # Score
                b = pd.to_numeric(rd[col], errors="coerce")  # Random
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue

                d = float(a - b)
                diffs.append(d)
                pp_diffs.append(d * 100.0)

                # relative % change vs baseline b
                if abs(b) > EPS:
                    rel_changes.append((d / float(b)) * 100.0)

                # symmetric % difference (signed)
                denom = (float(a) + float(b)) / 2.0
                if abs(denom) > EPS:
                    sym_changes.append((d / denom) * 100.0)

        # Convert to arrays
        diffs = np.asarray(diffs, dtype=float)
        pp_diffs = np.asarray(pp_diffs, dtype=float)
        rel_changes = np.asarray(rel_changes, dtype=float)
        sym_changes = np.asarray(sym_changes, dtype=float)

        if diffs.size < 2:
            print(f"DB: {DB} MODEL: {MODEL} -> not enough diffs for t-test (n={diffs.size}), skipping.")
            continue

        mean_diff = diffs.mean()
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)

        mean_pp = pp_diffs.mean()
        t_pp, p_pp = stats.ttest_1samp(pp_diffs, 0.0) if pp_diffs.size >= 2 else (np.nan, np.nan)

        mean_rel = rel_changes.mean() if rel_changes.size else np.nan
        t_rel, p_rel = stats.ttest_1samp(rel_changes, 0.0) if rel_changes.size >= 2 else (np.nan, np.nan)

        mean_sym = sym_changes.mean() if sym_changes.size else np.nan
        t_sym, p_sym = stats.ttest_1samp(sym_changes, 0.0) if sym_changes.size >= 2 else (np.nan, np.nan)

        print(
            f"DB: {DB} MODEL: {MODEL} "
            f"| mean_diff: {mean_diff:.6f} (t={t_stat:.3f}, p={p_val:.2g} "
            f"| mean_pp: {mean_pp:.3f}pp (t={t_pp:.3f}, p={p_pp:.2g} "
            f"| mean_rel: {mean_rel:.3f}% (t={t_rel:.3f}, p={p_rel:.2g} "
            f"| mean_sym: {mean_sym:.3f}% (t={t_sym:.3f}, p={p_sym:.2g}"
        )
