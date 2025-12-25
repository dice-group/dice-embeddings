import pandas as pd
import glob
import numpy as np
from scipy import stats

from config import DBS, MODELS, RECIPRIOCAL

EPS = 1e-12

for DB in DBS:
    for MODEL in MODELS:
        files = sorted(glob.glob(f"./final_results/{RECIPRIOCAL}/{DB}/{MODEL}/delete/*.csv"))

        if not files:
            print(f"DB: {DB} MODEL: {MODEL} -> no CSV files, skipping.")
            continue

        diffs = []
        pps = []
        rels = []

        for f in files:
            df = pd.read_csv(f)

            if "Deletion Ratios" not in df.columns:
                print(f"Missing 'Deletion Ratios' in {f}, skipping.")
                continue

            df["Deletion Ratios"] = df["Deletion Ratios"].astype(str).str.strip()

            rd_df = df[df["Deletion Ratios"] == "Random"]
            sc_df = df[df["Deletion Ratios"] == "Score"]

            if rd_df.empty or sc_df.empty:
                print(f"No Random/Score row in {f}, skipping.")
                continue

            rd = rd_df.iloc[0]
            sc = sc_df.iloc[0]

            metric_cols = [c for c in df.columns if c != "Deletion Ratios"]

            for col in metric_cols:
                a = pd.to_numeric(sc[col], errors="coerce")  # Score
                b = pd.to_numeric(rd[col], errors="coerce")  # Random
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue

                d = float(a - b)
                diffs.append(d)
                pps.append(d * 100.0)

                if abs(b) > EPS:
                    rels.append((d / float(b)) * 100.0)

        diffs = np.asarray(diffs, float)
        pps   = np.asarray(pps, float)
        rels  = np.asarray(rels, float)

        def wilcoxon_vs_zero(x, name):
            # drop NaNs and zeros (Wilcoxon ignores zeros but will error if too few nonzero)
            x = x[np.isfinite(x)]
            x_nz = x[np.abs(x) > EPS]

            if x_nz.size < 2:
                return f"{name}: n={x_nz.size} (not enough nonzero samples)"

            # two-sided test; use alternative="greater"/"less" if you have a directional hypothesis
            stat, p = stats.wilcoxon(x_nz, zero_method="wilcox", alternative="two-sided")
            return f"{name}: median={np.median(x_nz):.6f}, W={stat:.3f}, p={p:.4g}, n={x_nz.size}"

        print(f"DB={DB} MODEL={MODEL}")
        print("  " + wilcoxon_vs_zero(diffs, "diff (Score-Random)"))
        print("  " + wilcoxon_vs_zero(pps,   "pp diff"))
        print("  " + wilcoxon_vs_zero(rels,  "rel % change"))
