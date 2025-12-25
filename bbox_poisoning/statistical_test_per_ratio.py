import glob
import numpy as np
import pandas as pd
from scipy import stats
import glob
import numpy as np
import pandas as pd
from scipy import stats

import glob
import numpy as np
import pandas as pd
from scipy import stats

DBS = ["UMLS", "KINSHIP"]
models = ["DistMult", "ComplEx", "QMult"]

for DB in DBS:
    print(f"\n=== DB: {DB} ===")
    for model in models:
        files = sorted(glob.glob(f"./final_results/without_recipriocal/{DB}/{model}/delete/*.csv"))

        if not files:
            print(f"model: {model}  (no files found)")
            continue

        ratio_diffs = None  # dict: ratio -> list of diffs over files

        for f in files:
            df = pd.read_csv(f)
            df["Deletion Ratios"] = df["Deletion Ratios"].astype(str).str.strip()

            rd = df[df["Deletion Ratios"] == "Random"].iloc[0]
            cl = df[df["Deletion Ratios"] == "Score"].iloc[0]

            # init ratio_diffs once, from columns (skip the "Deletion Ratios" column)
            if ratio_diffs is None:
                ratio_cols = [c for c in df.columns if c != "Deletion Ratios"]
                ratio_diffs = {col: [] for col in ratio_cols}

            for col in ratio_diffs.keys():
                diff = float(cl[col] - rd[col])
                ratio_diffs[col].append(diff)

        print(f"model: {model}")
        for col, diffs_list in ratio_diffs.items():
            diffs = np.array(diffs_list, dtype=float)
            mean_diff = diffs.mean()

            if len(diffs) > 1:
                t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
                print(
                    f"  ratio {col}: mean_diff={mean_diff:.6f}, "
                    f"t_stat={t_stat:.3f}, p_val={p_val:.4f}, n={len(diffs)}"
                )
            else:
                # only one run → no t-test possible
                print(
                    f"  ratio {col}: mean_diff={mean_diff:.6f}, "
                    f"t_stat=NA, p_val=NA, n={len(diffs)}"
                )



for DB in DBS:
    for model in models:
        files = sorted(glob.glob(f"./final_results/without_recipriocal/{DB}/{model}/delete/*.csv"))

        # ratio -> list of values across runs
        random_vals = None   # dict: ratio_col -> [values of Random over files]
        score_vals  = None   # dict: ratio_col -> [values of Score  over files]

        for f in files:
            df = pd.read_csv(f)
            df["Deletion Ratios"] = df["Deletion Ratios"].astype(str).str.strip()

            rd = df[df["Deletion Ratios"] == "Random"].iloc[0]
            cl = df[df["Deletion Ratios"] == "Score"].iloc[0]

            # initialize dicts on first file
            if random_vals is None:
                ratio_cols = [c for c in df.columns if c != "Deletion Ratios"]
                random_vals = {col: [] for col in ratio_cols}
                score_vals  = {col: [] for col in ratio_cols}

            for col in random_vals.keys():
                random_vals[col].append(float(rd[col]))
                score_vals[col].append(float(cl[col]))

            print(f"\nmodel: {model}")
            for col in random_vals.keys():
                r = np.array(random_vals[col], dtype=float)
                s = np.array(score_vals[col], dtype=float)
                diffs = s - r

                r_mean, r_std = r.mean(), r.std(ddof=1)
                s_mean, s_std = s.mean(), s.std(ddof=1)
                mean_diff = diffs.mean()
                t_stat, p_val = stats.ttest_1samp(diffs, 0.0)

                print(
                    f"  ratio {col}: "
                    f"Random mean={r_mean:.4f} std={r_std:.4f} | "
                    f"Score mean={s_mean:.4f} std={s_std:.4f} | "
                    f"diff={mean_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
                )
