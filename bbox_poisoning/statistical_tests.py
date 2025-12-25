import pandas as pd
import glob, os
import numpy as np
from scipy import stats

recipriocal = "with_recipriocal"

DBS = [ "UMLS", "KINSHIP"  ] 
models = [ "DistMult", "ComplEx", "QMult", "Pykeen_MuRE", "Pykeen_RotatE", "Keci", "DeCaL", "Pykeen_BoxE" ]

for DB in DBS:
    for model in models:
        files = sorted(glob.glob(f"./final_results/{recipriocal}/{DB}/{model}/delete/*.csv"))
        diffs = []

        for f in files:
            df = pd.read_csv(f)
            df["Deletion Ratios"] = df["Deletion Ratios"].str.strip()
            rd = df[df["Deletion Ratios"] == "Random"].iloc[0]
            cl = df[df["Deletion Ratios"] == "Score"].iloc[0]
            for col in df.columns[1:]:
                diffs.append(float(cl[col] - rd[col]))

        diffs = np.array(diffs)
        mean_diff = diffs.mean()
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        print("DB:", DB, "model:", model, " mean_diff: ", mean_diff, "t_stat:", t_stat, "p_val:", p_val)


