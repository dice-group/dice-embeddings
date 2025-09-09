import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import re

def parse_ratio(val):
    try:
        val = str(val).strip().replace("%", "")
        return float(val) / 100 if float(val) > 1 else float(val)
    except:
        return None

def make_sheet_name(dataset, model, used):
    # Excel sheet name rules: max 31 chars, no : \ / ? * [ ]
    base = f"{dataset}-{model}"
    name = re.sub(r'[:\\/?*\[\]]', '-', base)[:31] or "Sheet"
    # ensure uniqueness
    if name not in used:
        used.add(name)
        return name
    i = 2
    while True:
        candidate = (name[:31 - len(f"_{i}")]) + f"_{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1

root = "../../from_vm/final_results"
pattern_name = "results-*.csv"

sheets = {}
used_names = set()

for dirpath, dirnames, filenames in os.walk(root):
    matches = sorted(glob.glob(os.path.join(dirpath, pattern_name)))
    if not matches:
        continue

    method  = os.path.basename(dirpath)
    model   = os.path.basename(os.path.dirname(dirpath))
    dataset = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))

    all_dfs = []
    for path in matches:
        df = pd.read_csv(path)
        method_col = df.columns[0]
        long_df = df.melt(id_vars=[method_col], var_name="Ratio", value_name="Score")
        long_df = long_df.rename(columns={method_col: "Method"})
        all_dfs.append(long_df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["Ratio"] = combined["Ratio"].map(parse_ratio)
    combined["Score"] = pd.to_numeric(combined["Score"], errors="coerce")
    combined = combined.dropna(subset=["Ratio", "Score"])

    stats = (
        combined.groupby(["Method", "Ratio"])["Score"]
        .agg(mean="mean", std=lambda s: s.std(ddof=1))
        .reset_index()
    )

    mean_wide = stats.pivot(index="Method", columns="Ratio", values="mean").sort_index(axis=1)
    std_wide  = stats.pivot(index="Method", columns="Ratio", values="std").reindex(columns=mean_wide.columns)

    formatted = pd.DataFrame(index=mean_wide.index)
    for c in mean_wide.columns:
        m = mean_wide[c].fillna(np.nan)
        s = std_wide[c].fillna(np.nan)
        formatted[c] = m.map(lambda x: f"{x:.3f}") + " ± " + s.map(lambda x: f"{x:.3f}")

    formatted.columns = [f"{c:g}" for c in formatted.columns]

    sheet_name = make_sheet_name(dataset, model, used_names)
    sheets[sheet_name] = formatted

out_path = Path("../statistics_results/statistics_results.xlsx")
out_path.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    for name, df in sheets.items():
        df.to_excel(writer, sheet_name=name)

print(f"wrote {out_path}")
