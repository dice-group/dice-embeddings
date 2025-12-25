import pandas as pd

per_file = pd.read_csv("frac_compare.metrics_per_file.tsv", sep="\t")
diff_vs_base = pd.read_csv("frac_compare.metrics_diff_vs_baseline.tsv", sep="\t")

# 1. Inspect what the "baseline" row actually is
print("=== metrics_per_file: unique values in identifying columns ===")
for col in ["file", "setting", "tag", "name"]:
    if col in per_file.columns:
        print(f"\nColumn: {col}")
        print(per_file[col].unique())

# try to find baseline row(s)
baseline_rows = per_file[
    per_file.apply(
        lambda row: any(
            isinstance(v, str) and "baseline" in v.lower()
            for v in row.astype(str)
        ),
        axis=1
    )
]
print("\n=== Baseline rows in metrics_per_file.tsv ===")
print(baseline_rows)

# 2. Show the metric differences for each variant vs baseline
print("\n=== diff_vs_baseline head ===")
print(diff_vs_base.head())

# 3. If there is a 'file' or 'frac' column, summarize how each variant differs
group_cols = [c for c in ["file", "frac", "setting"] if c in diff_vs_base.columns]
metric_cols = [c for c in diff_vs_base.columns if c not in group_cols]

print("\n=== Mean diff per file/frac vs baseline ===")
print(diff_vs_base.groupby(group_cols)[metric_cols].mean())
