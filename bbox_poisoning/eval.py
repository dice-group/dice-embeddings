import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

def _parse_run_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, header=None)
    methods = df.iloc[:, 0].astype(str)
    vals = df.iloc[:, 1:]

    mask = methods.str.strip().str.lower() != "triple injection ratios"
    methods = methods[mask]
    vals = vals[mask].apply(pd.to_numeric, errors="coerce")

    vals.columns = [f"R{i+1}" for i in range(vals.shape[1])]
    out = vals.copy()
    out.index = methods
    return out

def load_experiment_folder(folder_path: str, target: str = "random"):

    run_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    if not run_files:
        raise ValueError("No .csv files found in the folder.")

    runs = [_parse_run_csv(os.path.join(folder_path, f)) for f in run_files]
    all_methods = sorted(set().union(*[set(df.index) for df in runs]))

    per_run_means = {}
    for i, df in enumerate(runs, start=1):
        aligned = df.reindex(index=all_methods)
        per_run_means[f"run{i}"] = aligned.mean(axis=1, skipna=True)

    data_df = pd.DataFrame(per_run_means)  # rows=methods, cols=runs

    if target not in data_df.index:
        raise ValueError(f"Target method '{target}' not found in the data.")
    results = []
    for method in data_df.index:
        if method == target:
            continue
        x = data_df.loc[method].values
        y = data_df.loc[target].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            t_stat, p_val, mean_diff = np.nan, np.nan, np.nan
        else:
            t_stat, p_val = ttest_rel(x[mask], y[mask])
            mean_diff = float(np.nanmean(x[mask] - y[mask]))
        results.append((method, t_stat, p_val, mean_diff))

    t_test_df = pd.DataFrame(results, columns=["Method", "T-statistic", "p-value", "Mean diff"])\
                   .sort_values("p-value", na_position="last")

    return data_df, t_test_df

def plot_mean_std_from_df(data_df: pd.DataFrame, title="Mean ± Std across runs"):
    means = data_df.mean(axis=1, skipna=True).sort_values(ascending=False)
    stds  = data_df.std(axis=1, ddof=1, skipna=True).reindex(means.index)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(means))
    plt.errorbar(x, means.values, yerr=stds.values, fmt='o-', capsize=4)
    plt.xticks(x, means.index, rotation=20, ha='right')
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


folder_path = "../junks/older_one/Keci/rel/"
data_df, t_test_df = load_experiment_folder(folder_path, target="random")
print(t_test_df)
plot_mean_std_from_df(data_df, title="Experiment Results")
