import pandas as pd
import glob, os
from pathlib import Path

def collect_stats(base_dir, edit_type):
    results = []
    for db in ["UMLS", "KINSHIP", "NELL-995-h100"]:
        for model in ["DistMult", "ComplEx", "Pykeen_TransE", "Pykeen_TransH",
                      "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL"]:
            folder = Path(base_dir) / db / model / edit_type
            csv_files = glob.glob(os.path.join(folder, "*.csv"))
            if not csv_files:
                continue

            dfs = [pd.read_csv(f, header=None) for f in csv_files]
            ratios = dfs[0].iloc[0, 1:].astype(float).values
            methods = dfs[0].iloc[1:, 0].values

            # Compute median performance across seeds for each method × ratio
            method_medians = {}
            for i, method in enumerate(methods, start=1):
                values = []
                for df in dfs:
                    vals = df.iloc[i, 1:].astype(float).values
                    values.append(vals)
                values = pd.DataFrame(values, columns=ratios)
                method_medians[method] = values.median()

            # Now compute drop relative to Random baseline at the same ratio
            random_medians = method_medians.get("Random")
            if random_medians is None:
                continue

            for method, medians in method_medians.items():
                for r in ratios:
                    baseline = random_medians[r]
                    drop = (baseline - medians[r]) / baseline * 100 if baseline != 0 else 0
                    results.append({
                        "Dataset": db,
                        "Model": model,
                        "Method": method,
                        "Ratio": r,
                        "Drop%": drop
                    })
    return pd.DataFrame(results)

base_dir = "../../../all_ratios/" #"../VM/sofar/wo/"
df_add = collect_stats(base_dir, "add")
df_del = collect_stats(base_dir, "del")

df_add.to_pickle("df_add.pkl")
df_del.to_pickle("df_del.pkl")


print(df_add)
print(df_del)

# Example summary: average drop per model across datasets
summary_add = df_add.groupby("Model")["Drop%"].mean().sort_values()
summary_del = df_del.groupby("Model")["Drop%"].mean().sort_values()

print("Addition (avg drop %):")
print(summary_add)
print("\nDeletion (avg drop %):")
print(summary_del)


summary_add = df_add.groupby("Method")["Drop%"].mean().sort_values()
summary_del = df_del.groupby("Method")["Drop%"].mean().sort_values()
print("Addition (avg drop %):")
print(summary_add)
print("\nDeletion (avg drop %):")
print(summary_del)

print("##################ADD###################")

summary_add = (
    df_add.groupby(["Dataset", "Method"])["Drop%"]
    .mean()
    .reset_index()
    .sort_values(["Dataset", "Drop%"])
)
print(summary_add)

print("#################DEL####################")

summary_del = (
    df_del.groupby(["Dataset", "Method"])["Drop%"]
    .mean()
    .reset_index()
    .sort_values(["Dataset", "Drop%"])
)
print(summary_del)

print("##################ADD###################")

summary_add = (
    df_add.groupby(["Dataset", "Model", "Method"])["Drop%"]
    .mean()
    .reset_index()
    .sort_values(["Dataset", "Drop%"])
)
print(summary_add)

print("#################DEL####################")

summary_del = (
    df_del.groupby(["Dataset", "Model", "Method"])["Drop%"]
    .mean()
    .reset_index()
    .sort_values(["Dataset", "Drop%"])
)
print(summary_del)