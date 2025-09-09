
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

def plot_boxplots_from_folder(folder_path, db, model, edit_type):
    figsize = (10, 8)

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(csv_files)
    seeds = []
    for f in csv_files:
        seeds.append(f.split("-")[-1].replace(".csv", ""))

    dfs = [pd.read_csv(f, header=None) for f in csv_files]

    ratios = dfs[0].iloc[0, 1:].astype(float).values

    methods = dfs[0].iloc[1:, 0].values

    data = []
    for df_idx, df in enumerate(dfs):
        for i, method in enumerate(methods, start=1):
            values = df.iloc[i, 1:].astype(float).values
            for r, v in zip(ratios, values):
                data.append({
                    "Ratio": r,
                    "Method": method,
                    "Value": v,
                    "Run": df_idx
                })

    long_df = pd.DataFrame(data)

    plt.figure(figsize=figsize)

    palette = sns.color_palette("pastel")



    ax = sns.boxplot(
        data=long_df,
        x="Ratio", y="Value", hue="Method", showfliers=False, width=0.6, palette=palette
    )

    xticks = ax.get_xticks()
    for pos in xticks[:-1]:
        ax.axvline(pos + 0.5, color="gray", linestyle="--", alpha=0.2)


    sns.set_theme(style="whitegrid", context="talk")


    plt.suptitle(f"Seeds: {', '.join(seeds)}", fontsize=10, y=0.02)

    plt.title(f"{db}-{model}-{edit_type}", fontsize=16)
    plt.ylabel("MRR", fontsize=14)
    plt.xlabel("Triple Injection Ratio", fontsize=14)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.5)
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)

    #plt.show()
    plt.savefig(f"../vis_new/{db}@{model}@{edit_type}", dpi=300, bbox_inches="tight")



DBS = ["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR"]
MODELS = [ "DistMult", "ComplEx", 'Pykeen_TransE', 'Pykeen_TransH', "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL" ]


#base_dir = Path("../final_results/wo")
base_dir = Path("../../../from_vm/wo")

for db in base_dir.iterdir():
    if db.is_dir():
        for model in db.iterdir():
            if model.is_dir():
                for type_dir in model.iterdir():
                    if type_dir.is_dir():
                        plot_boxplots_from_folder(type_dir, db.name, model.name, type_dir.name)








