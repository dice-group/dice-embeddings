
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

def plot_boxplots_from_folder(folder_path, db, model, edit_type):

    figsize = (8, 8)

    if "Pykeen_" in model:
        model = model.replace("Pykeen_", "", 1)

    print("*******************************************************")
    print("folder_path:", folder_path)
    print("edit_type:", edit_type)
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
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
    #palette = sns.color_palette("deep")
    palette = sns.color_palette(["#4C72B0", "#7B4C9A", "#55A868"])

    ax = sns.boxplot(
        data=long_df,
        x="Ratio", 
        y="Value", 
        hue="Method",
        showfliers=False, 
        width=0.55, 
        palette=palette,
        #linewidth=1.5, 
        medianprops={"color": "black", "linewidth": 2}
    )

    #xticks = ax.get_xticks()
    #for pos in xticks[:-1]:
    #    ax.axvline(pos + 0.5, color="gray", linestyle="--", alpha=0.2)

    #sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    sns.set_theme(style="white", context="talk", font_scale=1.2)

    #plt.suptitle(f"Seeds: {', '.join(seeds)}", fontsize=10, y=0.02)
    #plt.ylabel("MRR", fontsize=24)
    #plt.title(f"Dataset:{db}, Model:{model}", fontsize=22)

    #if edit_type == "add":
    #    plt.xlabel("Addition Ratio", fontsize=24)

    #elif edit_type == "del":
    #    plt.xlabel("Deletion Ratio", fontsize=24)


    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ["Closeness" if lab == "Harmonic closeness" else lab for lab in labels]
    plt.legend(handles, labels, loc='best', fontsize=18, framealpha=0.4)


    #plt.legend([], [], frameon=False)
    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)

    #sns.despine(offset=10, trim=True)  
    #ax.grid(axis="y", linestyle="--", alpha=0.6)  
    #ax.grid(axis="x", visible=False)             

    #plt.show()
    report_path = Path(f"../vis_sep16/{edit_type}")
    report_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(report_path / f"{db}_{model}_{edit_type}.png", dpi=400, bbox_inches="tight")


DBS = ["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR"]
MODELS = [ "DistMult", "ComplEx", 'Pykeen_TransE', 'Pykeen_TransH', "Pykeen_MuRE", "Pykeen_RotatE", "Keci", "DeCaL" ]


base_dir = Path("../../../all_ratios/")

for db in base_dir.iterdir():
    if db.is_dir():
        for model in db.iterdir():
            if model.is_dir():
                for type_dir in model.iterdir():
                    if type_dir.is_dir():
                        plot_boxplots_from_folder(type_dir, db.name, model.name, type_dir.name)








