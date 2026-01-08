from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import DBS

EXPECTED = ["DB", "RunModel", "DataModel", "Ratio", "Seed", "MRR"]

def load_performance_csv(path: str) -> pd.DataFrame:
    path = Path(path)

    # Try normal header first
    df = pd.read_csv(path)
    if all(c in df.columns for c in EXPECTED):
        pass
    else:
        # Fallback: headerless file
        df = pd.read_csv(path, header=None, names=EXPECTED)

    # Clean types
    df["DB"] = df["DB"].astype(str)
    df["RunModel"] = df["RunModel"].astype(str)
    df["DataModel"] = df["DataModel"].astype(str)

    # Ratio/Seed might be strings; coerce safely
    df["Ratio"] = pd.to_numeric(df["Ratio"], errors="coerce")
    df["Seed"]  = pd.to_numeric(df["Seed"], errors="coerce")
    df["MRR"]   = pd.to_numeric(df["MRR"], errors="coerce")

    df = df.dropna(subset=["Ratio", "Seed", "MRR"])
    df["Ratio"] = df["Ratio"].astype(int)
    df["Seed"]  = df["Seed"].astype(int)

    return df


def heatmap(matrix: pd.DataFrame, title: str, save_path: str = None, fmt="{:.3f}"):
    # matrix: rows = RunModel, cols = DataModel
    mat = matrix.copy()
    mat = mat.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(1 + 0.8 * mat.shape[1], 1 + 0.6 * mat.shape[0]))
    im = ax.imshow(mat.values, aspect="auto")

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)

    # <-- add axis labels (the thing you asked for)
    ax.set_xlabel("DataModel (dataset source)")
    ax.set_ylabel("RunModel (model being trained/evaluated)")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=90)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()
    plt.close(fig)


def compute_delta_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    ΔMRR = MRR(run_model on data_model) - MRR(run_model on run_model)
    matched by (DB, RunModel, Ratio, Seed)
    """
    base = df[df["RunModel"] == df["DataModel"]].copy()
    base = base.rename(columns={"MRR": "MRR_base"})[["DB", "RunModel", "Ratio", "Seed", "MRR_base"]]

    merged = df.merge(base, on=["DB", "RunModel", "Ratio", "Seed"], how="left")
    merged["DeltaMRR"] = merged["MRR"] - merged["MRR_base"]
    return merged


def main(csv_path: str, out_dir: str = "analysis_plots", db=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_performance_csv(csv_path)

    # 1) Heatmap: mean MRR across all ratios+seeds
    mean_mrr = df.groupby(["RunModel", "DataModel"])["MRR"].mean().unstack()
    heatmap(mean_mrr, f"Mean MRR (all ratios, all seeds) db= {db}",
            save_path=str(out_dir / f"heatmap_mean_mrr_all_{db}.png"),
            fmt="{:.3f}")

    # 2) Heatmap: mean ΔMRR vs baseline (RunModel==DataModel)
    merged = compute_delta_vs_baseline(df)
    mean_delta = merged.groupby(["RunModel", "DataModel"])["DeltaMRR"].mean().unstack()
    heatmap(mean_delta, f"Mean ΔMRR vs baseline (RunModel==DataModel) db= {db}",
            save_path=str(out_dir / f"heatmap_mean_delta_all_{db}.png"),
            fmt="{:+.3f}")

    # 3) Per-ratio heatmaps
    for ratio in sorted(df["Ratio"].unique()):
        df_r = df[df["Ratio"] == ratio]
        mean_mrr_r = df_r.groupby(["RunModel", "DataModel"])["MRR"].mean().unstack()
        heatmap(mean_mrr_r, f"Mean MRR (ratio={ratio}) db= {db}",
                save_path=str(out_dir / f"heatmap_mean_mrr_ratio_{ratio}_{db}.png"),
                fmt="{:.3f}")

        merged_r = compute_delta_vs_baseline(df_r)
        mean_delta_r = merged_r.groupby(["RunModel", "DataModel"])["DeltaMRR"].mean().unstack()
        heatmap(mean_delta_r, f"Mean ΔMRR vs baseline (ratio={ratio} db= {db})",
                save_path=str(out_dir / f"heatmap_mean_delta_ratio_{ratio}_{db}.png"),
                fmt="{:+.3f}")

    # 4) Print “which DataModel is worse than baseline” with effect size (pp)
    #    (aggregated over ratio+seed)
    mean_delta_pp = (mean_delta * 100.0)  # percentage points
    print("\nMean ΔMRR vs baseline (percentage points):")
    print(mean_delta_pp.round(3).fillna(np.nan))

    print(f"\nSaved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Change this to your actual CSV path
    for db in DBS:
        main(f"./combinations/{db}/performance.csv", out_dir="analysis_plots", db=db)
