from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REQUIRED_COLS = ["DB", "RunModel", "DataModel", "Ratio", "Seed", "MRR"]


def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    for c in ["DB", "RunModel", "DataModel"]:
        df[c] = df[c].astype(str).str.strip()

    df["Ratio"] = pd.to_numeric(df["Ratio"], errors="coerce")
    df["Seed"]  = pd.to_numeric(df["Seed"], errors="coerce")
    df["MRR"]   = pd.to_numeric(df["MRR"], errors="coerce")

    df = df.dropna(subset=["DB", "RunModel", "DataModel", "Ratio", "Seed", "MRR"])
    df = df.drop_duplicates(subset=["DB", "RunModel", "DataModel", "Ratio", "Seed"])

    return df


def grouped_colored_boxplot_all_ratios(
    df: pd.DataFrame,
    db: str,
    runmodel: str,
    outpath: str,
) -> None:
    g = df[(df["DB"] == db) & (df["RunModel"] == runmodel)].copy()
    if g.empty:
        raise ValueError(f"No rows found for DB={db}, RunModel={runmodel}")

    ratios = sorted(g["Ratio"].unique())
    datamodels = sorted(g["DataModel"].unique())

    # Use matplotlib's default color cycle (not hard-coded colors)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    dm2color = {dm: palette[i % len(palette)] for i, dm in enumerate(datamodels)}

    # Layout tuning
    group_gap = 1.6
    box_width = 0.65

    data = []
    positions = []
    box_dms = []
    ratio_centers = []

    pos = 1.0
    for r in ratios:
        start = pos
        for dm in datamodels:
            vals = g[(g["Ratio"] == r) & (g["DataModel"] == dm)]["MRR"].astype(float).values
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(pos)
            box_dms.append(dm)
            pos += 1.0
        end = pos - 1.0
        ratio_centers.append((start + end) / 2.0 if end >= start else start)
        pos += group_gap

    fig_w = max(12, 1.2 * len(ratios))
    fig_h = 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,   # <-- enables filled boxes
        showfliers=True,
        whis=1.5,
        manage_ticks=False,
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1.3),
        capprops=dict(linewidth=1.3),
        boxprops=dict(linewidth=1.3),
    )

    # Color each box by its DataModel
    for box, dm in zip(bp["boxes"], box_dms):
        box.set_facecolor(dm2color[dm])
        box.set_alpha(0.45)

    # Keep fliers readable
    for flier in bp.get("fliers", []):
        flier.set_markersize(3)
        flier.set_alpha(0.6)

    ax.set_title(f"MRR over seeds (colored by DataModel)\nDB={db} | RunModel={runmodel}")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("MRR")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Ratio ticks at group centers
    ax.set_xticks(ratio_centers)
    ax.set_xticklabels([str(int(r)) if float(r).is_integer() else str(r) for r in ratios])

    # Vertical separators between ratio groups
    for i in range(len(ratio_centers) - 1):
        mid = (ratio_centers[i] + ratio_centers[i + 1]) / 2.0
        ax.axvline(mid, linewidth=1, alpha=0.15)

    # Legend mapping DataModel -> color
    handles = [Patch(facecolor=dm2color[dm], edgecolor="black", alpha=0.45, label=dm) for dm in datamodels]
    ax.legend(handles=handles, title="DataModel", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():

    df = load_results_csv("./score/UMLS/performance.csv")  # <-- change path

    outdir = "boxplots_all_ratios_colored"
    for (db, runmodel), _ in df.groupby(["DB", "RunModel"], sort=True):
        outpath = os.path.join(outdir, f"box__DB={db}__RunModel={runmodel}.png".replace("/", "_"))
        grouped_colored_boxplot_all_ratios(df, db=db, runmodel=runmodel, outpath=outpath)



if __name__ == "__main__":
    main()
