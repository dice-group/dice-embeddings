import pandas as pd
import matplotlib.pyplot as plt

def plot_score_correlations_with_selected_labels(
    csv_path,
    highlight_triples,   # list of (h, r, t) using your CSV values (strings or ints)
    out_dir=None,
    alpha=0.15,
    point_size=6,
):
    df = pd.read_csv(csv_path)

    required = ["h", "r", "t", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain columns {required}. Missing: {missing}")

    def norm(x):
        return str(x).strip()

    # Normalize h/r/t columns to strings for matching + labeling
    df["h"] = df["h"].map(norm)
    df["r"] = df["r"].map(norm)
    df["t"] = df["t"].map(norm)

    exclude = {"h", "r", "t"}
    metric_cols = [
        c for c in df.columns
        if c not in exclude and c != "score" and pd.api.types.is_numeric_dtype(df[c])
    ]

    mean_score = df["score"].mean()

    # Build a set of normalized highlight triples
    highlight_set = set((norm(h), norm(r), norm(t)) for (h, r, t) in highlight_triples)

    # Find rows that match highlights (no int casting)
    annotate_mask = df.apply(lambda row: (row["h"], row["r"], row["t"]) in highlight_set, axis=1)
    annotate_rows = df.index[annotate_mask].tolist()

    for col in metric_cols:
        tmp = df[["score", col]].dropna()
        pearson = tmp.corr(method="pearson").iloc[0, 1]
        spearman = tmp.corr(method="spearman").iloc[0, 1]
        mean_x = tmp[col].mean()

        plt.figure(figsize=(7, 5))
        plt.scatter(df[col], df["score"], alpha=alpha, s=point_size)

        # Mean lines
        plt.axhline(mean_score, linestyle="--", linewidth=1.5,
                    label=f"mean(score) = {mean_score:.4g}")
        plt.axvline(mean_x, linestyle="--", linewidth=1.5,
                    label=f"mean({col}) = {mean_x:.4g}")

        # Annotate only highlighted triples
        offsets = [(8, 8), (8, -12), (-40, 8), (-40, -12)]
        for i, idx in enumerate(annotate_rows):
            if pd.isna(df.at[idx, col]) or pd.isna(df.at[idx, "score"]):
                continue
            x = df.at[idx, col]
            y = df.at[idx, "score"]
            label = f"({df.at[idx,'h']},{df.at[idx,'r']},{df.at[idx,'t']})"
            dx, dy = offsets[i % len(offsets)]
            plt.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.6),
            )

        plt.xlabel(col)
        plt.ylabel("score")
        plt.title(f"score vs {col} | Pearson r={pearson:.3f}, Spearman ρ={spearman:.3f}")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.xscale("log")

        if out_dir:
            safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in col)
            plt.savefig(f"{out_dir.rstrip('/')}/score_vs_{safe}.png", dpi=200)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Use your actual string triples (matching the CSV h/r/t values)
    highlight = [
        ("Biomedical_occupation_or_discipline", "isa", "occupation_or_discipline"),
        ("occupation_or_discipline", "issue_in", "biomedical_occupation_or_discipline"),
        ("biomedical_occupation_or_discipline", "issue_in", "occupation_or_discipline"),
    ]

    plot_score_correlations_with_selected_labels(
        "triples_with_scores_and_centrality.csv",
        highlight
    )
