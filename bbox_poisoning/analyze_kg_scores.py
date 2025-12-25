import os
import pandas as pd
import matplotlib.pyplot as plt


def analyze_kge_scores(csv_path: str, out_dir: str = "analysis_out", top_n: int = 50, bins: int = 50):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"h", "r", "t", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Basic cleanup
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["h"] = df["h"].astype(str)
    df["r"] = df["r"].astype(str)
    df["t"] = df["t"].astype(str)

    # 1) Frequency distributions
    freq_h = df["h"].value_counts().rename_axis("h").reset_index(name="count")
    freq_r = df["r"].value_counts().rename_axis("r").reset_index(name="count")
    freq_t = df["t"].value_counts().rename_axis("t").reset_index(name="count")

    freq_h.to_csv(os.path.join(out_dir, "freq_h.csv"), index=False)
    freq_r.to_csv(os.path.join(out_dir, "freq_r.csv"), index=False)
    freq_t.to_csv(os.path.join(out_dir, "freq_t.csv"), index=False)

    # 2) Score distribution summaries per h, r, t (tables)
    def group_stats(key):
        g = df.groupby(key)["score"]
        stats = g.agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
        q = g.quantile([0.05, 0.25, 0.75, 0.95]).unstack().reset_index()
        q.columns = [key, "q05", "q25", "q75", "q95"]
        out = stats.merge(q, on=key, how="left")
        return out.sort_values("count", ascending=False)

    stats_h = group_stats("h")
    stats_r = group_stats("r")   # <-- most important
    stats_t = group_stats("t")

    stats_h.to_csv(os.path.join(out_dir, "score_stats_by_h.csv"), index=False)
    stats_r.to_csv(os.path.join(out_dir, "score_stats_by_r.csv"), index=False)
    stats_t.to_csv(os.path.join(out_dir, "score_stats_by_t.csv"), index=False)

    # 3) Plots (focused on r)

    # Overall score histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df["score"], bins=bins)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title("Overall score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scores_overall_hist.png"), dpi=200)
    plt.close()

    # Top-N relations by frequency (bar)
    top_r = freq_r.head(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top_r)), top_r["count"])
    plt.xticks(range(len(top_r)), top_r["r"], rotation=75, ha="right")
    plt.ylabel("count")
    plt.title(f"Top {top_n} relations by frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"r_frequency_top{top_n}.png"), dpi=200)
    plt.close()

    # Score distribution per relation: boxplot for top-N relations
    top_r_names = top_r["r"].tolist()
    sub = df[df["r"].isin(top_r_names)].copy()
    sub["r"] = pd.Categorical(sub["r"], categories=top_r_names, ordered=True)

    data = [sub.loc[sub["r"] == rel, "score"].values for rel in top_r_names]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=top_r_names, showfliers=False)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("score")
    plt.title(f"Score distribution by relation (boxplot, top {top_n} relations)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"score_by_r_boxplot_top{top_n}.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(stats_r["count"], stats_r["mean"])
    plt.xscale("log")  # relations are usually long-tailed
    plt.xlabel("relation frequency (log scale)")
    plt.ylabel("mean(score)")
    plt.title("Mean score vs relation frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "r_mean_score_vs_count.png"), dpi=200)
    plt.close()

    return {
        "rows_used": len(df),
        "unique_h": df["h"].nunique(),
        "unique_r": df["r"].nunique(),
        "unique_t": df["t"].nunique(),
        "out_dir": out_dir,
    }


if __name__ == "__main__":
    info = analyze_kge_scores(
        csv_path="triples_with_scores_and_centrality.csv",
        out_dir="analysis_out",
        top_n=50,   # change if you want
        bins=50
    )
    print(info)
