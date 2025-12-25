import os
import pandas as pd

def split_removed_and_train_to_dir(
    csv_path: str,
    k: int,
    out_dir: str,
    betweenness_col: str = "edge_betweenness_dir",
    removed_filename: str = "removed.txt",
    train_filename: str = "train.txt",
):
    df = pd.read_csv(csv_path)

    needed = {"h", "r", "t", "score", betweenness_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    # Normalize h/r/t to strings
    for c in ["h", "r", "t"]:
        df[c] = df[c].astype(str).str.strip()

    mean_score = df["score"].mean() # DistMult, Complex
    #mean_score = df["score"].max() # QMult
    #mean_score = df["score"].min() # RotatE, MuRE

    tmp = df[["h", "r", "t", "score", betweenness_col]].dropna().copy()
    tmp["abs_score_diff"] = (tmp["score"] - mean_score).abs()

    # closest to mean score first, then highest betweenness
    tmp = tmp.sort_values(
        by=["abs_score_diff", betweenness_col],
        ascending=[True, False],
        kind="mergesort",
    )

    removed_df = tmp.head(k).copy()
    removed_keys = set(zip(removed_df["h"], removed_df["r"], removed_df["t"]))

    all_keys = list(zip(df["h"], df["r"], df["t"]))
    keep_mask = [key not in removed_keys for key in all_keys]
    train_df = df.loc[keep_mask, ["h", "r", "t"]].copy()

    # Make output dir + write files
    os.makedirs(out_dir, exist_ok=True)
    removed_path = os.path.join(out_dir, removed_filename)
    train_path = os.path.join(out_dir, train_filename)

    removed_df[["h", "r", "t"]].to_csv(removed_path, sep="\t", header=False, index=False)
    train_df.to_csv(train_path, sep="\t", header=False, index=False)

    return removed_path, train_path, mean_score, len(removed_df), len(train_df)


if __name__ == "__main__":
    out_dir = "splits"   # <-- choose your directory name/path
    k = 10           # <-- choose k

    removed_path, train_path, mean_score, n_removed, n_train = split_removed_and_train_to_dir(
        csv_path="triples_with_scores_and_centrality.csv",
        k=k,
        out_dir=out_dir,
        betweenness_col="edge_betweenness_dir",  # or "edge_betweenness_undir"
    )

    print(f"mean(score) = {mean_score}")
    print(f"Wrote {n_removed} triples to: {removed_path}")
    print(f"Wrote {n_train} triples to:   {train_path}")
