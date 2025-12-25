import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv("triples_with_scores_and_centrality.csv")
df = df.dropna(subset=["r", "score"])

min_n = 10
rel_counts = df["r"].value_counts()
order = rel_counts[rel_counts >= min_n].index.tolist()

scores_all = df.loc[df["r"].isin(order), "score"].to_numpy()
bins = np.linspace(scores_all.min(), scores_all.max(), 30)

n = len(order)
ncols = 6
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows), sharex=True, sharey=True)
axes = np.array(axes).reshape(-1)

for ax, rel in zip(axes, order):
    s = df.loc[df["r"] == rel, "score"].to_numpy()
    ax.hist(s, bins=bins, alpha=0.85)
    ax.set_title(f"{rel} (n={len(s)})", fontsize=9)
    ax.grid(alpha=0.15)

# Turn off unused axes
for ax in axes[len(order):]:
    ax.axis("off")

fig.suptitle("Per-relation score histograms", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
