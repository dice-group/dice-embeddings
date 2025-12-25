import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load
df = pd.read_csv("triples_with_scores_and_centrality.csv")
df = df.dropna(subset=["r", "score"])

# Optional: drop relations with too few samples (otherwise you get silly “distributions”)
min_n = 10
rel_counts = df["r"].value_counts()
order = rel_counts[rel_counts >= min_n].index.tolist()

# Common bins for comparability
scores_all = df.loc[df["r"].isin(order), "score"].to_numpy()
bins = np.linspace(scores_all.min(), scores_all.max(), 40)
centers = 0.5 * (bins[:-1] + bins[1:])

# Plot setup
plt.figure(figsize=(14, max(6, 0.28 * len(order))))
ax = plt.gca()

# Vertical spacing between ridges
gap = 1.0

for i, rel in enumerate(order):
    s = df.loc[df["r"] == rel, "score"].to_numpy()
    hist, _ = np.histogram(s, bins=bins, density=True)

    # Scale each relation to comparable height (shape comparison)
    if hist.max() > 0:
        hist = hist / hist.max() * 0.9

    y0 = i * gap
    ax.fill_between(centers, y0, y0 + hist, alpha=0.8, linewidth=0.8)
    ax.plot(centers, y0 + hist, linewidth=0.8)

# Labels
ax.set_yticks([i * gap for i in range(len(order))])
ax.set_yticklabels(order, fontsize=8)
ax.invert_yaxis()  # most frequent at top if order is frequency-sorted
ax.set_xlabel("Score")
ax.set_title("Per-relation score distributions (ridgeline histograms; normalized per relation)")
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.show()
