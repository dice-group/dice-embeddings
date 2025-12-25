import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your file
df = pd.read_csv("triples_with_scores_and_centrality.csv")

# Sanity check columns
assert "r" in df.columns and "score" in df.columns, f"Need columns r and score, got: {df.columns.tolist()}"

# Order relations by frequency (optional but makes the plot nicer)
rel_counts = df["r"].value_counts()
order = rel_counts.index.tolist()

# Gather scores per relation
data = [df.loc[df["r"] == rel, "score"].dropna().to_numpy() for rel in order]

# Plot: one figure, one big boxplot
plt.figure(figsize=(max(12, 0.35 * len(order)), 6))
plt.boxplot(data, vert=True, showfliers=False)  # showfliers=True if you want outliers

plt.xticks(
    ticks=np.arange(1, len(order) + 1),
    labels=order,
    rotation=90,
    ha="center",
    fontsize=8,
)
plt.ylabel("Score")
plt.title("Per-relation score distribution (boxplot; outliers hidden)")
plt.tight_layout()
plt.show()
