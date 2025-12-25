import os
import random

def random_remove_triples(
    train_path: str,
    k: int,
    out_dir: str = "random_split",
    seed: int = 42,
    removed_name: str = "removed.txt",
    train_name: str = "train.txt",
):
    # Read lines (keep original formatting, but strip trailing newlines)
    with open(train_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    n = len(lines)
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, {n}], got {k}")

    rng = random.Random(seed)
    remove_idx = set(rng.sample(range(n), k))  # unique indices

    removed = [lines[i] for i in remove_idx]
    kept = [lines[i] for i in range(n) if i not in remove_idx]

    os.makedirs(out_dir, exist_ok=True)
    removed_path = os.path.join(out_dir, removed_name)
    new_train_path = os.path.join(out_dir, train_name)

    with open(removed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(removed) + ("\n" if removed else ""))

    with open(new_train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))

    return removed_path, new_train_path, n, len(removed), len(kept)


if __name__ == "__main__":
    removed_path, new_train_path, n, n_removed, n_kept = random_remove_triples(
        train_path="./KGs/UMLS/train.txt",
        k=10,              # how many triples to remove
        out_dir="rand_split",
        seed=123,               # change seed for a different random split
    )

    print(f"Original triples: {n}")
    print(f"Removed: {n_removed} -> {removed_path}")
    print(f"Remaining: {n_kept} -> {new_train_path}")
