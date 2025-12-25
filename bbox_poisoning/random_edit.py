import random
from typing import List, Tuple

Triple = Tuple[str, str, str]

def read_triples(path: str) -> List[Triple]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # fallback if the file is whitespace-separated
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Bad line (expected 3 columns): {line}")
            h, r, t = (p.strip() for p in parts)
            triples.append((h, r, t))
    return triples

def write_triples(path: str, triples: List[Triple]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

def corrupt_heads_or_tails(
    in_path: str,
    out_path: str,
    corruption_rate: float = 1.0,   # 1.0 = corrupt every triple; 0.2 = corrupt 20%
    p_corrupt_head: float = 0.5,    # probability corrupt head vs tail (when chosen to corrupt)
    seed: int = 42,
    ensure_change: bool = True,     # avoid replacing with the same value
) -> None:
    rng = random.Random(seed)

    triples = read_triples(in_path)
    if not triples:
        raise ValueError("No triples found.")

    # Candidate pools from existing values
    heads = [h for (h, _, _) in triples]
    tails = [t for (_, _, t) in triples]

    corrupted: List[Triple] = []
    for (h, r, t) in triples:
        if rng.random() > corruption_rate:
            corrupted.append((h, r, t))
            continue

        if rng.random() < p_corrupt_head:
            # corrupt head
            new_h = rng.choice(heads)
            if ensure_change and len(set(heads)) > 1:
                while new_h == h:
                    new_h = rng.choice(heads)
            corrupted.append((new_h, r, t))
        else:
            # corrupt tail
            new_t = rng.choice(tails)
            if ensure_change and len(set(tails)) > 1:
                while new_t == t:
                    new_t = rng.choice(tails)
            corrupted.append((h, r, new_t))

    write_triples(out_path, corrupted)

if __name__ == "__main__":
    corrupt_heads_or_tails(
        in_path="./splits/removed.txt",
        out_path="./splits/removed_corrupted.txt",
        corruption_rate=1.0,    # corrupt every triple
        p_corrupt_head=0.5,     # head/tail split
        seed=123,
        ensure_change=True,
    )
    print("Wrote corrupted triples")
