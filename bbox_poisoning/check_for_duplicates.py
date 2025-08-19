from collections import Counter

def find_duplicate_triples(path, sep="\t"):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(sep) if sep is not None else s.split()
            if len(parts) != 3:
                # skip malformed lines; change to 'raise' if you prefer
                continue
            triples.append(tuple(parts))

    counts = Counter(triples)
    return [(t, c) for t, c in counts.items() if c > 1]


dups = find_duplicate_triples("./UMLS/active_poisoning_whitebox/Pykeen_MuRE/adverserial_fgsm_triples/521/random-one/0/train.txt")
if dups:
    for t, c in dups:
        print(f"Duplicate {c}×: {t}")
else:
    print("No duplicates.")
