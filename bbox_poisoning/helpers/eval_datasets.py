from typing import Set, Tuple

def _read_triples(path: str):
    triples: Set[Tuple[str, str, str]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t") if "\t" in s else s.split()
            if len(parts) < 3:
                # skip malformed lines
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.add((h, r, t))
    return triples

def count_triples_in_one_but_not_the_other(file_a: str, file_b: str) -> int:
    a = _read_triples(file_a)
    b = _read_triples(file_b)
    return len(a.symmetric_difference(b))

files_A = "../saved_datasets/wo/UMLS/active_poisoning_whitebox/ComplEx/add_hc/52/random-one/0/train.txt"
file_B = "../saved_datasets/wo/UMLS/active_poisoning_whitebox/DistMult/add_hc/52/random-one/0/train.txt"

#files_A = "./saved_datasets/wo/UMLS/random/ComplEx/2/random-one/0/train.txt"
#file_B = "./saved_datasets/wo/UMLS/random/ComplEx/2/random-one/1/train.txt"

n = count_triples_in_one_but_not_the_other(files_A, file_B)
print(n)

n = count_triples_in_one_but_not_the_other(file_B, files_A)
print(n)

only_in_a = _read_triples(files_A) - _read_triples(file_B)
only_in_b = _read_triples(files_A) - _read_triples(file_B)

print(len(only_in_a), len(only_in_b), len(only_in_a) + len(only_in_b))

