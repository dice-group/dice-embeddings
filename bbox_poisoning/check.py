from typing import List, Set, Tuple

Triple = Tuple[str, str, str]

def read_triples_file(path: str) -> Set[Triple]:
    triples: Set[Triple] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t") if "\t" in s else s.split()
            if len(parts) < 3:
                continue
            triples.add((parts[0], parts[1], parts[2]))
    return triples

def common_triples_across_files(file_paths: List[str]) -> Set[Triple]:
    if not file_paths:
        return set()
    first = True
    commons: Set[Triple] = set()
    for p in file_paths:
        s = read_triples_file(p)
        if first:
            commons = set(s)
            first = False
        else:
            commons &= s
        if not commons:  # early exit if empty
            break
    return commons

def save_triples(triples: Set[Triple], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


files = [
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/0/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/1/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/2/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/3/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/4/train.txt",
            ]
commons = common_triples_across_files(files)
print(f"Common triples: {len(commons)}")

