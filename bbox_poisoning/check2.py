import random
from typing import List, Tuple, Set, Dict
from collections import Counter

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

def write_triples_file(path: str, triples: List[Triple], shuffle: bool = True) -> None:
    out = list(triples)
    if shuffle:
        random.shuffle(out)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in out:
            f.write(f"{h}\t{r}\t{t}\n")

def load_runs(file_paths: List[str]) -> List[Set[Triple]]:
    runs: List[Set[Triple]] = []
    for p in file_paths:
        runs.append(read_triples_file(p))
    return runs

def build_component_counts(runs: List[Set[Triple]]) -> Dict[str, Counter]:
    head_c = Counter()
    rel_c  = Counter()
    tail_c = Counter()
    hr_c   = Counter()
    rt_c   = Counter()
    ht_c   = Counter()
    for run in runs:
        H = set([h for (h,_,_) in run])
        R = set([r for (_,r,_) in run])
        T = set([t for (_,_,t) in run])
        for x in H: head_c[x] += 1
        for x in R: rel_c[x]  += 1
        for x in T: tail_c[x] += 1
        for (h,r,t) in run:
            hr_c[(h,r)] += 1
            rt_c[(r,t)] += 1
            ht_c[(h,t)] += 1
    return {"H": head_c, "R": rel_c, "T": tail_c, "HR": hr_c, "RT": rt_c, "HT": ht_c}

def score_triple(t: Triple, counts: Dict[str, Counter], w=(1.0,1.0,1.0,1.0,1.0,0.5)) -> float:
    h, r, tail = t
    return (
        w[0]*counts["H"][h] +
        w[1]*counts["R"][r] +
        w[2]*counts["T"][tail] +
        w[3]*counts["HR"][(h,r)] +
        w[4]*counts["RT"][(r,tail)] +
        w[5]*counts["HT"][(h,tail)]
    )

def collect_pool(runs: List[Set[Triple]]) -> Set[Triple]:
    pool: Set[Triple] = set()
    for run in runs:
        pool |= run
    return pool

def score_all_triples(pool: Set[Triple], counts: Dict[str, Counter], w=(1,1,1,1,1,0.5)) -> List[Tuple[Triple,float]]:
    scored: List[Tuple[Triple,float]] = []
    for t in pool:
        s = score_triple(t, counts, w=w)
        scored.append((t, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def pick_top_k(scored: List[Tuple[Triple,float]], k: int, per_entity_cap: int = 0) -> List[Tuple[Triple,float]]:
    if k <= 0:
        return []
    if per_entity_cap <= 0:
        return scored[:k]
    chosen: List[Tuple[Triple,float]] = []
    used_h = Counter(); used_r = Counter(); used_t = Counter()
    for t, s in scored:
        h, r, tail = t
        if used_h[h] < per_entity_cap and used_r[r] < per_entity_cap and used_t[tail] < per_entity_cap:
            chosen.append((t, s))
            used_h[h] += 1; used_r[r] += 1; used_t[tail] += 1
            if len(chosen) >= k:
                break
    return chosen

def update_reference_with_topk(
    run_files: List[str],
    reference_path: str,
    output_reference_path: str,
    k: int,
    per_entity_cap: int = 0,
    weights=(1.0, 1.2, 1.0, 1.5, 1.5, 0.5),
    shuffle_output: bool = True
) -> Tuple[List[Tuple[Triple,float]], int]:

    runs = load_runs(run_files)
    counts = build_component_counts(runs)
    pool = collect_pool(runs)
    scored = score_all_triples(pool, counts, w=weights)
    topk = pick_top_k(scored, k=k, per_entity_cap=per_entity_cap)

    ref_set = read_triples_file(reference_path)
    before = len(ref_set)
    for t, _s in topk:
        if t not in ref_set:
            ref_set.add(t)
    total_written = len(ref_set)

    write_triples_file(output_reference_path, list(ref_set), shuffle=shuffle_output)
    # Optionally: print(f"Saved updated reference with {total_written} triples -> {output_reference_path}")
    return topk, total_written



run_files = [
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/0/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/1/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/2/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/3/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/4/train.txt",
            ]

reference = "./UMLS/clean/train.txt"

topk, new_size = update_reference_with_topk(
    run_files=run_files,
    reference_path=reference,
    output_reference_path="train.txt",
    k=2000,
    per_entity_cap=65,              # 0 to disable diversity cap
    weights=(1.0,1.2,1.0,1.5,1.5,0.5),  # tweak if you want
    shuffle_output=True
)

for (h,r,t), s in topk:
    print(h, r, t, f"{s:.2f}")

print(f"Updated reference size: {new_size}")
print(len(topk))
