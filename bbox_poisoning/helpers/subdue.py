import random

def read_triples_file(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t") if "\t" in s else s.split()
            if len(parts) < 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples

def triples_to_set(triples):
    return set(triples)

def build_reference_set(reference_path):
    return triples_to_set(read_triples_file(reference_path))

def file_minus_reference(file_path, reference_set):
    file_set = triples_to_set(read_triples_file(file_path))
    return file_set - reference_set

def all_files_minus_reference(file_paths, reference_path):
    ref_set = build_reference_set(reference_path)
    result = {}
    for p in file_paths:
        result[p] = file_minus_reference(p, ref_set)
    return result

def common_triples_in_all_diffs(file_paths, reference_path):
    diffs = all_files_minus_reference(file_paths, reference_path)
    if not diffs:
        return set()
    print("--------", set.intersection(*diffs.values()))
    return set.intersection(*diffs.values())




if __name__ == "__main__":
    files = [
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/0/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/1/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/2/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/3/train.txt",
        "./UMLS/active_poisoning_whitebox/Keci/low_scores/1982/rel/4/train.txt",
            ]

    reference = "./UMLS/clean/train.txt"

    common_triples = common_triples_in_all_diffs(files, reference)
    print(f"Common triples in all runs but not in reference: {len(common_triples)} found")



