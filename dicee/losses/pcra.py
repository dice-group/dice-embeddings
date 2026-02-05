import os
from typing import Dict, Iterable, List, Tuple

from dicee.losses.custom_losses import compute_prior_path_confidence


def _map_add(mp, key1, key2, value):
    if key1 not in mp:
        mp[key1] = {}
    if key2 not in mp[key1]:
        mp[key1][key2] = 0.0
    mp[key1][key2] += value


def _map_add1(mp, key):
    if key not in mp:
        mp[key] = 0
    mp[key] += 1


def _parse_triple(line: str, order: str) -> Tuple[str, str, str]:
    seg = line.strip().split()
    if len(seg) < 3:
        raise ValueError(f"Bad triple line: {line!r}")
    if order == "s r o":
        return seg[0], seg[1], seg[2]
    if order == "s o r":
        return seg[0], seg[2], seg[1]
    raise ValueError(f"Unsupported triple order: {order}")


def _read_triples(path: str, order: str) -> List[Tuple[str, str, str]]:
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            h, r, t = _parse_triple(line, order)
            triples.append((h, r, t))
    return triples


def _to_index_dict(mapping, key_name: str):
    if isinstance(mapping, dict):
        # If mapping is id->label, invert to label->id
        if mapping and all(isinstance(k, int) for k in mapping.keys()):
            return {v: k for k, v in mapping.items()}
        return mapping
    # pandas Series -> dict
    if hasattr(mapping, "to_dict") and not hasattr(mapping, "columns"):
        series_dict = mapping.to_dict()
        if series_dict and all(isinstance(k, int) for k in series_dict.keys()):
            return {v: k for k, v in series_dict.items()}
        return series_dict
    df = mapping
    if hasattr(mapping, "to_pandas"):
        df = mapping.to_pandas()
    if hasattr(df, "columns"):
        if key_name in df.columns and len(df.columns) == 1:
            return dict(zip(df[key_name], df.index))
        if key_name in df.columns and "index" in df.columns:
            return dict(zip(df[key_name], df["index"]))
        if len(df.columns) >= 2:
            return dict(zip(df[df.columns[0]], df[df.columns[1]]))
    raise TypeError(f"Unsupported mapping type for {key_name}: {type(mapping)}")


def _build_graph(
    triples: Iterable[Tuple[str, str, str]],
    relation_to_idx: Dict[str, int],
    relation_num: int,
):
    ok = {}
    adjacency = {}
    for h, r, t in triples:
        if r not in relation_to_idx:
            raise ValueError(
                f"Relation '{r}' not found in relation_to_idx. "
                f"Check --pcra_triple_order (expected 's r o' vs 's o r') "
                f"or ensure relation mappings include this relation."
            )
        r_id = relation_to_idx[r]
        key_ht = f"{h} {t}"
        key_th = f"{t} {h}"
        if key_ht not in ok:
            ok[key_ht] = {}
        ok[key_ht][r_id] = 1
        if key_th not in ok:
            ok[key_th] = {}
        ok[key_th][r_id + relation_num] = 1

        if h not in adjacency:
            adjacency[h] = {}
        adjacency[h].setdefault(r_id, {})[t] = 1
        if t not in adjacency:
            adjacency[t] = {}
        adjacency[t].setdefault(r_id + relation_num, {})[h] = 1
    return ok, adjacency


def _ensure_ok_pairs(ok, pairs: Iterable[Tuple[str, str]]):
    for h, t in pairs:
        key_ht = f"{h} {t}"
        key_th = f"{t} {h}"
        ok.setdefault(key_ht, {})
        ok.setdefault(key_th, {})


def _normalize_path_map(h_e_p, min_prob: float):
    normalized = {}
    for pair, path_map in h_e_p.items():
        total = sum(path_map.values())
        if total <= 0:
            continue
        kept = {}
        for rel_path, val in path_map.items():
            val = val / total
            if val > min_prob:
                kept[rel_path] = val
        if kept:
            normalized[pair] = kept
    return normalized


def compute_pcra(
    dataset_dir: str,
    relation_to_idx: Dict[str, int],
    triple_order: str = "s r o",
    min_prob: float = 0.01,
):
    train_path = os.path.join(dataset_dir, "train.txt")
    test_path = os.path.join(dataset_dir, "test.txt")
    e1e2_path = os.path.join(dataset_dir, "e1_e2.txt")

    train_triples = _read_triples(train_path, triple_order)
    if not train_triples:
        return {}, {}, {}, 0

    if not set(r for _, r, _ in train_triples[:50]).issubset(set(relation_to_idx.keys())):
        relation_to_idx = {r: i for i, r in enumerate(sorted(set(r for _, r, _ in train_triples)))}
        print("[PCRA] Rebuilt relation_to_idx from train.txt for PP/AP computation.")
    else:
        print("[PCRA] Using provided relation_to_idx (keys match train.txt).")

    relation_num = len(relation_to_idx)
    try:
        ok, adjacency = _build_graph(train_triples, relation_to_idx, relation_num)
    except ValueError as e:
        relation_to_idx = {r: i for i, r in enumerate(sorted(set(r for _, r, _ in train_triples)))}
        relation_num = len(relation_to_idx)
        print("[PCRA] Rebuilt relation_to_idx after ValueError in _build_graph.")
        ok, adjacency = _build_graph(train_triples, relation_to_idx, relation_num)

    test_triples = _read_triples(test_path, triple_order)
    _ensure_ok_pairs(ok, [(h, t) for h, _, t in test_triples])

    if os.path.exists(e1e2_path):
        with open(e1e2_path, "r") as f:
            pairs = [tuple(line.strip().split()[:2]) for line in f if line.strip()]
        _ensure_ok_pairs(ok, pairs)

    path_dict = {}
    path_r_dict = {}
    h_e_p = {}

    for h in adjacency:
        for rel1 in adjacency[h]:
            e2_set = adjacency[h][rel1]
            for e2 in e2_set:
                path1 = str(rel1)
                _map_add1(path_dict, path1)
                for key in ok.get(f"{h} {e2}", {}):
                    _map_add1(path_r_dict, (path1, key))
                _map_add(h_e_p, f"{h} {e2}", path1, 1.0 / len(e2_set))

        for rel1 in adjacency[h]:
            e2_set = adjacency[h][rel1]
            for e2 in e2_set:
                if e2 in adjacency:
                    for rel2 in adjacency[e2]:
                        e3_set = adjacency[e2][rel2]
                        path2 = f"{rel1} {rel2}"
                        for e3 in e3_set:
                            _map_add1(path_dict, path2)
                            if f"{h} {e3}" in ok:
                                for key in ok[f"{h} {e3}"]:
                                    _map_add1(path_r_dict, (path2, key))
                            if f"{h} {e3}" in ok:
                                _map_add(
                                    h_e_p,
                                    f"{h} {e3}",
                                    path2,
                                    h_e_p[f"{h} {e2}"][str(rel1)] * 1.0 / len(e3_set),
                                )

    h_e_p = _normalize_path_map(h_e_p, min_prob=min_prob)
    return h_e_p, path_dict, path_r_dict, relation_num


def compute_prior_confidence_map(
    dataset_dir: str,
    entity_to_idx: Dict[str, int],
    relation_to_idx: Dict[str, int],
    triple_order: str = "s r o",
    epsilon: float = 1e-6,
    min_prob: float = 0.01,
):
    entity_to_idx = _to_index_dict(entity_to_idx, "entity")
    relation_to_idx = _to_index_dict(relation_to_idx, "relation")

    train_path = os.path.join(dataset_dir, "train.txt")
    train_triples = _read_triples(train_path, triple_order)
    if not train_triples:
        return {}

    # Fallback: rebuild mappings if provided ones don't match the data
    sample_rels = {r for _, r, _ in train_triples[:50]}
    if not sample_rels.issubset(set(relation_to_idx.keys())):
        relation_to_idx = {r: i for i, r in enumerate(sorted(set(r for _, r, _ in train_triples)))}
    sample_ents = {h for h, _, _ in train_triples[:50]} | {t for _, _, t in train_triples[:50]}
    if not sample_ents.issubset(set(entity_to_idx.keys())):
        entity_to_idx = {e: i for i, e in enumerate(sorted(set([h for h, _, _ in train_triples] + [t for _, _, t in train_triples])))}

    h_e_p, path_dict, path_r_dict, _ = compute_pcra(
        dataset_dir=dataset_dir,
        relation_to_idx=relation_to_idx,
        triple_order=triple_order,
        min_prob=min_prob,
    )

    pp_map = {}
    for h, r, t in train_triples:
        if h not in entity_to_idx or t not in entity_to_idx or r not in relation_to_idx:
            continue
        pair_key = f"{h} {t}"
        path_map = h_e_p.get(pair_key)
        if not path_map:
            continue
        path_set = [(path_str, path_map[path_str]) for path_str in path_map]

        rel_path_prior = {}
        path_prior = {}
        r_id = relation_to_idx[r]
        for path_str in path_map:
            rel_path_prior[path_str] = path_r_dict.get((path_str, r_id), 0.0)
            path_prior[path_str] = path_dict.get(path_str, 0.0)

        pp_value = compute_prior_path_confidence(
            path_set=path_set,
            rel_path_prior=rel_path_prior,
            path_prior=path_prior,
            epsilon=epsilon,
        )
        key = (entity_to_idx[h], r_id, entity_to_idx[t])
        pp_map[key] = pp_value

    return pp_map


def load_pra_paths(dataset_dir, entity_to_idx, relation_to_idx):
    """
    Builds the necessary files for AP

    Example:
    path_data[(10, 3, 25)] = [
    ([7,9], 0.5),
    ([12], 0.2)]

    for the triple with IDs (head=10, relation=3, tail=25), there are two relation‑paths between the head and tail:

    Path [7, 9] with reliability/probability 0.5
    Path [12] with reliability/probability 0.2
    Those paths are what AP uses to compute adaptive confidence for that triple.

    Paths are used in In LocalTripleWithPriorAndAdaptivePathLoss: _get_adaptive_confidence uses the paths to compute AP

    """
    entity_to_idx = _to_index_dict(entity_to_idx, "entity")
    relation_to_idx = _to_index_dict(relation_to_idx, "relation")
    path_data = {}

    def _parse_pra_file(path):
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            tokens = f.read().strip().split()
        i = 0
        n = len(tokens)
        while i < n:
            h = tokens[i]; t = tokens[i + 1]; rel = tokens[i + 2]
            i += 3
            try:
                rel_id = int(rel)
            except ValueError:
                rel_id = relation_to_idx.get(rel, None)
            if rel_id is None:
                # skip this triple block
                path_count = int(tokens[i]); i += 1
                for _ in range(path_count):
                    path_len = int(tokens[i]); i += 1
                    i += path_len
                    i += 1
                continue
            path_count = int(tokens[i]); i += 1
            triple_key = (entity_to_idx.get(h), rel_id, entity_to_idx.get(t))
            paths = []
            for _ in range(path_count):
                path_len = int(tokens[i]); i += 1
                rel_path = [int(tokens[i + j]) for j in range(path_len)]
                i += path_len
                pr = float(tokens[i]); i += 1
                paths.append((rel_path, pr))
            if triple_key[0] is not None and triple_key[2] is not None:
                if triple_key not in path_data:
                    path_data[triple_key] = []
                path_data[triple_key].extend(paths)

    _parse_pra_file(os.path.join(dataset_dir, "train_pra.txt"))
    _parse_pra_file(os.path.join(dataset_dir, "neg_train_pra.txt"))
    return path_data
