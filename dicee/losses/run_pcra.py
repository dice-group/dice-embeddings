import argparse
import os
import sys
import time
import random

def map_add(mp, key1, key2, value):
    if key1 not in mp:
        mp[key1] = {}
    if key2 not in mp[key1]:
        mp[key1][key2] = 0.0
    mp[key1][key2] += value


def map_add1(mp, key):
    if key not in mp:
        mp[key] = 0
    mp[key] += 1


def parse_triple(line, order):
    seg = line.strip().split()
    if len(seg) < 3:
        return None
    if order == "s r o":
        return seg[0], seg[1], seg[2]
    if order == "s o r":
        return seg[0], seg[2], seg[1]
    raise ValueError(f"Unsupported triple order: {order}")


def read_triples(path, order):
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = parse_triple(line, order)
            if parsed:
                h, r, t = parsed
                triples.append((h, r, t))
    return triples


def build_relation_mapping(triples):
    relation2id = {}
    id2relation = {}
    for _, r, _ in triples:
        if r not in relation2id:
            idx = len(relation2id)
            relation2id[r] = idx
            id2relation[idx] = r
    return relation2id, id2relation


def build_entity_mapping(triples):
    entity2id = {}
    id2entity = {}
    for h, _, t in triples:
        if h not in entity2id:
            idx = len(entity2id)
            entity2id[h] = idx
            id2entity[idx] = h
        if t not in entity2id:
            idx = len(entity2id)
            entity2id[t] = idx
            id2entity[idx] = t
    return entity2id, id2entity


def write_mapping(path, mapping):
    with open(path, "w") as f:
        for k, v in mapping.items():
            f.write(f"{k} {v}\n")


def resolve_batch_dataset_dirs(batch_root, datasets, max_perturbation):
    dirs = []
    for dataset_name in datasets:
        base = os.path.join(batch_root, dataset_name)
        if not os.path.isdir(base):
            continue
        for subdir in os.listdir(base):
            full_path = os.path.join(base, subdir)
            if not os.path.isdir(full_path):
                continue
            try:
                perturbation = float(subdir)
            except ValueError:
                continue
            if max_perturbation is not None and perturbation > max_perturbation:
                continue
            dirs.append((dataset_name, perturbation, full_path))
    dirs.sort(key=lambda x: (x[0], x[1]))
    return [p for _, _, p in dirs]


def generate_pra_for_dataset(args, dataset_dir):
    print(f"[PCRA] Processing dataset_dir={dataset_dir}")
    train_path = os.path.join(dataset_dir, "train.txt")
    valid_path = os.path.join(dataset_dir, "valid.txt")
    test_path = os.path.join(dataset_dir, "test.txt")
    e1e2_path = os.path.join(dataset_dir, "e1_e2.txt")

    train_triples = read_triples(train_path, args.triple_order)
    valid_triples = read_triples(valid_path, args.triple_order)
    test_triples = read_triples(test_path, args.triple_order)

    if not train_triples:
        raise FileNotFoundError(f"train.txt not found or empty in {dataset_dir}")

    relation2id, id2relation = build_relation_mapping(train_triples)
    relation_num = len(relation2id)
    for rid, rname in list(id2relation.items()):
        id2relation[rid + relation_num] = "~" + rname

    entity2id, id2entity = build_entity_mapping(train_triples)

    if args.write_mappings:
        write_mapping(os.path.join(dataset_dir, "relation2id.txt"), relation2id)
        write_mapping(os.path.join(dataset_dir, "entity2id.txt"), entity2id)

    ok = {}
    a = {}

    for h, r, t in train_triples:
        rel_id = relation2id[r]
        key_ht = f"{h} {t}"
        key_th = f"{t} {h}"
        if key_ht not in ok:
            ok[key_ht] = {}
        ok[key_ht][rel_id] = 1
        if key_th not in ok:
            ok[key_th] = {}
        ok[key_th][rel_id + relation_num] = 1

        if h not in a:
            a[h] = {}
        if rel_id not in a[h]:
            a[h][rel_id] = {}
        a[h][rel_id][t] = 1

        if t not in a:
            a[t] = {}
        if (rel_id + relation_num) not in a[t]:
            a[t][rel_id + relation_num] = {}
        a[t][rel_id + relation_num][h] = 1

    for h, _, t in test_triples:
        ok.setdefault(f"{h} {t}", {})
        ok.setdefault(f"{t} {h}", {})

    if os.path.exists(e1e2_path):
        with open(e1e2_path, "r") as f:
            for line in f:
                seg = line.strip().split()
                if len(seg) >= 2:
                    ok[f"{seg[0]} {seg[1]}"] = {}
                    ok[f"{seg[1]} {seg[0]}"] = {}

    path_dict = {}
    path_r_dict = {}
    train_path = {}
    h_e_p = {}

    step = 0
    time1 = time.time()
    path_num = 0

    for e1 in a:
        step += 1
        print(step, end=" ")
        for rel1 in a[e1]:
            e2_set = a[e1][rel1]
            for e2 in e2_set:
                map_add1(path_dict, str(rel1))
                for key in ok.get(f"{e1} {e2}", {}):
                    map_add1(path_r_dict, f"{rel1}->{key}")
                map_add(h_e_p, f"{e1} {e2}", str(rel1), 1.0 / len(e2_set))

        for rel1 in a[e1]:
            e2_set = a[e1][rel1]
            for e2 in e2_set:
                if e2 in a:
                    for rel2 in a[e2]:
                        e3_set = a[e2][rel2]
                        for e3 in e3_set:
                            map_add1(path_dict, f"{rel1} {rel2}")
                            if f"{e1} {e3}" in ok:
                                for key in ok[f"{e1} {e3}"]:
                                    map_add1(path_r_dict, f"{rel1} {rel2}->{key}")
                            if f"{e1} {e3}" in ok:
                                map_add(
                                    h_e_p,
                                    f"{e1} {e3}",
                                    f"{rel1} {rel2}",
                                    h_e_p[f"{e1} {e2}"][str(rel1)] * 1.0 / len(e3_set),
                                )

        for e2 in a:
            if f"{e1} {e2}" in h_e_p:
                path_num += len(h_e_p[f"{e1} {e2}"])
                bb = {}
                aa = {}
                sum_val = 0.0
                for rel_path in h_e_p[f"{e1} {e2}"]:
                    bb[rel_path] = h_e_p[f"{e1} {e2}"][rel_path]
                    sum_val += bb[rel_path]
                for rel_path in bb:
                    bb[rel_path] /= sum_val
                    if bb[rel_path] > args.min_prob:
                        aa[rel_path] = bb[rel_path]
                train_path.update({k: 1 for k in aa})
        print(path_num, time.time() - time1)
        sys.stdout.flush()

    if args.write_path_stats:
        path2_path = os.path.join(dataset_dir, "path2.txt")
        with open(path2_path, "w") as g:
            for e1 in a:
                for e2 in a:
                    if f"{e1} {e2}" in h_e_p:
                        g.write(f"{e1} {e2}\n")
                        bb = {}
                        aa = {}
                        sum_val = 0.0
                        for rel_path in h_e_p[f"{e1} {e2}"]:
                            bb[rel_path] = h_e_p[f"{e1} {e2}"][rel_path]
                            sum_val += bb[rel_path]
                        for rel_path in bb:
                            bb[rel_path] /= sum_val
                            if bb[rel_path] > args.min_prob:
                                aa[rel_path] = bb[rel_path]
                        g.write(str(len(aa)))
                        for rel_path in aa:
                            g.write(f" {len(rel_path.split())} {rel_path} {aa[rel_path]}")
                        g.write("\n")

        confidence_path = os.path.join(dataset_dir, "confidence.txt")
        with open(confidence_path, "w") as g:
            for rel_path in train_path:
                out = []
                for i in range(relation_num):
                    key = f"{rel_path}->{i}"
                    if rel_path in path_dict and key in path_r_dict:
                        out.append(f" {i} {path_r_dict[key] * 1.0 / path_dict[rel_path]}")
                if out:
                    g.write(f"{len(rel_path.split())} {rel_path}\n")
                    g.write(str(len(out)))
                    for item in out:
                        g.write(item)
                    g.write("\n")

    def write_pos_pra(name, triples):
        out_path = os.path.join(dataset_dir, f"{name}_pra.txt")
        if not triples:
            return
        with open(out_path, "w") as f_out:
            for e1, rel, e2 in triples:
                rel_id = relation2id[rel]
                f_out.write(f"{e1} {e2} {rel_id}\n")
                b = {}
                a_local = {}
                if f"{e1} {e2}" in h_e_p:
                    sum_val = 0.0
                    for rel_path in h_e_p[f"{e1} {e2}"]:
                        b[rel_path] = h_e_p[f"{e1} {e2}"][rel_path]
                        sum_val += b[rel_path]
                    for rel_path in b:
                        b[rel_path] /= sum_val
                        if b[rel_path] > args.min_prob:
                            a_local[rel_path] = b[rel_path]
                f_out.write(str(len(a_local)))
                for rel_path in a_local:
                    f_out.write(f" {len(rel_path.split())} {rel_path} {a_local[rel_path]}")
                f_out.write("\n")

                # reverse triple
                f_out.write(f"{e2} {e1} {rel_id + relation_num}\n")
                b = {}
                a_local = {}
                if f"{e2} {e1}" in h_e_p:
                    sum_val = 0.0
                    for rel_path in h_e_p[f"{e2} {e1}"]:
                        b[rel_path] = h_e_p[f"{e2} {e1}"][rel_path]
                        sum_val += b[rel_path]
                    for rel_path in b:
                        b[rel_path] /= sum_val
                        if b[rel_path] > args.min_prob:
                            a_local[rel_path] = b[rel_path]
                f_out.write(str(len(a_local)))
                for rel_path in a_local:
                    f_out.write(f" {len(rel_path.split())} {rel_path} {a_local[rel_path]}")
                f_out.write("\n")
        print(f"Wrote: {out_path}")

    def write_neg_pra(name, triples):
        out_path = os.path.join(dataset_dir, f"neg_{name}_pra.txt")
        if not triples:
            return
        rng = random.Random(args.seed)
        entities = list(entity2id.keys())
        with open(out_path, "w") as f_out:
            for h, r, t in triples:
                rel_id = relation2id[r]
                for _ in range(args.neg_ratio):
                    if rng.random() < 0.5:
                        h_neg = rng.choice(entities)
                        e1, e2 = h_neg, t
                    else:
                        t_neg = rng.choice(entities)
                        e1, e2 = h, t_neg
                    f_out.write(f"{e1} {e2} {rel_id}\n")
                    b = {}
                    a_local = {}
                    if f"{e1} {e2}" in h_e_p:
                        sum_val = 0.0
                        for rel_path in h_e_p[f"{e1} {e2}"]:
                            b[rel_path] = h_e_p[f"{e1} {e2}"][rel_path]
                            sum_val += b[rel_path]
                        for rel_path in b:
                            b[rel_path] /= sum_val
                            if b[rel_path] > args.min_prob:
                                a_local[rel_path] = b[rel_path]
                    f_out.write(str(len(a_local)))
                    for rel_path in a_local:
                        f_out.write(f" {len(rel_path.split())} {rel_path} {a_local[rel_path]}")
                    f_out.write("\n")
        print(f"Wrote: {out_path}")

    write_pos_pra("train", train_triples)
    # write_pos_pra("valid", valid_triples)
    # write_pos_pra("test", test_triples)
    write_neg_pra("train", train_triples)

    if args.write_path_stats:
        print(f"Wrote: {path2_path}")
        print(f"Wrote: {confidence_path}")


def main():
    parser = argparse.ArgumentParser(description="PCRA generator")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Single dataset dir containing train.txt/valid.txt/test.txt")
    parser.add_argument("--batch_root", type=str, default=None,
                        help="Root like Datasets_Perturbed for batch processing")
    parser.add_argument("--batch_datasets", type=str, nargs="+", default=["KINSHIP", "UMLS"],
                        help="Dataset names under --batch_root to process")
    parser.add_argument("--max_perturbation", type=float, default=None,
                        help="Only process perturbation folders <= this value (e.g., 0.32)")
    parser.add_argument("--triple_order", type=str, default="s r o",
                        help='"s r o" or "s o r"')
    parser.add_argument("--min_prob", type=float, default=0.01)
    parser.add_argument("--neg_ratio", type=int, default=1,
                        help="Number of negative triples per positive for neg_train_pra.txt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--write_mappings", action="store_true",
                        help="Write relation2id.txt and entity2id.txt")
    parser.add_argument("--write_path_stats", action="store_true",
                        help="Write path2.txt and confidence.txt")
    args = parser.parse_args()

    dataset_dirs = []
    if args.batch_root:
        dataset_dirs.extend(
            resolve_batch_dataset_dirs(
                batch_root=args.batch_root,
                datasets=args.batch_datasets,
                max_perturbation=args.max_perturbation,
            )
        )
    if args.dataset_dir:
        dataset_dirs.append(args.dataset_dir)

    # Keep order, remove duplicates.
    seen = set()
    dataset_dirs = [d for d in dataset_dirs if not (d in seen or seen.add(d))]

    if not dataset_dirs:
        raise ValueError("Provide --dataset_dir, or --batch_root with matching folders.")

    for dataset_dir in dataset_dirs:
        generate_pra_for_dataset(args=args, dataset_dir=dataset_dir)


if __name__ == "__main__":
    main()
