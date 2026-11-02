"""Generate multi-hop queries from a dataset and evaluate a trained KGE model on them.

Queries are sampled by dicee.QueryGenerator: easy answers are reachable in the train+valid graph and
hard answers only in the full train+valid+test graph. Per query, every hard answer is ranked against all
entities that are not answers, and MRR / Hits@k are averaged over hard answers, then over queries.

Example: python benchmarks/multihop_query_eval.py --path_to_model Experiments/<run> --dataset_dir KGs/UMLS --query_types 1p 2p 2i
"""
import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from dicee import KGE, QueryGenerator

QUERY_TYPES = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]


def write_tsv_splits(dataset_dir: Path, out_dir: Path) -> Dict[str, str]:
    """QueryGenerator reads tab-separated files, so rewrite each split as TSV."""
    paths = {}
    for split in ("train", "valid", "test"):
        df = pd.read_csv(dataset_dir / f"{split}.txt", sep=r"\s+", header=None, dtype=str)
        paths[split] = str(out_dir / f"{split}.tsv")
        df.to_csv(paths[split], sep="\t", header=False, index=False)
    return paths


def evaluate_query_type(model: KGE, generator: QueryGenerator, query_type: str, gen_num: int, k: int, tnorm: str,
                        use_logits: bool) -> Dict[str, float]:
    queries, easy, _, hard = generator.get_queries(query_type, gen_num)
    mrr, hits = [], {1: [], 3: [], 10: []}
    for query in (q for qs in queries.values() for q in qs):
        scores = model.answer_multi_hop_query(query_type=query_type, query=query, only_scores=True, k=k, tnorm=tnorm,
                                             use_logits=use_logits).detach().flatten().float()
        hard_idx = torch.tensor([model.entity_to_idx[e] for e in hard[query]])
        is_answer = torch.zeros(len(scores), dtype=torch.bool)
        is_answer[[model.entity_to_idx[e] for e in hard[query] | easy[query]]] = True
        # Filter every other answer, easy or hard: a hard answer competes only with non-answers.
        ranks = (1 + (scores[~is_answer][None, :] > scores[hard_idx][:, None]).sum(1)).tolist()
        mrr.append(sum(1 / r for r in ranks) / len(ranks))
        for h in hits:
            hits[h].append(sum(r <= h for r in ranks) / len(ranks))
    n = max(len(mrr), 1)
    return {"num_queries": len(mrr), "MRR": sum(mrr) / n, **{f"H@{h}": sum(v) / n for h, v in hits.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path_to_model", required=True, help="Directory of a trained dicee run.")
    p.add_argument("--dataset_dir", type=Path, required=True, help="Directory with train.txt, valid.txt and test.txt.")
    p.add_argument("--query_types", nargs="+", default=["1p", "2p", "3p", "2i", "3i"], choices=QUERY_TYPES)
    p.add_argument("--gen_num", type=int, default=100, help="Queries to sample per type.")
    p.add_argument("--k", type=int, default=10, help="Intermediate entities kept for multi-hop projections.")
    p.add_argument("--tnorm", default="min", choices=["prod", "min"], help="T-norm for intersections and unions.")
    p.add_argument("--logits", action="store_true", help="Combine raw logits instead of sigmoid probabilities (misranks with prod).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=None, help="Optional JSON file for the results.")
    cli = p.parse_args()

    model = KGE(path=cli.path_to_model)
    model.model.eval()
    # The generator pairs relation ids with r // 2, so give original relations even ids.
    ent2id = dict(model.entity_to_idx)
    rel2id = {r: 2 * i for i, r in enumerate(sorted(r for r in model.relation_to_idx if not r.endswith("_inverse")))}
    results: Dict[str, Dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        paths = write_tsv_splits(cli.dataset_dir, Path(tmp))
        generator = QueryGenerator(paths["train"], paths["valid"], paths["test"], ent2id=ent2id, rel2id=rel2id, seed=cli.seed)
        for query_type in cli.query_types:
            results[query_type] = evaluate_query_type(model, generator, query_type, cli.gen_num, cli.k, cli.tnorm, cli.logits)
            print(query_type, json.dumps(results[query_type]), flush=True)
    if cli.out:
        cli.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
