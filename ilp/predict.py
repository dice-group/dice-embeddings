"""CLI-first inference from a trained model bundle.

    # top-k tail predictions for (head, relation, ?)
    python -m ilp.predict infer --model bundle.pt --data KGs/mykg/train.txt \\
        --head Alice --relation knows --k 10

    # score a single (head, relation, tail) triple
    python -m ilp.predict score --model bundle.pt --data KGs/mykg/train.txt \\
        --triple Alice knows Bob

The `--data` triples build the KG context used for subgraph extraction.
"""
from __future__ import annotations

import argparse

from .dataset import KnowledgeGraph, augment_with_inverse, read_triples
from .eval import filter_known_relations, score_candidates
from .model import load_bundle


def _context_kg(data_path: str, vocab: dict, fmt: str) -> KnowledgeGraph:
    triples = read_triples(data_path, fmt=fmt)
    triples, _ = filter_known_relations(triples, vocab)
    return KnowledgeGraph(augment_with_inverse(triples))


def cmd_infer(args: argparse.Namespace) -> None:
    model, vocab, fixed_values, cfg, device = load_bundle(args.model)
    fmt = cfg.get("triple_format", "head_relation_tail")
    kg = _context_kg(args.data, vocab, fmt)
    scores = score_candidates(
        model, args.head, args.relation, sorted(kg.entities), kg, vocab, fixed_values,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        batch_size=128, device=device,
        collapse_z=cfg.get("collapse_z", False),
        subgraph_hops=cfg.get("subgraph_hops", 2),
        use_hop_distance_tokens=cfg.get("use_hop_distance_tokens", False),
    )
    for tail, s in sorted(scores.items(), key=lambda kv: -kv[1])[: args.k]:
        print(f"{s:.4f}\t{tail}")


def cmd_score(args: argparse.Namespace) -> None:
    head, relation, tail = args.triple
    model, vocab, fixed_values, cfg, device = load_bundle(args.model)
    fmt = cfg.get("triple_format", "head_relation_tail")
    kg = _context_kg(args.data, vocab, fmt)
    scores = score_candidates(
        model, head, relation, [tail], kg, vocab, fixed_values,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        batch_size=1, device=device,
        collapse_z=cfg.get("collapse_z", False),
        subgraph_hops=cfg.get("subgraph_hops", 2),
        use_hop_distance_tokens=cfg.get("use_hop_distance_tokens", False),
    )
    print(f"{scores[tail]:.4f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ilp.predict", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("infer", help="Top-k tail predictions for (head, relation, ?).")
    i.add_argument("--model", required=True, help="Bundled .pt (model + vocab + cfg).")
    i.add_argument("--data", required=True, help="Triples file for KG context.")
    i.add_argument("--head", required=True)
    i.add_argument("--relation", required=True)
    i.add_argument("--k", type=int, default=10)
    i.set_defaults(func=cmd_infer)

    s = sub.add_parser("score", help="Score a single (head, relation, tail) triple.")
    s.add_argument("--model", required=True, help="Bundled .pt (model + vocab + cfg).")
    s.add_argument("--data", required=True, help="Triples file for KG context.")
    s.add_argument("--triple", nargs=3, metavar=("HEAD", "RELATION", "TAIL"), required=True)
    s.set_defaults(func=cmd_score)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
