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

from .scorer import Scorer


def cmd_infer(args: argparse.Namespace) -> None:
    scorer = Scorer.from_bundle(args.model, args.data)
    for tail, s in scorer.rank(args.head, args.relation, k=args.k):
        print(f"{s:.4f}\t{tail}")


def cmd_score(args: argparse.Namespace) -> None:
    head, relation, tail = args.triple
    scorer = Scorer.from_bundle(args.model, args.data)
    print(f"{scorer.score(head, relation, tail):.4f}")


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
