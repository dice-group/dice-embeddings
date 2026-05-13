"""Inference and scoring with a trained GraphPFN model.

Subcommands
-----------
infer
    Rank all candidate entities for a (head, relation, ?) query.

score
    Estimate P(triple is true) via N Monte Carlo context passes.

Examples
--------
Predict top-5 tail predictions (context_size auto-loaded from checkpoint)::

    python pfn_infer.py infer \\
        --model model.pt \\
        --train-file KGs/Countries-S1/train.txt \\
        --head slovakia --relation neighbor --k 5

Predict top-10 (default k) with an explicit context cap::

    python pfn_infer.py infer \\
        --model model.pt \\
        --train-file KGs/Countries-S1/train.txt \\
        --head slovakia --relation neighbor \\
        --context-size 64

Cap support to 128 triples regardless of context_size in the checkpoint::

    python pfn_infer.py infer \\
        --model model.pt \\
        --train-file KGs/Countries-S1/train.txt \\
        --head slovakia --relation neighbor \\
        --support-size 128 --k 10

Print the support triples before scoring::

    python pfn_infer.py infer \\
        --model model.pt \\
        --train-file KGs/Countries-S1/train.txt \\
        --head slovakia --relation neighbor \\
        --show-support --k 5

Backward-compatible two-argument --query form::

    python pfn_infer.py infer \\
        --model model.pt --train-file KGs/Countries-S1/train.txt \\
        --query slovakia neighbor --k 5

Score a fully-specified triple (10 passes, context size 32)::

    python pfn_infer.py score \\
        --model model.pt \\
        --data KGs/Countries-S1/train.txt \\
        --triple slovakia neighbor austria \\
        --n 10 --context-size 32

Score with more passes for a tighter estimate::

    python pfn_infer.py score \\
        --model model.pt \\
        --data KGs/UMLS/train.txt \\
        --triple concept_disease treatment concept_drug \\
        --n 50 --context-size 64
"""

import argparse
import sys
from typing import List, Tuple

import torch

from pfn_inference import infer, score_triple
from pfn_model import TriplePFN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: torch.device) -> Tuple[TriplePFN, dict]:
    """Load a TriplePFN checkpoint; return (model, run_config)."""
    ckpt = torch.load(model_path, map_location=device)
    run_cfg = ckpt.get("run_config", {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt, dict) and "hparams" in ckpt:
        model = TriplePFN(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = TriplePFN()
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, run_cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GraphPFN inference and scoring.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # infer subcommand
    # ------------------------------------------------------------------
    infer_parser = subparsers.add_parser(
        "infer",
        help="Predict top-k tail entities for a (head, relation, ?) query.",
        description=(
            "Load a trained GraphPFN model, use a KG file as in-context support, "
            "and rank all candidate entities for a given (head, relation) query."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt).")
    infer_parser.add_argument(
        "--train-file", "--data", dest="train_file", type=str, required=False, metavar="TRAIN_TXT",
        help="Path to train.txt whose triples are used as in-context support.",
    )
    infer_parser.add_argument("--head", type=str, required=False, help="Head entity string for the query.")
    infer_parser.add_argument("--relation", type=str, required=False, help="Relation string for the query.")
    infer_parser.add_argument(
        "--query", type=str, nargs=2, required=False, metavar=("HEAD", "RELATION"),
        help="Backward-compatible: --query <head> <relation>.",
    )
    infer_parser.add_argument("--k", type=int, default=10, help="Number of top-k predictions to display.")
    infer_parser.add_argument(
        "--context-size", type=int, default=None, metavar="N",
        help="Max support triples used. Defaults to the value stored in the checkpoint.",
    )
    infer_parser.add_argument(
        "--support-size", type=int, default=None, metavar="N",
        help="Optional additional cap on support triples (applied after --context-size).",
    )
    infer_parser.add_argument(
        "--show-support", action="store_true",
        help="Print support triples before scoring.",
    )

    # ------------------------------------------------------------------
    # score subcommand
    # ------------------------------------------------------------------
    score_parser = subparsers.add_parser(
        "score",
        help="Estimate P(triple is true) via Monte Carlo context sampling.",
    )
    score_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt).")
    score_parser.add_argument(
        "--triple", type=str, nargs=3, required=True, metavar=("HEAD", "RELATION", "TAIL"),
        help="Triple to score.",
    )
    score_parser.add_argument("--data", type=str, required=True, help="Data file for context sampling.")
    score_parser.add_argument("--n", type=int, default=10, help="Number of independent passes.")
    score_parser.add_argument("--context-size", type=int, default=32, help="Support triples per pass.")

    # Default to infer if no subcommand given
    if len(sys.argv) == 1 or sys.argv[1] not in ("infer", "score", "-h", "--help"):
        sys.argv.insert(1, "infer")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # infer
    # ------------------------------------------------------------------
    if args.command == "infer":
        # Resolve --query alias
        if args.query is not None:
            if args.head is None:
                args.head = args.query[0]
            if args.relation is None:
                args.relation = args.query[1]

        missing = []
        if args.train_file is None:
            missing.append("--train-file/--data")
        if args.head is None:
            missing.append("--head or --query")
        if args.relation is None:
            missing.append("--relation or --query")
        if missing:
            infer_parser.error("the following arguments are required: " + ", ".join(missing))
        if args.support_size is not None and args.support_size <= 0:
            infer_parser.error("--support-size must be a positive integer")

        model, run_cfg = _load_model(args.model, device)

        if args.context_size is None:
            args.context_size = int(run_cfg.get("context_size", 128))
            print(f"Using checkpoint context_size={args.context_size} for inference.")
        if args.context_size <= 0:
            infer_parser.error("--context-size must be a positive integer")

        if run_cfg:
            print(
                "Loaded checkpoint run settings: "
                f"context_size={run_cfg.get('context_size')}, "
                f"support_sampler={run_cfg.get('support_sampler')}, "
                f"permute_support={run_cfg.get('permute_support')}"
            )

        # Load support triples
        support: List[Tuple[str, str, str]] = []
        with open(args.train_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 3:
                    support.append((parts[0], parts[1], parts[2]))

        total_support = len(support)
        effective_cap = args.context_size
        if args.support_size is not None:
            effective_cap = min(effective_cap, args.support_size)

        if total_support > effective_cap:
            support = support[:effective_cap]
            print(
                f"Support capped: {total_support:,} -> {len(support):,} triples "
                f"(context_size={args.context_size}, support_size={args.support_size})."
            )
        else:
            print(f"Support loaded: {len(support):,} triples from '{args.train_file}'.")

        if args.show_support:
            print(f"\n{'─'*52}")
            print(f"  {'#':<5}  {'Head':<20}  {'Relation':<15}  Tail")
            print(f"  {'─'*5}  {'─'*20}  {'─'*15}  {'─'*20}")
            for i, (h, r, t) in enumerate(support, start=1):
                print(f"  {i:<5}  {h:<20}  {r:<15}  {t}")
            print(f"{'─'*52}\n")

        results = infer(model, args.head, args.relation, support, k=args.k, device=device)

        print(f"\nTop-{args.k} predictions for ({args.head}, {args.relation}, ?):")
        print(f"  {'Entity':<30}  Log P(true)")
        print(f"  {'-'*30}  -----------")
        for rank, (entity, raw_score) in enumerate(results, start=1):
            log_p_true = torch.nn.functional.logsigmoid(torch.tensor(raw_score)).item()
            print(f"  {rank}. {entity:<28}  {log_p_true:>11.4f}")

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------
    elif args.command == "score":
        ckpt = torch.load(args.model, map_location="cpu")
        if isinstance(ckpt, dict) and "hparams" in ckpt:
            model = TriplePFN(**ckpt["hparams"])
            model.load_state_dict(ckpt["state_dict"])
        else:
            model = TriplePFN()
            model.load_state_dict(ckpt)
        model.eval()

        h, r, t = args.triple
        prob = score_triple(model, h, r, t, args.data, n=args.n, context_size=args.context_size)
        print(f"P({h} {r} {t}) = {prob:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
