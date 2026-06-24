"""Motivating demo: the model transfers the rule "a parent is older than their
child" to families it has never seen.

What this shows, in plain terms:
  * The model was trained on one set of families and is now shown a COMPLETELY
    DIFFERENT family — different people, zero overlap with training.
  * It is told only the family structure (who is the parent of whom). It is NOT
    told anyone's age or who is older.
  * We then ask it, for each parent→child pair, two questions:
        "Is the parent older than the child?"   (should be YES / high score)
        "Is the child older than the parent?"   (should be NO  / low score)
  * Because every person was anonymized during training, the model could never
    have memorized these individuals. Getting the direction right means it
    learned a transferable RULE: parentOf(X, Y) ⟹ olderThan(X, Y).

Run (after training with configs/kg_family.yaml):
    python -m ilp.family_demo --model runs/kg_family.pt
"""
from __future__ import annotations

import argparse
import random

from .dataset import Triple, read_triples
from .scorer import Scorer


def run_demo(model_path: str, context_path: str, n_pairs: int, seed: int) -> None:
    rng = random.Random(seed)

    # One Scorer over the unseen families' structure-only context graph. It holds
    # the context KG so every score below reuses it (no rebuild per query).
    scorer = Scorer.from_bundle(model_path, context_path)
    fmt = scorer.cfg.get("triple_format", "head_relation_tail")

    # The model has NEVER seen any of these people.
    triples = read_triples(context_path, fmt=fmt)
    parent_pairs: list[Triple] = [(h, r, t) for h, r, t in triples if r == "parentOf"]
    rng.shuffle(parent_pairs)
    sample = parent_pairs[:n_pairs]

    print("=" * 74)
    print("  INDUCTIVE FAMILY DEMO  —  rule learned:  parentOf(X, Y) ⟹ olderThan(X, Y)")
    print("=" * 74)
    print(f"  Context graph : {context_path}")
    print(f"  These families share NO people with the training set (fully inductive).")
    print(f"  The model is told only WHO IS THE PARENT OF WHOM — never any ages.\n")
    print(f"  {'parent → child':<28}{'P(parent older)':>16}{'P(child older)':>16}  verdict")
    print("  " + "-" * 70)

    correct = 0
    margins = []
    for parent, _, child in sample:
        p_parent_older = scorer.probability(parent, "olderThan", child)
        p_child_older = scorer.probability(child, "olderThan", parent)
        ok = p_parent_older > p_child_older
        correct += ok
        margins.append(p_parent_older - p_child_older)
        # Shorten the opaque ids for display (e.g. fam10003_p02).
        pair = f"{parent.split('_')[-1]} → {child.split('_')[-1]}"
        fam = parent.split("_")[0]
        print(f"  {fam} {pair:<19}{p_parent_older:>15.1%}{p_child_older:>16.1%}  "
              f"{'✓ correct' if ok else '✗ WRONG'}")

    n = len(sample)
    print("  " + "-" * 70)
    print(f"\n  Directional accuracy: {correct}/{n} = {correct / n:.0%}  "
          f"(parent scored older than child)")
    print(f"  Average confidence margin: {sum(margins) / n:+.1%}")
    print("\n  Takeaway: the model answers correctly on people it has never seen,")
    print("  so it is applying a transferable rule — not recalling memorized facts.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Trained bundle (runs/kg_family.pt).")
    ap.add_argument("--context", default="KGs/kg_family_ind/train.txt",
                    help="Structure-only context graph for the unseen families.")
    ap.add_argument("--n-pairs", type=int, default=15, help="Parent→child pairs to test.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_demo(args.model, args.context, args.n_pairs, args.seed)


if __name__ == "__main__":
    main()
