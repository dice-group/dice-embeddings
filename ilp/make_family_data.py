"""Generate a small *inductive* family knowledge graph for a motivating demo.

The point of this dataset is to show, in terms a non-technical audience can
follow, that the model learns a *rule* rather than memorizing people:

        parentOf(X, Y)       ⟹   olderThan(X, Y)
        grandparentOf(X, Y)  ⟹   olderThan(X, Y)

Every person is anonymized to a random slot at training time (see dataset.py),
and the training families and the test families share **no people at all**
(disjoint entity sets). So when the model answers "is this parent older than
their child?" on a brand-new family, it can only be applying a transferable
rule — it has never seen these individuals.

Layout produced (under <out>/):

    kg_family/train.txt           training families: ALL facts incl. olderThan
    kg_family_ind/train.txt       NEW families, *structure only* (the "context"
                                  the model is allowed to see at inference)
    kg_family_ind/test.txt        the held-out olderThan facts to be predicted

Relations
    parentOf       grandparentOf      marriedTo (symmetric context)
    siblingOf (symmetric context)     olderThan (the TARGET, derived from
                                      parentOf / grandparentOf)

Run:
    python -m ilp.make_family_data --out KGs --train-families 150 --test-families 30
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

Triple = tuple[str, str, str]

# Structural relations the model is allowed to see as context at inference.
STRUCT_RELS = ("parentOf", "grandparentOf", "marriedTo", "siblingOf")
# The relation we hold out and ask the model to predict.
TARGET_REL = "olderThan"


def _person(family: int, idx: int) -> str:
    # Globally-unique, opaque ids. Disjoint families never share a person, so
    # the train/test entity sets are guaranteed disjoint by construction.
    return f"fam{family:04d}_p{idx:02d}"


def build_family(family: int, rng: random.Random) -> tuple[list[Triple], list[Triple]]:
    """Build one three-generation family.

    Returns (structure, target):
      - structure: parentOf / grandparentOf / marriedTo / siblingOf facts
      - target:    olderThan facts derived from parentOf + grandparentOf

    The two lists are kept separate so the inference families can expose only
    structure to the model and hold the olderThan facts out as queries.
    """
    struct: list[Triple] = []
    target: list[Triple] = []
    next_idx = 0

    def new_person() -> str:
        nonlocal next_idx
        p = _person(family, next_idx)
        next_idx += 1
        return p

    def parent_of(p: str, c: str) -> None:
        struct.append((p, "parentOf", c))
        target.append((p, "olderThan", c))   # a parent is older than their child

    def grandparent_of(gp: str, gc: str) -> None:
        struct.append((gp, "grandparentOf", gc))
        target.append((gp, "olderThan", gc))  # a grandparent is older than grandchild

    # --- generation 0: a grandparent couple --------------------------------
    gp_a, gp_b = new_person(), new_person()
    struct.append((gp_a, "marriedTo", gp_b))

    # --- generation 1: their children (+ in-law spouses) -------------------
    n_children = rng.randint(2, 3)
    gen1 = [new_person() for _ in range(n_children)]
    for c in gen1:
        parent_of(gp_a, c)
        parent_of(gp_b, c)
    for a, b in zip(gen1, gen1[1:]):           # sibling chain among gen-1
        struct.append((a, "siblingOf", b))

    # --- generation 2: grandchildren ---------------------------------------
    for parent in gen1:
        if rng.random() < 0.85:                # most gen-1 children start a family
            spouse = new_person()
            struct.append((parent, "marriedTo", spouse))
            n_gc = rng.randint(1, 3)
            grandkids = [new_person() for _ in range(n_gc)]
            for gc in grandkids:
                parent_of(parent, gc)
                parent_of(spouse, gc)
                grandparent_of(gp_a, gc)
                grandparent_of(gp_b, gc)
            for a, b in zip(grandkids, grandkids[1:]):  # sibling chain among gen-2
                struct.append((a, "siblingOf", b))

    return struct, target


def write_triples(path: Path, triples: list[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{h}\t{r}\t{t}\n" for h, r, t in triples))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="KGs", help="Root output dir (default: KGs).")
    ap.add_argument("--train-families", type=int, default=150)
    ap.add_argument("--test-families", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)

    # Training families: the model sees the WHOLE family, olderThan included,
    # so it can learn the parentOf/grandparentOf → olderThan association.
    train: list[Triple] = []
    for f in range(args.train_families):
        struct, target = build_family(f, rng)
        train.extend(struct + target)
    write_triples(out / "kg_family" / "train.txt", train)

    # Inference families: a DISJOINT set of people (family ids offset so no id
    # ever collides). The model sees only the structure; olderThan is held out.
    offset = 10_000
    obs: list[Triple] = []
    test: list[Triple] = []
    for f in range(offset, offset + args.test_families):
        struct, target = build_family(f, rng)
        obs.extend(struct)
        test.extend(target)
    write_triples(out / "kg_family_ind" / "train.txt", obs)
    write_triples(out / "kg_family_ind" / "test.txt", test)

    # Sanity: assert disjoint entity sets (the inductive guarantee).
    def ents(ts: list[Triple]) -> set[str]:
        return {h for h, _, _ in ts} | {t for _, _, t in ts}
    overlap = ents(train) & (ents(obs) | ents(test))
    assert not overlap, f"entity leakage between train and test: {sorted(overlap)[:5]}"

    print(f"train families: {args.train_families:>4}  triples: {len(train):>6}  "
          f"entities: {len(ents(train)):>5}")
    print(f"test  families: {args.test_families:>4}  context triples: {len(obs):>6}  "
          f"held-out olderThan queries: {len(test):>5}  entities: {len(ents(obs) | ents(test)):>5}")
    print(f"entity overlap train∩test: {len(overlap)}  (must be 0 for inductive)")
    print(f"\nwrote:\n  {out/'kg_family'/'train.txt'}\n"
          f"  {out/'kg_family_ind'/'train.txt'}\n  {out/'kg_family_ind'/'test.txt'}")


if __name__ == "__main__":
    main()
