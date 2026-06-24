"""Self-contained motivating example: inductive transfer of a relational rule.

1.  Generates a synthetic family knowledge graph (the training KG). Every
    training family states the full structure (who is the parent of whom,
    who is married, …) **and** the target facts ``olderThan``, derived from
        parentOf(X, Y)       ⟹   olderThan(X, Y)
        grandparentOf(X, Y)  ⟹   olderThan(X, Y)

2.  Trains the inductive model on that KG. Because every person is
    anonymized to a random ``[Z_*]`` slot at every sample draw, the model
    cannot memorize individuals — it can only learn the *rule* relating
    ``parentOf`` to ``olderThan``.

3.  Evaluates on a second synthetic dataset shaped as a list of tuples
    ``(context_triples, query_triple)``. Each tuple is a brand-new family:
    its people never appeared in training (disjoint entity sets), but the
    relations are the same. The context exposes only the structure
    (``parentOf`` etc.); ``olderThan`` is held out as the query. The model
    must read the context and decide, for a never-seen parent/child pair,
    whether ``(parent, olderThan, child)`` holds.

4.  Asserts the model answers directionally correctly (parent scored older
    than child) on these unseen families — i.e. it transferred the rule.


Run with
``pytest ilp/tests/test_inductive_family.py -v -s`` to see the demo table.
"""
from __future__ import annotations

import random

import pytest
import torch

from ilp.dataset import Triple
from ilp.model import load_bundle
from ilp.scorer import Scorer
from ilp.train import train_model

# Relations the model is allowed to see as context at inference time.
STRUCT_RELS = ("parentOf", "grandparentOf", "marriedTo", "siblingOf")
# The relation we hold out and ask the model to predict.
TARGET_REL = "olderThan"


# --------------------------------------------------------------------------- #
#  Synthetic family-graph generation                                          #
# --------------------------------------------------------------------------- #
def _person(family: int, idx: int) -> str:
    # Opaque, globally-unique ids. Disjoint family numbers never share a person,
    # so train/eval entity sets are disjoint by construction (fully inductive).
    return f"fam{family:04d}_p{idx:02d}"


def build_family(family: int, rng: random.Random) -> tuple[list[Triple], list[Triple]]:
    """Build one three-generation family → (structure, target).

    ``structure``: parentOf / grandparentOf / marriedTo / siblingOf facts.
    ``target``:    olderThan facts derived from parentOf + grandparentOf.

    The two are kept separate so eval families can expose only the structure
    and hold the olderThan facts out as queries.
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
        target.append((p, TARGET_REL, c))        # a parent is older than their child

    def grandparent_of(gp: str, gc: str) -> None:
        struct.append((gp, "grandparentOf", gc))
        target.append((gp, TARGET_REL, gc))       # a grandparent is older than grandchild

    # generation 0: a grandparent couple
    gp_a, gp_b = new_person(), new_person()
    struct.append((gp_a, "marriedTo", gp_b))

    # generation 1: their children (+ in-law spouses)
    gen1 = [new_person() for _ in range(rng.randint(2, 3))]
    for c in gen1:
        parent_of(gp_a, c)
        parent_of(gp_b, c)
    for a, b in zip(gen1, gen1[1:]):              # sibling chain among gen-1
        struct.append((a, "siblingOf", b))

    # generation 2: grandchildren
    for parent in gen1:
        if rng.random() < 0.85:                   # most gen-1 children start a family
            spouse = new_person()
            struct.append((parent, "marriedTo", spouse))
            grandkids = [new_person() for _ in range(rng.randint(1, 3))]
            for gc in grandkids:
                parent_of(parent, gc)
                parent_of(spouse, gc)
                grandparent_of(gp_a, gc)
                grandparent_of(gp_b, gc)
            for a, b in zip(grandkids, grandkids[1:]):  # sibling chain among gen-2
                struct.append((a, "siblingOf", b))

    return struct, target


def make_train_triples(n_families: int, rng: random.Random) -> list[Triple]:
    """Training KG: full families, structure AND olderThan, so the rule is learnable."""
    train: list[Triple] = []
    for f in range(n_families):
        struct, target = build_family(f, rng)
        train += struct + target
    return train


def make_eval_tuples(
    n_families: int, rng: random.Random, family_offset: int = 10_000,
) -> list[tuple[list[Triple], Triple]]:
    """Inductive eval set: a list of ``(context_triples, query_triple)`` tuples.

    Each query is one held-out ``olderThan`` fact; its context is its family's
    structure-only graph (no olderThan). Family ids are offset so no person ever
    collides with the training set — unseen entities, seen relations.
    """
    eval_tuples: list[tuple[list[Triple], Triple]] = []
    for f in range(family_offset, family_offset + n_families):
        struct, target = build_family(f, rng)
        for query in target:
            eval_tuples.append((struct, query))     # all queries share the family's context
    return eval_tuples


def _entities(triples) -> set[str]:
    return {h for h, _, _ in triples} | {t for _, _, t in triples}



def _demo_cfg(data_dir) -> dict:
    return {
        # Model — small but expressive enough to learn the rule.
        "d_model": 64, "n_heads": 4, "n_triple_layers": 2, "n_sab": 2, "dropout": 0.1,
        # Optimization.
        "epochs": 20, "batch_size": 128, "lr": 3.0e-4, "weight_decay": 1.0e-2,
        "warmup_steps": 100, "grad_clip": 1.0,
        # Sampling.
        "max_triples": 64, "z_pool_size": 100,
        "cardinality_cutoff": 0, "tail_diversity_cutoff": 0.0,
        "neg_samples_per_pos": 4, "neg_sampler": "two_hop",
        "subgraph_hops": 2, "use_hop_distance_tokens": False, "collapse_z": False,
        # Dual subgraph: represent the candidate by its OWN k-hop subgraph, so a
        # never-seen child is grounded by structure rather than an empty token.
        "dual_subgraph": True,
        # No schema entities — every PERSON anonymizes to a random [Z_*] slot, so
        # the model literally cannot memorize people. 
        "type_relation": "",

        "triple_format": "head_relation_tail", "data_dir": str(data_dir),
        "run_name": "kg_family", "seed": 0, "num_workers": 4, "device": "cuda",
        "val_every": 1_000_000,
    }


@pytest.mark.slow
def test_inductive_family_rule_transfer(tmp_path):
    """Train on family A, transfer ``parentOf ⟹ olderThan`` to disjoint family B."""
    torch.manual_seed(0)
    gen_rng = random.Random(0)

    # 1. Generate data. Train KG on disk (train_model reads {data_dir}/train.txt).
    train_triples = make_train_triples(n_families=100, rng=gen_rng)
    eval_tuples = make_eval_tuples(n_families=15, rng=gen_rng)

    # Inductive guarantee: train and eval entity sets are disjoint.
    eval_ents = _entities([q for _, q in eval_tuples]) | _entities(
        [t for ctx, _ in eval_tuples for t in ctx]
    )
    assert not (_entities(train_triples) & eval_ents), "entity leakage breaks inductiveness"

    data_dir = tmp_path / "kg_family"
    data_dir.mkdir(parents=True)
    (data_dir / "train.txt").write_text(
        "".join(f"{h}\t{r}\t{t}\n" for h, r, t in train_triples)
    )

    # 2. Train. 
    cfg = _demo_cfg(data_dir)
    model_path = tmp_path / "model.pt"
    train_model(cfg, save_path=model_path, run_dir=None, eval_after=False)

    # 3. Inductive evaluation over the (context, query) tuples. One Scorer per
    #    distinct family context (queries that share a context reuse the Scorer).
    model, vocab, fixed_values, loaded_cfg, device = load_bundle(model_path)

    rows: list[tuple[str, str, float, float]] = []
    scorer: Scorer | None = None
    current_ctx_id: int | None = None
    for context, (parent, _rel, child) in eval_tuples:
        if id(context) != current_ctx_id:          # new family → new context graph
            scorer = Scorer.from_components(
                model, vocab, fixed_values, loaded_cfg, device, context
            )
            current_ctx_id = id(context)
        p_parent_older = scorer.probability(parent, TARGET_REL, child)
        p_child_older = scorer.probability(child, TARGET_REL, parent)
        rows.append((parent, child, p_parent_older, p_child_older))

    n = len(rows)
    correct = sum(1 for _p, _c, p_fwd, p_rev in rows if p_fwd > p_rev)
    mean_margin = sum(p_fwd - p_rev for _p, _c, p_fwd, p_rev in rows) / n
    acc = correct / n

    print("\n" + "=" * 74)
    print("  INDUCTIVE FAMILY DEMO  —  rule:  parentOf(X, Y) ⟹ olderThan(X, Y)")
    print("=" * 74)
    print("  Eval families share NO people with training (fully inductive).")
    print(f"  (showing first 20 of {n} queries)")
    print(f"  {'parent → child':<30}{'P(parent older)':>16}{'P(child older)':>16}")
    print("  " + "-" * 70)
    for parent, child, p_fwd, p_rev in rows[:20]:
        pair = f"{parent.split('_')[-1]} → {child.split('_')[-1]}"
        print(f"  {parent.split('_')[0]} {pair:<21}{p_fwd:>15.1%}{p_rev:>16.1%}")
    print("  " + "-" * 70)
    print(f"  Directional accuracy: {correct}/{n} = {acc:.0%}   "
          f"mean margin: {mean_margin:+.1%}")
    print("=" * 74)

    # 4. Assert the rule transferred to unseen people.
    assert acc >= 0.85, f"directional accuracy too low: {acc:.0%} ({correct}/{n})"
    assert mean_margin > 0.0, f"parent not scored older on average (margin={mean_margin:+.2%})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
