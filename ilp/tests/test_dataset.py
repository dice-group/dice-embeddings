"""Sanity checks on vocabulary and anonymization (spec §3, §4)."""
from __future__ import annotations

import random

import pytest

from ilp.dataset import KnowledgeGraph, build_sample, two_hop_neighborhood
from ilp.vocab import build_vocab


def _toy_triples():
    """A toy KG where:
      - `rdf:type` ranges over classes {Person, Company}
      - `gender` is controlled vocab {F, M}
      - `knows` has many distinct objects (> cutoff) → people remain instances
      - `worksAt` has one object {Acme} → Acme gets fixed
    """
    people = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"]
    triples = [(p, "rdf:type", "Person") for p in people]
    triples.append(("Acme", "rdf:type", "Company"))
    # alice knows everyone else — 7 distinct objects, > cutoff(=3)
    for p in people[1:]:
        triples.append(("alice", "knows", p))
    # everyone works at Acme — single object, < cutoff
    for p in people:
        triples.append((p, "worksAt", "Acme"))
    # gender — 2 values, controlled vocab
    for p in people[:4]:
        triples.append((p, "gender", "F"))
    for p in people[4:]:
        triples.append((p, "gender", "M"))
    return triples


CUTOFF = 3


def test_vocab_partitions_correctly():
    triples = _toy_triples()
    vocab, fixed_values = build_vocab(triples, z_pool_size=8, cardinality_cutoff=CUTOFF)
    # Classes appear as VAL
    assert "[VAL_Person]" in vocab
    assert "[VAL_Company]" in vocab
    # Controlled-vocab (gender has 2 values < cutoff) appear as VAL
    assert "[VAL_F]" in vocab and "[VAL_M]" in vocab
    # worksAt has 1 distinct object (Acme) < cutoff → Acme is fixed
    assert "[VAL_Acme]" in vocab
    # Instance people are NOT in vocab as VAL (knows-range > cutoff)
    assert "[VAL_alice]" not in vocab
    assert "[VAL_bob]" not in vocab
    # All relations present
    for r in {"rdf:type", "knows", "worksAt", "gender"}:
        assert f"[REL_{r}]" in vocab
    # Z pool
    for i in range(8):
        assert f"[Z_{i}]" in vocab
    assert vocab["[PAD]"] == 0
    assert vocab["[X]"] == 1


def test_type_relation_accepts_list():
    """`type_relation` may be a list of relations whose tails all become VAL.

    Toy KG designed so name-based promotion is the only mechanism that
    differentiates the three forms:
      - `rdf:type` → 1 tail (`Person`): always fixed (cardinality < cutoff)
      - `is_a`     → 5 tails (`C0`..`C4`): above cutoff=4, so only promoted
                     if `is_a` is named in `type_relation`
      - `knows`    → 50 tails: never promoted
    """
    triples = [("alice", "rdf:type", "Person")]
    triples += [(f"x{i}", "is_a", f"C{i % 5}") for i in range(10)]  # 5 classes
    triples += [("alice", "knows", f"u{i}") for i in range(50)]
    classes = {f"C{i}" for i in range(5)}

    # List form: Person + all 5 Cs promoted.
    _, fv_list = build_vocab(triples, z_pool_size=4, cardinality_cutoff=4,
                             type_relation=["rdf:type", "is_a"])
    assert "Person" in fv_list and classes.issubset(fv_list)

    # String form (back-compat): only Person promoted; is_a tails stay anon.
    _, fv_str = build_vocab(triples, z_pool_size=4, cardinality_cutoff=4,
                            type_relation="rdf:type")
    assert "Person" in fv_str and classes.isdisjoint(fv_str)

    # Empty / None: nothing promoted by name; Person still sneaks in via
    # cardinality (1 tail < cutoff). is_a tails stay anonymized.
    _, fv_none = build_vocab(triples, z_pool_size=4, cardinality_cutoff=4,
                             type_relation=None)
    assert "Person" in fv_none and classes.isdisjoint(fv_none)


def test_anonymization_marks_anchor_as_X():
    triples = _toy_triples()
    vocab, fixed_values = build_vocab(triples, z_pool_size=16, cardinality_cutoff=CUTOFF)
    kg = KnowledgeGraph(triples)
    sample = build_sample(
        "alice", "knows", "bob", 1.0, kg, vocab, fixed_values,
        max_triples=64, z_pool=16, rng=random.Random(0),
        exclude_triple=("alice", "knows", "bob"),
    )
    x_id = vocab["[X]"]
    z_ids = {vocab[f"[Z_{i}]"] for i in range(16)}
    n_valid = int(sample["mask"].sum())
    valid_triples = sample["triples"][:n_valid].tolist()
    flat = {tok for triple in valid_triples for tok in (triple[0], triple[2])}
    # alice → [X]
    assert x_id in flat
    # other people (instances) → some [Z_*]
    assert any(tok in z_ids for tok in flat)


def test_padding_and_mask_consistent():
    triples = _toy_triples()
    vocab, fixed_values = build_vocab(triples, z_pool_size=16, cardinality_cutoff=CUTOFF)
    kg = KnowledgeGraph(triples)
    sample = build_sample(
        "alice", "knows", "bob", 1.0, kg, vocab, fixed_values,
        max_triples=128, z_pool=16, rng=random.Random(0),
    )
    n_valid = int(sample["mask"].sum())
    pad_rows = sample["triples"][n_valid:]
    assert (pad_rows == 0).all()


def test_two_hop_neighborhood_basic():
    triples = _toy_triples()
    kg = KnowledgeGraph(triples)
    nb = two_hop_neighborhood("alice", kg)
    flat_entities = {s for s, _, _ in nb} | {o for _, _, o in nb}
    # 1-hop: bob..heidi via knows, Person via rdf:type, Acme via worksAt, F via gender
    assert "bob" in flat_entities
    assert "Acme" in flat_entities
    # 2-hop adds M via bob's gender
    assert "M" in flat_entities


def test_z_pool_exhaustion_raises():
    # Build a KG where alice's 2-hop has many distinct instance entities,
    # and shrink the Z pool so allocation must fail.
    big = [(f"x{i}", "knows", f"y{i}") for i in range(50)]
    big += [("alice", "knows", f"x{i}") for i in range(50)]
    vocab2, fixed_values2 = build_vocab(big, z_pool_size=4, cardinality_cutoff=3)
    kg2 = KnowledgeGraph(big)
    with pytest.raises(RuntimeError):
        build_sample(
            "alice", "knows", "x0", 1.0, kg2, vocab2, fixed_values2,
            max_triples=200, z_pool=4, rng=random.Random(0),
        )


def test_hop_distance_tokens_encode_bfs_distance():
    """With use_hop_distance_tokens=True, each entity's hop token reflects
    its BFS distance from the anchor. Anchor → HOP_DIST_0, neighbors →
    HOP_DIST_1, their neighbors → HOP_DIST_2. Flag off: every position is
    HOP_NONE (a single constant)."""
    from ilp.vocab import HOP_NONE

    triples = _toy_triples()
    vocab, fixed_values = build_vocab(triples, z_pool_size=16, cardinality_cutoff=CUTOFF)
    kg = KnowledgeGraph(triples)

    sample_off = build_sample(
        "alice", "knows", "bob", 1.0, kg, vocab, fixed_values,
        max_triples=128, z_pool=16, rng=random.Random(0),
        use_hop_distance_tokens=False,
    )
    sample_on = build_sample(
        "alice", "knows", "bob", 1.0, kg, vocab, fixed_values,
        max_triples=128, z_pool=16, rng=random.Random(0),
        use_hop_distance_tokens=True,
    )

    none_id = vocab[HOP_NONE]
    n_valid = int(sample_on["mask"].sum())

    # Flag off: every hop slot is HOP_NONE.
    assert (sample_off["hop_distances"] == none_id).all()

    # Flag on: relation positions stay HOP_NONE; entity positions vary.
    valid_hops = sample_on["hop_distances"][:n_valid]
    assert (valid_hops[:, 1] == none_id).all()  # relation column = HOP_NONE
    entity_hop_ids = set(valid_hops[:, [0, 2]].flatten().tolist())
    assert vocab["[HOP_DIST_0]"] in entity_hop_ids  # anchor appears
    # Both 1-hop and 2-hop neighbors exist in the toy KG.
    assert vocab["[HOP_DIST_1]"] in entity_hop_ids


def test_hop_distance_tokens_preserve_triple_alignment_after_shuffle():
    """After the in-build shuffle, each hop-distance triple must still
    describe its paired token triple — never the entities from a different
    row."""
    triples = _toy_triples()
    vocab, fixed_values = build_vocab(triples, z_pool_size=16, cardinality_cutoff=CUTOFF)
    kg = KnowledgeGraph(triples)
    sample = build_sample(
        "alice", "knows", "bob", 1.0, kg, vocab, fixed_values,
        max_triples=64, z_pool=16, rng=random.Random(42),
        use_hop_distance_tokens=True,
    )
    n_valid = int(sample["mask"].sum())
    x_id = vocab["[X]"]
    hop0 = vocab["[HOP_DIST_0]"]

    # Wherever the [X] token appears in the triples tensor, the paired hop
    # at the same position must be HOP_DIST_0 (the anchor's distance).
    triples_t = sample["triples"][:n_valid]
    hops_t = sample["hop_distances"][:n_valid]
    for i in range(n_valid):
        for pos in (0, 2):
            if triples_t[i, pos].item() == x_id:
                assert hops_t[i, pos].item() == hop0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
