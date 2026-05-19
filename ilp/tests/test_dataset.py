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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
