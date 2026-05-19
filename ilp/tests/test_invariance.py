"""Spec §7.2 invariance regression tests.

`test_permutation_invariance` is architectural and must pass at init.
`test_z_relabeling_invariance` only passes after Z-randomized training.
"""
from __future__ import annotations

import random

import pytest
import torch

from ilp.dataset import KnowledgeGraph, build_sample
from ilp.eval import (
    test_permutation_invariance as _perm_check,
    test_z_relabeling_invariance as _z_check,
)
from ilp.model import InductiveKGModel
from ilp.vocab import build_vocab


def _toy_kg_and_vocab(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    # Tiny synthetic KG: 12 instance entities, 3 classes, 4 relations, including rdf:type.
    classes = ["ClassA", "ClassB", "ClassC"]
    ents = [f"e{i}" for i in range(12)]
    triples = []
    for e in ents:
        triples.append((e, "rdf:type", random.choice(classes)))
    rels = ["knows", "worksAt", "bornIn"]
    for _ in range(60):
        s, o = random.sample(ents, 2)
        r = random.choice(rels)
        triples.append((s, r, o))
    vocab, fixed_values = build_vocab(triples, z_pool_size=32, cardinality_cutoff=100)
    kg = KnowledgeGraph(triples)
    return triples, kg, vocab, fixed_values


def _toy_sample(kg, vocab, fixed_values, z_pool=32):
    h, r, t = "e0", "knows", "e1"
    return build_sample(
        h, r, t, 1.0, kg, vocab, fixed_values,
        max_triples=64, z_pool=z_pool, rng=random.Random(0),
    )


def _toy_model(vocab, dropout=0.0):
    # dropout=0 so eval mode is deterministic — strictly not needed since
    # the checker calls model.eval(), but defensive.
    return InductiveKGModel(
        vocab_size=len(vocab),
        x_token_id=vocab["[X]"],
        d_model=32,
        n_heads=4,
        n_triple_layers=2,
        n_sab=2,
        dropout=dropout,
    )


def test_permutation_invariance_at_init():
    _, kg, vocab, fixed_values = _toy_kg_and_vocab()
    sample = _toy_sample(kg, vocab, fixed_values)
    model = _toy_model(vocab)
    ok, diff = _perm_check(model, sample, atol=1e-4, rng=random.Random(1))
    assert ok, f"Permutation invariance violated at init (diff={diff:.2e})"


def test_z_relabeling_invariance_at_init_should_fail_or_pass_within_loose_tol():
    """At init this generally won't hold tightly — we just check it doesn't crash."""
    _, kg, vocab, fixed_values = _toy_kg_and_vocab()
    sample = _toy_sample(kg, vocab, fixed_values)
    model = _toy_model(vocab)
    ok, diff = _z_check(model, sample, vocab, z_pool_size=32, atol=1e-3, rng=random.Random(2))
    # Don't assert ok at init; just record. Document expectation.
    print(f"[init] z-relabel diff = {diff:.3e} (expected to shrink after training)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
