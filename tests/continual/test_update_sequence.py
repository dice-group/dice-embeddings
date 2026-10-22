import numpy as np
import pytest

from dicee.continual.update_sequence import generate_update_sequence, save_update_sequence, load_update_sequence


def _toy_triples(seed=0, n_entities=20, n_relations=3, n_triples=120):
    rng = np.random.default_rng(seed)
    # Build a connected-ish random graph: chain + random extra edges.
    triples = set()
    for e in range(1, n_entities):
        triples.add((e - 1, int(rng.integers(0, n_relations)), e))
    while len(triples) < n_triples:
        h, t = rng.integers(0, n_entities, size=2)
        if h == t:
            continue
        triples.add((int(h), int(rng.integers(0, n_relations)), int(t)))
    return np.array(sorted(triples)[:n_triples], dtype=np.int64)


def test_existing_entities_scenario_no_new_entities():
    triples = _toy_triples()
    t0, steps = generate_update_sequence(
        triples, scenario="existing_entities", seed=1, initial_fraction=0.5, num_steps=3,
    )
    for s in steps:
        assert s.num_new_entities == 0
        assert s.num_new_relations == 0


def test_partition_covers_all_triples_exactly_once():
    triples = _toy_triples()
    t0, steps = generate_update_sequence(
        triples, scenario="existing_entities", seed=2, initial_fraction=0.4, num_steps=4,
    )
    total = len(t0) + sum(s.num_added_triples for s in steps)
    assert total == len(triples)
    # cumulative is monotonically increasing
    prev_len = len(t0)
    for s in steps:
        assert len(s.cumulative_triples) == prev_len + s.num_added_triples
        prev_len = len(s.cumulative_triples)
    assert prev_len == len(triples)


def test_new_entities_scenario_introduces_new_entities():
    triples = _toy_triples()
    t0, steps = generate_update_sequence(
        triples, scenario="new_entities", seed=3, initial_fraction=0.3, num_steps=3,
    )
    total_new = sum(s.num_new_entities for s in steps)
    assert total_new > 0
    total = len(t0) + sum(s.num_added_triples for s in steps)
    assert total == len(triples)


def test_mixed_scenario_runs():
    triples = _toy_triples()
    t0, steps = generate_update_sequence(
        triples, scenario="mixed", seed=4, initial_fraction=0.3, num_steps=3,
    )
    total = len(t0) + sum(s.num_added_triples for s in steps)
    assert total == len(triples)


def test_reproducibility_same_seed():
    triples = _toy_triples()
    t0a, stepsa = generate_update_sequence(triples, scenario="mixed", seed=42, num_steps=3)
    t0b, stepsb = generate_update_sequence(triples, scenario="mixed", seed=42, num_steps=3)
    assert t0a.triples == t0b.triples
    for sa, sb in zip(stepsa, stepsb):
        assert np.array_equal(sa.added_triples, sb.added_triples)


def test_invalid_scenario_raises():
    triples = _toy_triples()
    with pytest.raises(ValueError):
        generate_update_sequence(triples, scenario="bogus")


def test_save_and_load_roundtrip(tmp_path):
    triples = _toy_triples()
    t0, steps = generate_update_sequence(triples, scenario="existing_entities", seed=5, num_steps=3)
    save_update_sequence(str(tmp_path), t0, steps, config={"seed": 5})
    t0_loaded, steps_loaded, config = load_update_sequence(str(tmp_path))
    assert t0_loaded.triples == t0.triples
    assert config["seed"] == 5
    for a, b in zip(steps, steps_loaded):
        assert np.array_equal(a.added_triples, b.added_triples)
        assert a.num_new_entities == b.num_new_entities
