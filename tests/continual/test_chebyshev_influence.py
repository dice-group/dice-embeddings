import numpy as np

from dicee.continual.chebyshev import build_normalized_adjacency, chebyshev_propagate
from dicee.continual.influence import compute_edit_signals, compute_influence_scores


def test_adjacency_isolated_nodes_zero_degree():
    triples = np.array([[0, 0, 1]])
    l_tilde = build_normalized_adjacency(triples, num_entities=5)
    dense = l_tilde.to_dense().numpy()
    # node 2,3,4 are isolated -> zero rows/cols
    assert np.allclose(dense[2], 0)
    assert np.allclose(dense[:, 2], 0)
    assert dense[0, 1] != 0
    assert dense[1, 0] != 0


def test_recurrence_k0_k1():
    triples = np.array([[0, 0, 1], [1, 0, 2]])
    l_tilde = build_normalized_adjacency(triples, num_entities=3)
    s0 = np.array([[1.0], [0.0], [0.0]])
    z = chebyshev_propagate(l_tilde, s0, K=0)
    assert len(z) == 1
    assert np.allclose(z[0], s0)

    z1 = chebyshev_propagate(l_tilde, s0, K=1)
    assert len(z1) == 2
    import torch
    expected = torch.sparse.mm(l_tilde, torch.as_tensor(s0))
    assert np.allclose(z1[1].numpy(), expected.numpy())


def test_recurrence_matches_manual_k2():
    triples = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
    l_tilde = build_normalized_adjacency(triples, num_entities=4)
    s0 = np.eye(4)
    z = chebyshev_propagate(l_tilde, s0, K=2)
    z0, z1, z2 = z
    import torch
    s0_t = torch.as_tensor(s0)
    expected_z1 = torch.sparse.mm(l_tilde, s0_t)
    expected_z2 = 2 * torch.sparse.mm(l_tilde, expected_z1) - z0
    assert np.allclose(z0.numpy(), s0)
    assert np.allclose(z1.numpy(), expected_z1.numpy())
    assert np.allclose(z2.numpy(), expected_z2.numpy())


def test_edit_signals_counts():
    added = np.array([[0, 0, 1], [1, 0, 2]])
    signals = compute_edit_signals(added, new_entity_ids=[2], num_entities=4)
    assert signals.added_count[0] == 1
    assert signals.added_count[1] == 2
    assert signals.added_count[2] == 1
    assert signals.is_new_entity[2] == 1.0
    assert signals.new_entity_count[1] == 1  # triple (1,0,2) touches new entity 2
    assert signals.new_entity_count[0] == 0  # triple (0,0,1) does not


def test_influence_zero_signal_gives_zero_scores():
    cumulative = np.array([[0, 0, 1], [1, 0, 2]])
    added = np.zeros((0, 3), dtype=np.int64)
    scores = compute_influence_scores(cumulative, added, new_entity_ids=[], num_entities=4, K=2)
    assert all(v == 0.0 for v in scores.values())


def test_influence_normalized_in_unit_range():
    cumulative = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
    added = np.array([[2, 0, 3]])
    scores = compute_influence_scores(cumulative, added, new_entity_ids=[3], num_entities=4, K=3, rho=0.8)
    assert max(scores.values()) <= 1.0 + 1e-6
    assert min(scores.values()) >= 0.0
    # entity 3 was directly, newly edited; it should receive non-trivial influence.
    assert scores[3] > 0.5


def test_influence_deterministic():
    cumulative = np.array([[0, 0, 1], [1, 0, 2]])
    added = np.array([[1, 0, 2]])
    s1 = compute_influence_scores(cumulative, added, [2], 3, K=3, rho=0.9)
    s2 = compute_influence_scores(cumulative, added, [2], 3, K=3, rho=0.9)
    assert s1 == s2


def test_influence_k0():
    cumulative = np.array([[0, 0, 1]])
    added = np.array([[0, 0, 1]])
    scores = compute_influence_scores(cumulative, added, [1], 2, K=0)
    assert len(scores) == 2
