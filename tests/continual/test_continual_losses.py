import numpy as np
import torch

from dicee.continual.continual_losses import preservation_weights, preservation_loss, sample_filtered_negatives
from dicee.models.real import DistMult


def _make_model(num_entities=5, num_relations=2, dim=8):
    args = {
        "model": "DistMult", "embedding_dim": dim, "num_entities": num_entities,
        "num_relations": num_relations, "learning_rate": 0.01, "lr": 0.01, "optim": "Adam",
        "scoring_technique": "KvsAll", "input_dropout_rate": 0.0, "hidden_dropout_rate": 0.0,
        "weight_decay": 0.0, "normalization": None, "byte_pair_encoding": False,
    }
    return DistMult(args)


def test_preservation_weights_high_influence_low_weight():
    heads = torch.tensor([0, 1])
    tails = torch.tensor([2, 3])
    influence = {0: 0.9, 1: 0.1, 2: 0.9, 3: 0.1}
    w = preservation_weights(heads, tails, influence, beta=2.0, device=torch.device("cpu"))
    # query 0 has high influence endpoint -> weaker preservation (smaller weight)
    assert w[0] < w[1]


def test_beta_zero_reduces_to_uniform_preservation_weights():
    """Section 7 invariant: beta=0 => w(q) == 1 for every stable triple,
    i.e. Chebyshev-weighted preservation collapses exactly to uniform
    preservation, regardless of the influence scores themselves."""
    heads = torch.tensor([0, 1, 2, 3])
    tails = torch.tensor([4, 5, 6, 7])
    # Deliberately varied / extreme influence values, including entities
    # absent from the mapping (default 0.0) and values outside [0, 1].
    influence = {0: 0.9, 1: 0.0, 2: 1.0, 4: 5.0, 5: -3.0, 6: 3.5}
    w = preservation_weights(heads, tails, influence, beta=0.0, device=torch.device("cpu"))
    uniform = torch.ones(len(heads))
    assert torch.allclose(w, uniform), f"expected all-ones weights at beta=0, got {w.tolist()}"


def test_beta_zero_chebyshev_loss_matches_uniform_loss():
    """End-to-end check of the same invariant through `preservation_loss`:
    with beta=0, the Chebyshev-weighted preservation loss on a fixed set of
    stable triples must equal the uniform-preservation loss exactly."""
    model = _make_model()
    model_old = _make_model()
    # Make old != new so the (f_new - f_old)^2 terms are non-trivial.
    with torch.no_grad():
        model.entity_embeddings.weight.add_(0.01)

    h = torch.tensor([0, 1, 2])
    r = torch.tensor([0, 1, 0])
    t = torch.tensor([3, 4, 2])

    influence_hat = {0: 0.8, 1: 0.2, 2: 0.5, 3: 0.9, 4: 0.1}
    w_chebyshev = preservation_weights(h, t, influence_hat, beta=0.0, device=torch.device("cpu"))
    w_uniform = torch.ones(len(h))

    loss_cheb = preservation_loss(model, model_old, h, r, t, w_chebyshev)
    loss_uniform = preservation_loss(model, model_old, h, r, t, w_uniform)
    assert torch.allclose(loss_cheb, loss_uniform), (
        f"beta=0 preservation loss ({loss_cheb.item()}) must equal the uniform "
        f"preservation loss ({loss_uniform.item()})"
    )


def test_preservation_loss_zero_when_models_identical():
    model = _make_model()
    model_old = _make_model()
    model_old.load_state_dict(model.state_dict())
    h = torch.tensor([0, 1])
    r = torch.tensor([0, 1])
    t = torch.tensor([2, 3])
    w = torch.ones(2)
    loss = preservation_loss(model, model_old, h, r, t, w)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_preservation_loss_no_grad_into_old_model():
    model = _make_model()
    model_old = _make_model()
    h = torch.tensor([0, 1])
    r = torch.tensor([0, 1])
    t = torch.tensor([2, 3])
    w = torch.ones(2)
    loss = preservation_loss(model, model_old, h, r, t, w)
    loss.backward()
    assert all(p.grad is None for p in model_old.parameters())
    assert any(p.grad is not None for p in model.parameters())


def test_sample_filtered_negatives_shape_and_relation_preserved():
    """Vectorized rewrite (Section 10 performance fix) must preserve the
    original contract: one block of `num_negatives` rows per input
    positive, in order, with the relation column untouched (only head OR
    tail is corrupted per candidate, never both, never the relation)."""
    rng = np.random.default_rng(0)
    pos = np.array([[0, 0, 1], [2, 1, 3], [4, 0, 5]], dtype=np.int64)
    known = {(0, 0, 1), (2, 1, 3), (4, 0, 5)}
    num_negatives = 5
    negs = sample_filtered_negatives(pos, num_entities=6, known_positive_triples=known,
                                      num_negatives=num_negatives, rng=rng)
    assert negs.shape == (len(pos) * num_negatives, 3)
    for i in range(len(pos)):
        block = negs[i * num_negatives:(i + 1) * num_negatives]
        assert (block[:, 1] == pos[i, 1]).all(), "relation column must never be corrupted"
        for row in block:
            h, r, t = int(row[0]), int(row[1]), int(row[2])
            assert h == pos[i, 0] or t == pos[i, 2], (
                "exactly one of head/tail must match the original positive "
                "(single-sided corruption)"
            )


def test_sample_filtered_negatives_avoids_known_positives_when_possible():
    """With max_tries >> 1 and a sparse known-positive set, essentially all
    sampled negatives should NOT be known positives (filtered rejection
    sampling must actually reject, not just corrupt blindly)."""
    rng = np.random.default_rng(0)
    n = 200
    pos = np.stack([rng.integers(0, 500, n), rng.integers(0, 20, n), rng.integers(0, 500, n)], axis=1).astype(np.int64)
    known = {tuple(x) for x in pos}
    negs = sample_filtered_negatives(pos, num_entities=500, known_positive_triples=known,
                                      num_negatives=10, rng=rng, max_tries=10)
    num_still_known = sum(tuple(row) in known for row in negs)
    frac_known = num_still_known / len(negs)
    assert frac_known < 0.01, f"too many sampled negatives are still known positives: {frac_known:.4f}"


def test_sample_filtered_negatives_empty_input():
    rng = np.random.default_rng(0)
    negs = sample_filtered_negatives(np.zeros((0, 3), dtype=np.int64), num_entities=10,
                                      known_positive_triples=set(), num_negatives=5, rng=rng)
    assert negs.shape == (0, 3)
