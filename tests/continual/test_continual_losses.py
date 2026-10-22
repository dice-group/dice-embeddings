import numpy as np
import torch

from dicee.continual.continual_losses import preservation_weights, preservation_loss
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
