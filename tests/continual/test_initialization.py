import copy

import torch

from dicee.continual.initialization import extend_entity_embeddings
from dicee.models.real import DistMult


def _make_model(num_entities=5, num_relations=2, dim=8):
    args = {
        "model": "DistMult",
        "embedding_dim": dim,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "learning_rate": 0.01,
        "lr": 0.01,
        "optim": "Adam",
        "scoring_technique": "KvsAll",
        "input_dropout_rate": 0.0,
        "hidden_dropout_rate": 0.0,
        "weight_decay": 0.0,
        "normalization": None,
        "byte_pair_encoding": False,
    }
    return DistMult(args)


def test_extend_preserves_existing_rows():
    model = _make_model(num_entities=5)
    old_weights = model.entity_embeddings.weight.data.clone()
    extend_entity_embeddings(model, new_num_entities=8)
    assert model.entity_embeddings.weight.shape[0] == 8
    assert torch.allclose(model.entity_embeddings.weight.data[:5], old_weights)
    assert model.num_entities == 8


def test_extend_noop_when_same_size():
    model = _make_model(num_entities=5)
    emb = model.entity_embeddings
    returned = extend_entity_embeddings(model, new_num_entities=5)
    assert returned is emb


def test_extend_rejects_shrinking():
    model = _make_model(num_entities=5)
    try:
        extend_entity_embeddings(model, new_num_entities=3)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_neighbor_informed_init():
    model = _make_model(num_entities=4)
    neighbors = {4: [0, 1], 5: [2]}
    extend_entity_embeddings(model, new_num_entities=6, neighbor_informed_init=True,
                              new_entity_neighbors=neighbors)
    expected_4 = model.entity_embeddings.weight.data[[0, 1]]
    # row 4 should equal mean of rows 0,1 as they existed BEFORE extension;
    # just check it's close to the current mean (rows 0,1 are untouched by extension).
    assert torch.allclose(model.entity_embeddings.weight.data[4], expected_4.mean(dim=0))


def test_optimizer_must_be_rebuilt_after_extension():
    model = _make_model(num_entities=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # take a step so Adam has state for the old entity_embeddings parameter
    x = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    loss = model.forward_triples(x).sum()
    loss.backward()
    optimizer.step()
    old_param_id = id(model.entity_embeddings.weight)
    assert old_param_id in {id(p) for p in optimizer.param_groups[0]["params"]}

    extend_entity_embeddings(model, new_num_entities=7)
    new_param_id = id(model.entity_embeddings.weight)
    assert new_param_id != old_param_id
    # The stale optimizer no longer references the (new) live parameter.
    assert new_param_id not in {id(p) for p in optimizer.param_groups[0]["params"]}

    # A freshly rebuilt optimizer does reference it, and training updates new rows too.
    new_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    before = model.entity_embeddings.weight.data[6].clone()
    x2 = torch.tensor([[6, 0, 0]], dtype=torch.long)
    new_optimizer.zero_grad()
    loss2 = model.forward_triples(x2).sum()
    loss2.backward()
    new_optimizer.step()
    after = model.entity_embeddings.weight.data[6].clone()
    assert not torch.allclose(before, after)
