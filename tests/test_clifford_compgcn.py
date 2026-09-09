"""Unit tests for the CliffordCompGCN prototype.

CliffordCompGCN / CompGCN are registered directly in
``dicee.static_funcs.MODEL_REGISTRY`` and are constructed exactly like any
other ``dicee`` model, i.e. from a flat ``args`` dict via
``dicee.static_funcs.intialize_model`` -- no bespoke construction path.

Covers:
    1. Clifford representation shape.
    2. Geometric product compatibility (matches the shared FullDeCaL routine).
    3. Entity x relation -> Clifford message.
    4. Message aggregation.
    5. One forward pass (NegSample-style and KvsAll-style), via BaseKGE.forward.
    6. Gradient propagation.
    7. A tiny synthetic KG trained for a few iterations, using the standard
       model API (BaseKGE.forward), no custom training loop.
"""
import numpy as np
import torch
import pytest

from dicee.models.clifford import _build_sign_table, clifford_geometric_product
from dicee.models.clifford_compgcn import (
    CliffordCompGCNLayer,
    CliffordLinear,
    FullCliffordGNNLayer,
    build_compgcn_graph,
)
from dicee.static_funcs import intialize_model


def _tiny_kg_args(model_name="CliffordCompGCN", **extra):
    """5 entities, 2 relations, 6 triples -- small enough to overfit quickly."""
    num_entities, num_relations = 5, 2
    train_set = np.array([
        [0, 0, 1],
        [1, 1, 2],
        [2, 0, 3],
        [3, 1, 4],
        [4, 0, 0],
        [0, 1, 2],
    ], dtype=np.int64)
    args = dict(
        model=model_name,
        embedding_dim=16,
        num_entities=num_entities,
        num_relations=num_relations,
        train_set=train_set,
        learning_rate=0.05,
        optim="Adam",
        scoring_technique="NegSample",
        p=1, q=1, r=0,
        num_gcn_layers=1,
        composition="corr",
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        normalization=None,
        init_param=None,
        weight_decay=0.0,
        byte_pair_encoding=False,
    )
    args.update(extra)
    return args, train_set


# --------------------------------------------------------------------------- #
# 1. Clifford representation shape
# --------------------------------------------------------------------------- #
def test_clifford_representation_shape():
    args, _ = _tiny_kg_args(embedding_dim=16, p=1, q=1, r=0)  # d = 4, re = 4
    model, labelling = intialize_model(args)

    assert labelling == "EntityPrediction"
    assert model.core.d == 4
    assert model.core.re == 4
    assert model.core.entity_embeddings.weight.shape == (args["num_entities"], 16)
    assert model.core.relation_embeddings.weight.shape == (2 * args["num_relations"] + 1, 16)

    bad_args, _ = _tiny_kg_args(embedding_dim=15, p=1, q=1, r=0)
    with pytest.raises(AssertionError):
        intialize_model(bad_args)


# --------------------------------------------------------------------------- #
# 2. Geometric product compatibility with the shared FullDeCaL routine
# --------------------------------------------------------------------------- #
def test_geometric_product_matches_shared_routine():
    n, re, batch = 2, 3, 4
    d = 1 << n
    sign_table, K_table, intersection_table, bits = _build_sign_table(n)
    eta = torch.tensor([1.0, -1.0])  # p=1, q=1
    eta_blade = (bits * eta + (1.0 - bits)).prod(dim=1)
    coeff_table = sign_table * eta_blade[intersection_table]

    h = torch.randn(batch, d, re)
    r = torch.randn(batch, d, re)
    z1 = clifford_geometric_product(h, r, coeff_table, K_table, d, re)

    layer = CliffordCompGCNLayer(dim=d * re, d=d, re=re, coeff_table=coeff_table, K_table=K_table)
    z2 = layer.phi(h.reshape(batch, d * re), r.reshape(batch, d * re)).view(batch, d, re)

    assert torch.allclose(z1, z2, atol=1e-6)
    assert not torch.isnan(z1).any()


# --------------------------------------------------------------------------- #
# 3. Entity x relation -> Clifford message
# --------------------------------------------------------------------------- #
def test_entity_relation_to_message():
    args, train_set = _tiny_kg_args(embedding_dim=8, p=1, q=0, r=0)  # d=2, re=4
    model, _ = intialize_model(args)

    triples = torch.as_tensor(train_set, dtype=torch.long)
    edge_index, edge_type = build_compgcn_graph(triples, args["num_entities"], args["num_relations"])

    layer = model.core.layers[0]
    h_src = model.core.entity_embeddings.weight.index_select(0, edge_index[0])
    r_edge = model.core.relation_embeddings.weight.index_select(0, edge_type)
    msg = layer.phi(h_src, r_edge)

    assert msg.shape == (edge_index.shape[1], args["embedding_dim"])
    assert not torch.isnan(msg).any()
    assert not torch.allclose(msg, h_src)
    assert not torch.allclose(msg, r_edge)


# --------------------------------------------------------------------------- #
# 4. Message aggregation
# --------------------------------------------------------------------------- #
def test_message_aggregation_changes_representation():
    args, _ = _tiny_kg_args(embedding_dim=8, p=1, q=0, r=0)
    model, _ = intialize_model(args)

    ent_before = model.core.entity_embeddings.weight.clone()
    ent_after, rel_after = model.core.encode(model._edge_index, model._edge_type)

    assert ent_after.shape == ent_before.shape
    assert not torch.isnan(ent_after).any()
    assert not torch.allclose(ent_after, ent_before)


# --------------------------------------------------------------------------- #
# 5. One forward pass (both scoring modes), for Clifford and vanilla CompGCN,
#    routed through BaseKGE.forward exactly like any other dicee model.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", ["CliffordCompGCN", "CompGCN"])
def test_forward_pass_through_base_kge(model_name):
    args, train_set = _tiny_kg_args(model_name=model_name, embedding_dim=16, num_gcn_layers=2)
    model, _ = intialize_model(args)

    triples = torch.as_tensor(train_set, dtype=torch.long)
    triple_scores = model(triples)
    assert triple_scores.shape == (triples.shape[0],)
    assert not torch.isnan(triple_scores).any()

    kva_scores = model.forward_k_vs_all(triples[:, :2])
    assert kva_scores.shape == (triples.shape[0], args["num_entities"])
    assert not torch.isnan(kva_scores).any()


# --------------------------------------------------------------------------- #
# 6. Gradient propagation
# --------------------------------------------------------------------------- #
def test_gradient_propagation():
    args, train_set = _tiny_kg_args(embedding_dim=16, num_gcn_layers=2)
    model, _ = intialize_model(args)
    triples = torch.as_tensor(train_set, dtype=torch.long)

    scores = model(triples)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
    loss.backward()

    assert model.core.entity_embeddings.weight.grad is not None
    assert model.core.relation_embeddings.weight.grad is not None
    assert model.core.entity_embeddings.weight.grad.abs().sum().item() > 0
    assert model.core.relation_embeddings.weight.grad.abs().sum().item() > 0
    for layer in model.core.layers:
        assert layer.W_in.weight.grad is not None


# --------------------------------------------------------------------------- #
# 7. Tiny synthetic KG: overfit for a few iterations (standard model API,
#    no custom training loop needed).
# --------------------------------------------------------------------------- #
def test_overfit_tiny_kg():
    torch.manual_seed(0)
    args, train_set = _tiny_kg_args(embedding_dim=16, num_gcn_layers=1, learning_rate=0.05)
    model, _ = intialize_model(args)
    triples = torch.as_tensor(train_set, dtype=torch.long)
    labels = torch.ones(triples.shape[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        scores = model(triples)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert not any(torch.isnan(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.3


# --------------------------------------------------------------------------- #
# FullCliffordGNN: CliffordCompGCN + grade-aware CliffordLinear transforms.    #
# --------------------------------------------------------------------------- #
def test_clifford_linear_grade_aware_shape_and_grad():
    d, re = 4, 3  # n=2 -> grades {0, 1, 2}
    lin = CliffordLinear(d, re)
    # grades present: scalar(0), two vectors(1), bivector(2)
    assert lin.grade_of_blade.tolist() == [0, 1, 1, 2]
    assert len(lin.grade_linears) == 3

    x = torch.randn(5, d * re, requires_grad=True)
    out = lin(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    for g in lin.grade_linears:
        assert g.weight.grad is not None
        assert g.weight.grad.abs().sum().item() > 0


def test_full_clifford_gnn_layer_uses_clifford_linear():
    n, re = 2, 3
    d = 1 << n
    sign_table, K_table, intersection_table, bits = _build_sign_table(n)
    eta = torch.tensor([1.0, -1.0])
    eta_blade = (bits * eta + (1.0 - bits)).prod(dim=1)
    coeff_table = sign_table * eta_blade[intersection_table]

    layer = FullCliffordGNNLayer(dim=d * re, d=d, re=re, coeff_table=coeff_table, K_table=K_table)
    assert isinstance(layer.W_0, CliffordLinear)
    assert isinstance(layer.W_in, CliffordLinear)
    assert isinstance(layer.W_out, CliffordLinear)
    assert isinstance(layer.W_loop, CliffordLinear)
    assert isinstance(layer.W_rel, CliffordLinear)

    # phi (geometric product) must be unchanged vs. CliffordCompGCNLayer
    h = torch.randn(4, d, re)
    r = torch.randn(4, d, re)
    z_full = layer.phi(h.reshape(4, d * re), r.reshape(4, d * re))
    z_shared = clifford_geometric_product(h, r, coeff_table, K_table, d, re).reshape(4, d * re)
    assert torch.allclose(z_full, z_shared, atol=1e-6)


def test_full_clifford_gnn_instantiation_forward_shapes_and_gradients():
    args, train_set = _tiny_kg_args(model_name="FullCliffordGNN", embedding_dim=16, num_gcn_layers=2)
    model, labelling = intialize_model(args)

    assert labelling == "EntityPrediction"
    assert model.core.d == 4 and model.core.re == 4
    for layer in model.core.layers:
        assert isinstance(layer, FullCliffordGNNLayer)

    triples = torch.as_tensor(train_set, dtype=torch.long)

    triple_scores = model(triples)
    assert triple_scores.shape == (triples.shape[0],)
    assert not torch.isnan(triple_scores).any()

    kva_scores = model.forward_k_vs_all(triples[:, :2])
    assert kva_scores.shape == (triples.shape[0], args["num_entities"])
    assert not torch.isnan(kva_scores).any()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(triple_scores, torch.ones_like(triple_scores))
    loss.backward()

    assert model.core.entity_embeddings.weight.grad is not None
    assert model.core.entity_embeddings.weight.grad.abs().sum().item() > 0
    assert model.core.relation_embeddings.weight.grad is not None
    assert model.core.relation_embeddings.weight.grad.abs().sum().item() > 0
    for layer in model.core.layers:
        for name in ("W_0", "W_in", "W_out", "W_loop", "W_rel"):
            clifford_linear = getattr(layer, name)
            assert isinstance(clifford_linear, CliffordLinear)
            for grade_lin in clifford_linear.grade_linears:
                assert grade_lin.weight.grad is not None


def test_full_clifford_gnn_overfit_tiny_kg():
    torch.manual_seed(0)
    args, train_set = _tiny_kg_args(model_name="FullCliffordGNN", embedding_dim=16, num_gcn_layers=1)
    model, _ = intialize_model(args)
    triples = torch.as_tensor(train_set, dtype=torch.long)
    labels = torch.ones(triples.shape[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        scores = model(triples)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert not any(torch.isnan(torch.tensor(losses)))
    assert losses[-1] < losses[0]
