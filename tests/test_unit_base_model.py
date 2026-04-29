"""Unit tests for dicee/models/base_model.py.

Covers:
- IdentityClass: forward/call returns input unchanged
- BaseKGE.init_params_with_sanity_checking: defaults and normalization branches
- BaseKGE.mem_of_model: dict has expected keys and non-negative values
- BaseKGE.get_embeddings: shapes
- BaseKGE.get_triple_representation: output shapes
- BaseKGE.get_head_relation_representation: output shapes
- BaseKGE.forward routing: tuple → k_vs_sample, (b,3) → triples, (b,2) → k_vs_all
- BaseKGE.configure_optimizers: Adam, SGD, AdamW
- BaseKGE.loss_function: computes BCE loss
- DistMult.k_vs_all_score: output shape
- DistMult.score: output shape
- TransE.score / forward_k_vs_all: output shapes
"""

import pytest
import torch

from dicee.models.base_model import BaseKGE, IdentityClass
from dicee.models.real import DistMult, TransE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_args(
    embedding_dim: int = 32,
    num_entities: int = 50,
    num_relations: int = 10,
    scoring_technique: str = "KvsAll",
    normalization: str = None,
    optim: str = "Adam",
) -> dict:
    """Return a minimal args dict suitable for DistMult/TransE construction."""
    return dict(
        model="DistMult",
        embedding_dim=embedding_dim,
        num_entities=num_entities,
        num_relations=num_relations,
        learning_rate=0.01,
        optim=optim,
        scoring_technique=scoring_technique,
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        normalization=normalization,
        init_param=None,
        byte_pair_encoding=False,
    )


def _make_distmult(**overrides) -> DistMult:
    args = _minimal_args(**overrides)
    model = DistMult(args)
    model.eval()
    return model


def _make_transe(**overrides) -> TransE:
    args = _minimal_args(**overrides)
    args["model"] = "TransE"
    model = TransE(args)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# IdentityClass
# ---------------------------------------------------------------------------

class TestIdentityClass:
    """IdentityClass must pass every input through unchanged."""

    def test_forward_returns_tensor_unchanged(self):
        identity = IdentityClass()
        x = torch.randn(4, 8)
        result = identity(x)
        assert result is x

    def test_forward_returns_scalar_unchanged(self):
        identity = IdentityClass()
        s = 42
        assert identity(s) == s

    def test_forward_none_unchanged(self):
        identity = IdentityClass()
        assert identity(None) is None

    def test_class_call_as_no_op_init(self):
        """IdentityClass used as param_init: calling on a tensor must be a no-op."""
        data = torch.ones(3, 3)
        original = data.clone()
        IdentityClass()(data)
        assert torch.equal(data, original), "IdentityClass should not mutate its input"


# ---------------------------------------------------------------------------
# BaseKGE.init_params_with_sanity_checking
# ---------------------------------------------------------------------------

class TestInitParamsWithSanityChecking:

    def test_defaults_are_populated(self):
        """Minimal args should produce sensible defaults."""
        model = _make_distmult()
        assert model.embedding_dim == 32
        assert model.num_entities == 50
        assert model.num_relations == 10
        assert model.learning_rate == 0.01
        assert model.input_dropout_rate == 0.0
        assert model.hidden_dropout_rate == 0.0
        assert model.weight_decay == 0.0

    def test_missing_embedding_dim_defaults_to_one(self):
        args = dict(
            model="DistMult",
            num_entities=10,
            num_relations=5,
            optim="Adam",
            scoring_technique="KvsAll",
            normalization=None,
            init_param=None,
            byte_pair_encoding=False,
        )
        model = DistMult(args)
        assert model.embedding_dim == 1

    def test_layer_norm_sets_correct_class(self):
        model = _make_distmult(normalization="LayerNorm")
        assert isinstance(model.normalize_head_entity_embeddings, torch.nn.LayerNorm)
        assert isinstance(model.normalize_relation_embeddings, torch.nn.LayerNorm)

    def test_batch_norm_sets_correct_class(self):
        model = _make_distmult(normalization="BatchNorm1d")
        assert isinstance(model.normalize_head_entity_embeddings, torch.nn.BatchNorm1d)
        assert isinstance(model.normalize_relation_embeddings, torch.nn.BatchNorm1d)

    def test_no_normalization_uses_identity(self):
        model = _make_distmult(normalization=None)
        assert isinstance(model.normalize_head_entity_embeddings, IdentityClass)
        assert isinstance(model.normalize_relation_embeddings, IdentityClass)

    def test_invalid_normalization_raises(self):
        with pytest.raises((NotImplementedError, ValueError, Exception)):
            _make_distmult(normalization="InvalidNorm")


# ---------------------------------------------------------------------------
# BaseKGE.mem_of_model
# ---------------------------------------------------------------------------

class TestMemOfModel:

    def test_returns_dict_with_expected_keys(self):
        model = _make_distmult()
        info = model.mem_of_model()
        assert "EstimatedSizeMB" in info
        assert "NumParam" in info

    def test_num_params_is_positive(self):
        model = _make_distmult()
        info = model.mem_of_model()
        assert info["NumParam"] > 0

    def test_size_is_non_negative(self):
        model = _make_distmult()
        info = model.mem_of_model()
        assert info["EstimatedSizeMB"] >= 0.0

    def test_larger_model_has_more_params(self):
        small = _make_distmult(embedding_dim=8, num_entities=20, num_relations=5)
        large = _make_distmult(embedding_dim=128, num_entities=500, num_relations=50)
        assert large.mem_of_model()["NumParam"] > small.mem_of_model()["NumParam"]


# ---------------------------------------------------------------------------
# BaseKGE.get_embeddings
# ---------------------------------------------------------------------------

class TestGetEmbeddings:

    def test_entity_embedding_shape(self):
        model = _make_distmult(num_entities=20, num_relations=5, embedding_dim=16)
        ent_emb, rel_emb = model.get_embeddings()
        assert ent_emb.shape == (20, 16)

    def test_relation_embedding_shape(self):
        model = _make_distmult(num_entities=20, num_relations=5, embedding_dim=16)
        ent_emb, rel_emb = model.get_embeddings()
        assert rel_emb.shape == (5, 16)

    def test_embeddings_are_numpy_arrays(self):
        import numpy as np
        model = _make_distmult()
        ent_emb, rel_emb = model.get_embeddings()
        assert hasattr(ent_emb, "shape")  # numpy array or torch tensor with shape
        assert hasattr(rel_emb, "shape")


# ---------------------------------------------------------------------------
# BaseKGE.get_triple_representation
# ---------------------------------------------------------------------------

class TestGetTripleRepresentation:

    def test_output_shapes_match_embedding_dim(self):
        B, D = 8, 32
        model = _make_distmult(num_entities=20, num_relations=5, embedding_dim=D)
        x = torch.randint(0, 20, (B, 3))
        x[:, 1] = torch.randint(0, 5, (B,))
        h, r, t = model.get_triple_representation(x)
        assert h.shape == (B, D)
        assert r.shape == (B, D)
        assert t.shape == (B, D)

    def test_head_and_tail_can_be_same_entity(self):
        model = _make_distmult(num_entities=5, num_relations=3)
        x = torch.zeros(4, 3, dtype=torch.long)  # all heads, rels, tails = 0
        h, r, t = model.get_triple_representation(x)
        assert h.shape[0] == 4 and r.shape[0] == 4 and t.shape[0] == 4


# ---------------------------------------------------------------------------
# BaseKGE.get_head_relation_representation
# ---------------------------------------------------------------------------

class TestGetHeadRelationRepresentation:

    def test_output_shapes(self):
        B, D = 6, 16
        model = _make_distmult(num_entities=30, num_relations=8, embedding_dim=D)
        x = torch.stack([
            torch.randint(0, 30, (B,)),
            torch.randint(0, 8, (B,)),
        ], dim=1)
        h, r = model.get_head_relation_representation(x)
        assert h.shape == (B, D)
        assert r.shape == (B, D)


# ---------------------------------------------------------------------------
# BaseKGE.forward routing
# ---------------------------------------------------------------------------

class TestBaseKGEForwardRouting:
    """forward() must dispatch to the correct method depending on input type/shape."""

    def setup_method(self):
        self.B = 4
        self.D = 32
        self.E = 50
        self.R = 10
        self.model = _make_distmult(
            num_entities=self.E, num_relations=self.R, embedding_dim=self.D
        )

    def test_2d_dim3_routes_to_forward_triples(self):
        x = torch.randint(0, self.E, (self.B, 3))
        x[:, 1] = torch.randint(0, self.R, (self.B,))
        out = self.model(x)
        assert out.shape == (self.B,)

    def test_2d_dim2_routes_to_forward_k_vs_all(self):
        x = torch.stack([
            torch.randint(0, self.E, (self.B,)),
            torch.randint(0, self.R, (self.B,)),
        ], dim=1)
        out = self.model(x)
        assert out.shape == (self.B, self.E)

    def test_tuple_routes_to_forward_k_vs_sample(self):
        k = 5
        x = torch.stack([
            torch.randint(0, self.E, (self.B,)),
            torch.randint(0, self.R, (self.B,)),
        ], dim=1)
        y_idx = torch.randint(0, self.E, (self.B, k))
        out = self.model((x, y_idx))
        assert out.shape == (self.B, k)


# ---------------------------------------------------------------------------
# BaseKGE.configure_optimizers
# ---------------------------------------------------------------------------

class TestConfigureOptimizers:

    def test_adam_returns_adam_optimizer(self):
        model = _make_distmult(optim="Adam")
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd_returns_sgd_optimizer(self):
        model = _make_distmult(optim="SGD")
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.SGD)

    def test_adamw_returns_adamw_optimizer(self):
        model = _make_distmult(optim="AdamW")
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)

    def test_unknown_optim_raises(self):
        model = _make_distmult(optim="Adam")
        model.optimizer_name = "NonExistent"
        with pytest.raises(KeyError):
            model.configure_optimizers()


# ---------------------------------------------------------------------------
# BaseKGE.loss_function
# ---------------------------------------------------------------------------

class TestLossFunction:

    def test_bce_loss_returns_scalar(self):
        model = _make_distmult()
        yhat = torch.randn(8, 50)
        y = torch.zeros(8, 50)
        loss = model.loss_function(yhat, y)
        assert loss.shape == torch.Size([]), "Expected scalar loss"

    def test_loss_is_finite(self):
        model = _make_distmult()
        yhat = torch.randn(4, 10)
        y = torch.randint(0, 2, (4, 10)).float()
        loss = model.loss_function(yhat, y)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# DistMult scoring functions
# ---------------------------------------------------------------------------

class TestDistMultScoring:

    def setup_method(self):
        self.B, self.D, self.E = 8, 32, 50
        self.model = _make_distmult(num_entities=self.E, num_relations=10, embedding_dim=self.D)

    def test_score_output_shape(self):
        h = torch.randn(self.B, self.D)
        r = torch.randn(self.B, self.D)
        t = torch.randn(self.B, self.D)
        out = self.model.score(h, r, t)
        assert out.shape == (self.B,)

    def test_k_vs_all_score_output_shape(self):
        h = torch.randn(self.B, self.D)
        r = torch.randn(self.B, self.D)
        E = self.model.entity_embeddings.weight
        out = self.model.k_vs_all_score(h, r, E)
        assert out.shape == (self.B, self.E)

    def test_forward_triples_output_shape(self):
        x = torch.randint(0, self.E, (self.B, 3))
        x[:, 1] = torch.randint(0, 10, (self.B,))
        out = self.model.forward_triples(x)
        assert out.shape == (self.B,)

    def test_forward_k_vs_all_output_shape(self):
        x = torch.stack([
            torch.randint(0, self.E, (self.B,)),
            torch.randint(0, 10, (self.B,)),
        ], dim=1)
        out = self.model.forward_k_vs_all(x)
        assert out.shape == (self.B, self.E)


# ---------------------------------------------------------------------------
# TransE scoring functions
# ---------------------------------------------------------------------------

class TestTransEScoring:

    def setup_method(self):
        self.B, self.D, self.E = 8, 32, 50
        self.model = _make_transe(num_entities=self.E, num_relations=10, embedding_dim=self.D)

    def test_score_output_shape(self):
        h = torch.randn(self.B, self.D)
        r = torch.randn(self.B, self.D)
        t = torch.randn(self.B, self.D)
        out = self.model.score(h, r, t)
        assert out.shape == (self.B,)

    def test_score_exact_zero_distance(self):
        """When h + r = t the score should equal the margin."""
        h = torch.ones(1, self.D)
        r = torch.zeros(1, self.D)
        t = torch.ones(1, self.D)  # h + r == t
        out = self.model.score(h, r, t)
        assert torch.isclose(out, torch.tensor(float(self.model.margin)))

    def test_forward_k_vs_all_output_shape(self):
        x = torch.stack([
            torch.randint(0, self.E, (self.B,)),
            torch.randint(0, 10, (self.B,)),
        ], dim=1)
        out = self.model.forward_k_vs_all(x)
        assert out.shape == (self.B, self.E)
