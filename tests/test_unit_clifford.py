"""Unit tests for Keci (Clifford algebra) model in dicee/models/clifford.py.

Covers:
- construct_cl_multivector: output shapes for various (p, q) combinations
- compute_sigma_pp: shape invariant for p >= 2
- compute_sigma_qq: shape invariant for q >= 2
- compute_sigma_pq: shape invariant for p >= 1 and q >= 1
- clifford_multiplication: all returned terms have correct shapes
- forward_k_vs_all / forward_triples: output shapes
- Invalid embedding_dim (non-integer r) raises AssertionError
"""

import pytest
import torch

from dicee.models.clifford import Keci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_keci(
    p: int = 0,
    q: int = 1,
    embedding_dim: int = 32,
    num_entities: int = 40,
    num_relations: int = 8,
) -> Keci:
    """Construct a Keci model with minimal args."""
    # embedding_dim must be divisible by (p + q + 1)
    args = dict(
        model="Keci",
        embedding_dim=embedding_dim,
        num_entities=num_entities,
        num_relations=num_relations,
        p=p,
        q=q,
        learning_rate=0.01,
        optim="Adam",
        scoring_technique="KvsAll",
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        normalization=None,
        init_param=None,
        byte_pair_encoding=False,
    )
    model = Keci(args)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# construct_cl_multivector shapes
# ---------------------------------------------------------------------------

class TestConstructCLMultivector:
    """Shape invariants for construct_cl_multivector across (p, q) settings."""

    @pytest.mark.parametrize("p,q,dim", [
        (0, 0, 16),   # r = 16, no blade parts
        (0, 1, 32),   # r = 16, one negative blade
        (1, 0, 32),   # r = 16, one positive blade
        (1, 1, 48),   # r = 16, one of each
        (2, 1, 64),   # r = 16, two positive + one negative
        (0, 2, 48),   # r = 16, two negative blades
        (2, 2, 80),   # r = 16, two of each
    ])
    def test_output_shapes(self, p, q, dim):
        """a0, ap, aq shapes must follow (B, r), (B, r, p), (B, r, q)."""
        model = _make_keci(p=p, q=q, embedding_dim=dim)
        r = dim // (p + q + 1)
        B = 8
        x = torch.randn(B, dim)

        a0, ap, aq = model.construct_cl_multivector(x, r=r, p=p, q=q)

        assert a0.shape == (B, r), f"a0 shape mismatch for p={p},q={q}: got {a0.shape}"
        assert ap.shape == (B, r, p), f"ap shape mismatch for p={p},q={q}: got {ap.shape}"
        assert aq.shape == (B, r, q), f"aq shape mismatch for p={p},q={q}: got {aq.shape}"

    def test_a0_equals_first_r_cols(self):
        """The scalar part must be exactly the first r columns of x."""
        model = _make_keci(p=0, q=1, embedding_dim=32)
        x = torch.randn(4, 32)
        a0, _, _ = model.construct_cl_multivector(x, r=16, p=0, q=1)
        assert torch.equal(a0, x[:, :16])


# ---------------------------------------------------------------------------
# compute_sigma_pp
# ---------------------------------------------------------------------------

class TestComputeSigmaPP:

    @pytest.mark.parametrize("p", [2, 3, 4])
    def test_shape_invariant(self, p):
        """sigma_pp shape must be (B, r, p*(p-1)//2)."""
        dim = 16 * (p + 1)  # r=16, (p+1) to satisfy divisibility with q=0
        model = _make_keci(p=p, q=0, embedding_dim=dim)
        B, r = 6, model.r
        # hp and rp: (B, r, p)
        hp = torch.randn(B, r, p)
        rp = torch.randn(B, r, p)
        sigma_pp = model.compute_sigma_pp(hp, rp)
        expected_pairs = p * (p - 1) // 2
        assert sigma_pp.shape == (B, r, expected_pairs), (
            f"sigma_pp shape for p={p}: expected (B={B}, r={r}, {expected_pairs}), got {sigma_pp.shape}"
        )

    def test_antisymmetry(self):
        """sigma_pp(hp, rp) == -sigma_pp(rp, hp)."""
        model = _make_keci(p=2, q=0, embedding_dim=48)
        B, r = 4, model.r
        hp = torch.randn(B, r, 2)
        rp = torch.randn(B, r, 2)
        forward = model.compute_sigma_pp(hp, rp)
        backward = model.compute_sigma_pp(rp, hp)
        assert torch.allclose(forward, -backward, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_sigma_qq
# ---------------------------------------------------------------------------

class TestComputeSigmaQQ:

    @pytest.mark.parametrize("q", [1, 2, 3])
    def test_shape_invariant(self, q):
        """sigma_qq shape must be (B, r, q*(q-1)//2) for q >= 1."""
        dim = 16 * (q + 1)
        model = _make_keci(p=0, q=q, embedding_dim=dim)
        B, r = 6, model.r
        hq = torch.randn(B, r, q)
        rq = torch.randn(B, r, q)
        sigma_qq = model.compute_sigma_qq(hq, rq)
        expected_pairs = q * (q - 1) // 2
        assert sigma_qq.shape == (B, r, expected_pairs), (
            f"sigma_qq shape for q={q}: expected (B={B}, r={r}, {expected_pairs}), got {sigma_qq.shape}"
        )

    def test_q1_returns_zeros(self):
        """For q=1 there are no pairs so sigma_qq must be all zeros."""
        model = _make_keci(p=0, q=1, embedding_dim=32)
        B, r = 4, model.r
        hq = torch.randn(B, r, 1)
        rq = torch.randn(B, r, 1)
        sigma_qq = model.compute_sigma_qq(hq, rq)
        assert sigma_qq.shape == (B, r, 0)
        assert sigma_qq.numel() == 0


# ---------------------------------------------------------------------------
# compute_sigma_pq
# ---------------------------------------------------------------------------

class TestComputeSigmaPQ:

    @pytest.mark.parametrize("p,q", [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 3),
    ])
    def test_shape_invariant(self, p, q):
        """sigma_pq shape must be (B, r, p, q)."""
        dim = 16 * (p + q + 1)
        model = _make_keci(p=p, q=q, embedding_dim=dim)
        B, r = 5, model.r
        hp = torch.randn(B, r, p)
        hq = torch.randn(B, r, q)
        rp = torch.randn(B, r, p)
        rq = torch.randn(B, r, q)
        sigma_pq = model.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq)
        assert sigma_pq.shape == (B, r, p, q), (
            f"sigma_pq shape for p={p},q={q}: expected ({B},{r},{p},{q}), got {sigma_pq.shape}"
        )

    def test_antisymmetry_swap_both_pairs(self):
        """Swapping (hp,hq) ↔ (rp,rq) negates sigma_pq."""
        p, q = 2, 1
        dim = 16 * (p + q + 1)
        model = _make_keci(p=p, q=q, embedding_dim=dim)
        B, r = 3, model.r
        hp = torch.randn(B, r, p)
        hq = torch.randn(B, r, q)
        rp = torch.randn(B, r, p)
        rq = torch.randn(B, r, q)
        forward = model.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq)
        # Swap both head and relation pairs: forward should negate
        swapped = model.compute_sigma_pq(hp=rp, hq=rq, rp=hp, rq=hq)
        assert torch.allclose(forward, -swapped, atol=1e-6)


# ---------------------------------------------------------------------------
# Keci forward passes
# ---------------------------------------------------------------------------

class TestKeciFeedForward:
    """Test forward_k_vs_all and forward_triples output shapes."""

    @pytest.mark.parametrize("p,q,dim", [
        (0, 0, 16),
        (0, 1, 32),
        (1, 0, 32),
        (1, 1, 48),
    ])
    def test_forward_k_vs_all_output_shape(self, p, q, dim):
        B, E, R = 8, 40, 8
        model = _make_keci(p=p, q=q, embedding_dim=dim, num_entities=E, num_relations=R)
        x = torch.stack([
            torch.randint(0, E, (B,)),
            torch.randint(0, R, (B,)),
        ], dim=1)
        out = model.forward_k_vs_all(x)
        assert out.shape == (B, E), f"forward_k_vs_all shape wrong for p={p},q={q}: {out.shape}"

    @pytest.mark.parametrize("p,q,dim", [
        (0, 0, 16),
        (0, 1, 32),
        (1, 1, 48),
    ])
    def test_forward_triples_output_shape(self, p, q, dim):
        B, E, R = 8, 40, 8
        model = _make_keci(p=p, q=q, embedding_dim=dim, num_entities=E, num_relations=R)
        x = torch.randint(0, E, (B, 3))
        x[:, 1] = torch.randint(0, R, (B,))
        out = model.forward_triples(x)
        assert out.shape == (B,), f"forward_triples shape wrong for p={p},q={q}: {out.shape}"


# ---------------------------------------------------------------------------
# Invalid embedding_dim raises AssertionError
# ---------------------------------------------------------------------------

class TestKeciBadEmbeddingDim:

    def test_embedding_dim_not_divisible_raises(self):
        """embedding_dim=33 is not divisible by (p=0 + q=1 + 1) = 2."""
        with pytest.raises(AssertionError, match="r = embedding_dim"):
            _make_keci(p=0, q=1, embedding_dim=33)

    def test_embedding_dim_divisible_succeeds(self):
        """embedding_dim=32 divisible by 2 → no exception."""
        model = _make_keci(p=0, q=1, embedding_dim=32)
        assert model.r == 16

    def test_r_is_stored_as_int(self):
        model = _make_keci(p=1, q=1, embedding_dim=48)
        assert isinstance(model.r, int)
        assert model.r == 16
