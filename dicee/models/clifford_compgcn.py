"""CliffordCompGCN — a minimal, deliberately simple prototype that replaces
the composition operator of CompGCN (Vashishth et al., 2020) with the
Clifford geometric product used by DeCaL / FullDeCaL.

Standard CompGCN aggregates relational messages as

    h_v^{(l+1)} = sigma( W_0 h_v^{(l)} + sum_{(u,r,v) in E} W_r * phi(h_u^{(l)}, h_r^{(l)}) )

where ``phi`` is a *vector* composition operator (subtraction, multiplication,
or circular-correlation).  This module keeps the CompGCN message-passing
skeleton (self-loop + relation-typed neighbour aggregation + relation
update) unchanged, and only swaps ``phi`` for the Clifford geometric product

    phi(h_u, h_r) = h_u (x) h_r        (geometric product in Cl_{p,q,r})

Reuse policy
------------
* The geometric product itself is **not** re-implemented here.  It is
  imported directly from :mod:`dicee.models.clifford`
  (:func:`clifford_geometric_product`, :func:`_build_sign_table`), which is
  the exact routine used by ``FullDeCaL``.
* Entities/relations follow the *same* ``embedding_dim = 2^n * re`` blade
  convention as ``FullDeCaL`` (fixed-signature mode), where ``n = p + q + r``.
* The decoder (triple scoring function) reuses the same geometric-product +
  inner-product formula as ``FullDeCaL.forward_triples`` /
  ``forward_k_vs_all``.  ``CliffordCompGCN`` differs from ``FullDeCaL`` only
  in *how the entity/relation representations that enter the decoder are
  produced* (via GNN message passing instead of being looked up directly
  from an embedding table).

This module intentionally does *not* implement attention, gating, or
residual variants — only a single ``sum`` aggregation, matching the
original CompGCN paper and the scope requested for this first prototype.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseKGE
from .clifford import _build_sign_table, clifford_geometric_product

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Graph construction (CompGCN convention: forward + inverse + self-loop)     #
# --------------------------------------------------------------------------- #
def build_compgcn_graph(triples: torch.LongTensor, num_entities: int, num_relations: int,
                         device: Optional[torch.device] = None) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Build a CompGCN-style multi-relational message-passing graph.

    For every training triple ``(h, r, t)`` two directed edges are added:
    ``h -> t`` labelled ``r`` (forward) and ``t -> h`` labelled
    ``r + num_relations`` (inverse).  A self-loop edge with a dedicated
    relation id ``2 * num_relations`` is added for every entity.

    Parameters
    ----------
    triples : torch.LongTensor
        Shape ``(N, 3)``, columns ``[head, relation, tail]``, 0-indexed with
        ``relation < num_relations``.
    num_entities, num_relations : int
        Sizes of the *original* (non-augmented) entity/relation vocabularies.

    Returns
    -------
    edge_index : torch.LongTensor
        Shape ``(2, E')`` with rows ``[src, dst]``.
    edge_type : torch.LongTensor
        Shape ``(E',)``, values in ``[0, 2 * num_relations]`` inclusive.
    """
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    src = torch.cat([h, t], dim=0)
    dst = torch.cat([t, h], dim=0)
    rel = torch.cat([r, r + num_relations], dim=0)

    self_loop_ent = torch.arange(num_entities, dtype=torch.long)
    src = torch.cat([src, self_loop_ent])
    dst = torch.cat([dst, self_loop_ent])
    rel = torch.cat([rel, torch.full((num_entities,), 2 * num_relations, dtype=torch.long)])

    edge_index = torch.stack([src, dst], dim=0)
    edge_type = rel
    if device is not None:
        edge_index, edge_type = edge_index.to(device), edge_type.to(device)
    return edge_index, edge_type


# --------------------------------------------------------------------------- #
#  Composition operators (phi)                                                #
# --------------------------------------------------------------------------- #
def compose_sub(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """CompGCN's ``sub`` composition: ``phi(h, r) = h - r``."""
    return h - r


def compose_mult(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """CompGCN's ``mult`` composition: ``phi(h, r) = h * r`` (Hadamard)."""
    return h * r


def compose_corr(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """CompGCN's ``corr`` composition: circular correlation via FFT."""
    H = torch.fft.rfft(h, dim=-1)
    R = torch.fft.rfft(r, dim=-1)
    return torch.fft.irfft(torch.conj(H) * R, n=h.shape[-1], dim=-1)


_VECTOR_COMPOSITIONS = {"sub": compose_sub, "mult": compose_mult, "corr": compose_corr}


# --------------------------------------------------------------------------- #
#  Single message-passing layer                                               #
# --------------------------------------------------------------------------- #
class _BaseCompGCNLayer(nn.Module):
    """Shared CompGCN skeleton: self-loop + directional relation transforms.

    ``phi`` (the composition function) is supplied by subclasses
    (:class:`CliffordCompGCNLayer` uses the Clifford geometric product,
    the vector baseline uses sub/mult/corr).
    """

    def __init__(self, dim: int, dropout: float = 0.0, activation=torch.tanh):
        super().__init__()
        self.dim = dim
        self.W_0 = nn.Linear(dim, dim, bias=False)
        self.W_in = nn.Linear(dim, dim, bias=False)    # forward-direction relations
        self.W_out = nn.Linear(dim, dim, bias=False)   # inverse-direction relations
        self.W_loop = nn.Linear(dim, dim, bias=False)  # self-loop relation
        self.W_rel = nn.Linear(dim, dim, bias=False)   # relation embedding update
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        for lin in (self.W_0, self.W_in, self.W_out, self.W_loop, self.W_rel):
            nn.init.xavier_normal_(lin.weight)

    def phi(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, ent_emb: torch.Tensor, rel_emb: torch.Tensor,
                edge_index: torch.LongTensor, edge_type: torch.LongTensor,
                num_relations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """One relational message-passing step.

        Parameters
        ----------
        ent_emb : torch.Tensor
            Shape ``(N, dim)`` current entity representations.
        rel_emb : torch.Tensor
            Shape ``(2 * num_relations + 1, dim)`` current relation
            representations (forward, inverse, self-loop).
        edge_index : torch.LongTensor
            Shape ``(2, E)`` ``[src, dst]``.
        edge_type : torch.LongTensor
            Shape ``(E,)``, values in ``[0, 2 * num_relations]``.
        num_relations : int
            Number of *original* relations (before fwd/inv/self-loop
            augmentation).

        Returns
        -------
        new_ent_emb : torch.Tensor
            Shape ``(N, dim)``.
        new_rel_emb : torch.Tensor
            Shape ``(2 * num_relations + 1, dim)``.
        """
        N = ent_emb.size(0)
        src, dst = edge_index[0], edge_index[1]

        h_src = ent_emb.index_select(0, src)          # (E, dim)
        r_edge = rel_emb.index_select(0, edge_type)    # (E, dim)

        # (1) Composition: phi(h_u, h_r) -- the ONLY place where CliffordCompGCN
        #     differs from vanilla CompGCN.
        msg = self.phi(h_src, r_edge)                  # (E, dim)

        # (2) Direction-specific relation-transform W_r (shared within each
        #     direction, as in the original CompGCN "basis-free" variant).
        fwd_mask = edge_type < num_relations
        inv_mask = (edge_type >= num_relations) & (edge_type < 2 * num_relations)
        loop_mask = edge_type >= 2 * num_relations

        transformed = torch.zeros_like(msg)
        if fwd_mask.any():
            transformed[fwd_mask] = self.W_in(msg[fwd_mask])
        if inv_mask.any():
            transformed[inv_mask] = self.W_out(msg[inv_mask])
        if loop_mask.any():
            transformed[loop_mask] = self.W_loop(msg[loop_mask])

        # (3) Sum aggregation into destination nodes (mean-normalised by
        #     in-degree for numerical stability -- still a "sum" aggregator
        #     up to a fixed per-node scaling, no attention/gating).
        agg = torch.zeros(N, self.dim, device=ent_emb.device, dtype=ent_emb.dtype)
        agg.index_add_(0, dst, transformed)
        deg = torch.zeros(N, device=ent_emb.device, dtype=ent_emb.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=ent_emb.dtype))
        agg = agg / deg.clamp(min=1).unsqueeze(-1)

        new_ent_emb = self.activation(self.dropout(self.W_0(ent_emb) + agg))
        new_rel_emb = self.W_rel(rel_emb)
        return new_ent_emb, new_rel_emb


class CompGCNLayer(_BaseCompGCNLayer):
    """Vanilla CompGCN layer using a vector composition operator (baseline)."""

    def __init__(self, dim: int, composition: str = "corr", dropout: float = 0.0):
        super().__init__(dim, dropout=dropout)
        if composition not in _VECTOR_COMPOSITIONS:
            raise ValueError(f"Unknown composition '{composition}', choices: {list(_VECTOR_COMPOSITIONS)}")
        self._phi = _VECTOR_COMPOSITIONS[composition]

    def phi(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self._phi(h, r)


class CliffordCompGCNLayer(_BaseCompGCNLayer):
    """CompGCN layer where ``phi`` is the Clifford geometric product.

    ``phi(h_u, h_r) = h_u (x) h_r`` in ``Cl_{p,q,r}``, using the exact same
    ``clifford_geometric_product`` routine as ``FullDeCaL``.
    """

    def __init__(self, dim: int, d: int, re: int, coeff_table: torch.Tensor, K_table: torch.Tensor,
                 dropout: float = 0.0):
        super().__init__(dim, dropout=dropout)
        assert dim == d * re
        self.d = d
        self.re = re
        self.register_buffer("coeff_table", coeff_table)
        self.register_buffer("K_table", K_table)

    def phi(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        h_blades = h.view(B, self.d, self.re)
        r_blades = r.view(B, self.d, self.re)
        z = clifford_geometric_product(h_blades, r_blades, self.coeff_table, self.K_table, self.d, self.re)
        return z.reshape(B, self.d * self.re)


# --------------------------------------------------------------------------- #
#  Clifford-native (grade-aware) linear transformation                        #
# --------------------------------------------------------------------------- #
class CliffordLinear(nn.Module):
    r"""Grade-aware linear transformation for flattened Clifford multivectors.

    Drop-in replacement for ``nn.Linear(D, D)`` (same ``(..., D) -> (..., D)``
    signature, ``D = d * re``) that respects the Clifford multivector
    structure instead of treating the flattened vector as an undifferentiated
    coefficient vector.

    A multivector decomposes into grades ``H = sum_{k=0}^{n} <H>_k`` where
    ``<H>_k`` collects the coefficients of all blades with ``k`` generators
    (e.g. grade 0 = scalar, grade 1 = vector, grade 2 = bivector, ...).  This
    module applies an independent ``nn.Linear(re, re)`` per grade,

    .. math::
        \Phi(H) = \sum_{k=0}^{n} \Phi_k(\langle H \rangle_k)

    shared across every blade of the same grade (so the transform is
    equivariant to how blades of equal grade are ordered), rather than one
    unconstrained ``nn.Linear(D, D)`` mixing all blades/grades together.

    Parameters
    ----------
    d : int
        Number of blades (``2^n``), as in :class:`FullDeCaL` /
        :class:`CliffordCompGCNLayer`.
    re : int
        Per-blade embedding width (``embedding_dim // d``).
    bias : bool
        Whether each per-grade ``nn.Linear`` has a bias term.
    """

    def __init__(self, d: int, re: int, bias: bool = False):
        super().__init__()
        self.d = d
        self.re = re
        # grade(K) = popcount(K) for blade index K in [0, d)
        idx = torch.arange(d, dtype=torch.long)
        grade = torch.zeros(d, dtype=torch.long)
        tmp = idx.clone()
        for _ in range(max(1, int(math.log2(d)) + 1)):
            grade += (tmp & 1)
            tmp >>= 1
        self.register_buffer("grade_of_blade", grade)
        num_grades = int(grade.max().item()) + 1
        self.grade_linears = nn.ModuleList([nn.Linear(re, re, bias=bias) for _ in range(num_grades)])
        for lin in self.grade_linears:
            nn.init.xavier_normal_(lin.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the grade-aware transform.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., D)`` with ``D = d * re``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., D)``.
        """
        *lead, D = x.shape
        assert D == self.d * self.re, f"CliffordLinear expected last dim {self.d * self.re}, got {D}"
        H = x.view(*lead, self.d, self.re)
        out = torch.zeros_like(H)
        for g, lin in enumerate(self.grade_linears):
            mask = self.grade_of_blade == g
            if mask.any():
                out[..., mask, :] = lin(H[..., mask, :])
        return out.reshape(*lead, D)


class FullCliffordGNNLayer(CliffordCompGCNLayer):
    """CompGCN layer combining the Clifford geometric product (``phi``, reused
    unchanged from :class:`CliffordCompGCNLayer`) with grade-aware
    :class:`CliffordLinear` message/update transformations, replacing the
    conventional unconstrained ``nn.Linear(D, D)`` used by
    :class:`CliffordCompGCNLayer`.

    This is the only difference vs. ``CliffordCompGCNLayer``: the geometric
    product composition is identical, only the *linear transformations*
    applied to the (flattened) Clifford coefficients are made Clifford/grade
    aware.
    """

    def __init__(self, dim: int, d: int, re: int, coeff_table: torch.Tensor, K_table: torch.Tensor,
                 dropout: float = 0.0):
        super().__init__(dim, d, re, coeff_table, K_table, dropout=dropout)
        self.W_0 = CliffordLinear(d, re)
        self.W_in = CliffordLinear(d, re)
        self.W_out = CliffordLinear(d, re)
        self.W_loop = CliffordLinear(d, re)
        self.W_rel = CliffordLinear(d, re)


# --------------------------------------------------------------------------- #
#  Full models                                                                 #
# --------------------------------------------------------------------------- #
class _CliffordCompGCNCore(nn.Module):
    r"""Clifford-valued relational GNN encoder + Clifford (DeCaL-style) decoder.

    Encoder
    -------
    ``num_layers`` stacked :class:`CliffordCompGCNLayer`:

        h_v^{(l+1)} = sigma( W_0 h_v^{(l)} + sum_{(u,r,v) in E} W_r (h_u^{(l)} (x) h_r^{(l)}) )

    Decoder
    -------
    Identical geometric-product scoring function used by ``FullDeCaL``:

        f(h, r, t) = < h^{(L)} (x) r^{(L)} , t^{(L)} >

    Parameters
    ----------
    num_entities, num_relations : int
        Sizes of the *original* KG vocabularies.
    embedding_dim : int
        Must satisfy ``embedding_dim % 2^(p+q+r) == 0`` (same convention as
        ``FullDeCaL``).
    p, q, r : int
        Fixed Clifford signature Cl_{p,q,r}.  ``n = p + q + r``,
        ``d = 2^n`` blades, ``re = embedding_dim // d``.
    num_layers : int
        Number of message-passing layers.
    dropout : float
        Dropout applied after each layer's aggregation.
    layer_cls : Type[CliffordCompGCNLayer]
        Layer class to stack (defaults to :class:`CliffordCompGCNLayer`).
        :class:`FullCliffordGNN` reuses this exact core, passing
        :class:`FullCliffordGNNLayer` instead, so no logic is duplicated
        between the two models beyond the layer implementation itself.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 p: int = 1, q: int = 1, r: int = 0, num_layers: int = 1, dropout: float = 0.0,
                 layer_cls: "Type[CliffordCompGCNLayer]" = None):
        super().__init__()
        if layer_cls is None:
            layer_cls = CliffordCompGCNLayer
        n = p + q + r
        d = 1 << n
        if embedding_dim % d != 0:
            raise AssertionError(
                f"CliffordCompGCN requires embedding_dim ({embedding_dim}) divisible by 2^(p+q+r) = 2^{n} = {d}."
            )
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.p, self.q, self.r_sig, self.n, self.d = p, q, r, n, d
        self.re = embedding_dim // d
        self.num_layers = num_layers

        # Fixed Clifford signature -> frozen coefficient table (same recipe as
        # FullDeCaL's fixed-signature mode).
        sign_table, K_table, intersection_table, bits = _build_sign_table(n)
        eta = torch.tensor([1.0] * p + [-1.0] * q + [0.0] * r, dtype=torch.float32)
        eta_blade = (bits * eta + (1.0 - bits)).prod(dim=1)
        coeff_table = sign_table * eta_blade[intersection_table]
        self.register_buffer("coeff_table", coeff_table)
        self.register_buffer("K_table", K_table)

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        # forward relations + inverse relations + 1 self-loop relation
        self.relation_embeddings = nn.Embedding(2 * num_relations + 1, embedding_dim)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        self.layers = nn.ModuleList([
            layer_cls(embedding_dim, d, self.re, coeff_table, K_table, dropout=dropout)
            for _ in range(num_layers)
        ])

    def encode(self, edge_index: torch.LongTensor,
               edge_type: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the stacked message-passing layers.

        Returns
        -------
        ent : torch.Tensor
            Shape ``(num_entities, embedding_dim)`` updated entity
            representations.
        rel : torch.Tensor
            Shape ``(2 * num_relations + 1, embedding_dim)`` updated relation
            representations.
        """
        ent = self.entity_embeddings.weight
        rel = self.relation_embeddings.weight
        for layer in self.layers:
            ent, rel = layer(ent, rel, edge_index, edge_type, self.num_relations)
        return ent, rel

    def _decode(self, ent: torch.Tensor, rel: torch.Tensor, h_idx: torch.LongTensor, r_idx: torch.LongTensor,
                t_idx: Optional[torch.LongTensor] = None) -> torch.Tensor:
        h = ent.index_select(0, h_idx).view(-1, self.d, self.re)
        r = rel.index_select(0, r_idx).view(-1, self.d, self.re)
        z = clifford_geometric_product(h, r, self.coeff_table, self.K_table, self.d, self.re)
        if t_idx is not None:
            t = ent.index_select(0, t_idx).view(-1, self.d, self.re)
            return torch.einsum('bkr,bkr->b', z, t)
        T = ent.view(-1, self.d, self.re)
        return torch.einsum('bkr,ekr->be', z, T)

    def forward_triples(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor,
                         triples: torch.LongTensor) -> torch.Tensor:
        """NegSample-style scoring: ``(B, 3) -> (B,)``."""
        ent, rel = self.encode(edge_index, edge_type)
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        return self._decode(ent, rel, h, r, t)

    def forward_k_vs_all(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor,
                          hr: torch.LongTensor) -> torch.Tensor:
        """KvsAll-style scoring: ``(B, 2) -> (B, num_entities)``."""
        ent, rel = self.encode(edge_index, edge_type)
        h, r = hr[:, 0], hr[:, 1]
        return self._decode(ent, rel, h, r, t_idx=None)


class _FullCliffordGNNCore(_CliffordCompGCNCore):
    """Encoder identical to :class:`_CliffordCompGCNCore` (Clifford
    geometric-product composition, sum aggregation, Full-DeCaL decoder) but
    stacking :class:`FullCliffordGNNLayer` instead of
    :class:`CliffordCompGCNLayer`, i.e. the message/update *transformations*
    ``W_0, W_in, W_out, W_loop, W_rel`` are grade-aware :class:`CliffordLinear`
    modules rather than unconstrained ``nn.Linear(D, D)``.

    All encode/decode logic is inherited unchanged from
    :class:`_CliffordCompGCNCore` -- only the layer class differs.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 p: int = 1, q: int = 1, r: int = 0, num_layers: int = 1, dropout: float = 0.0):
        super().__init__(num_entities, num_relations, embedding_dim, p=p, q=q, r=r,
                          num_layers=num_layers, dropout=dropout, layer_cls=FullCliffordGNNLayer)


class _CompGCNCore(nn.Module):
    """Vanilla CompGCN baseline (sub / mult / corr composition), same skeleton
    as :class:`_CliffordCompGCNCore` but without the Clifford geometric product.

    Decoder is a simple DistMult-style inner product ``<phi(h, r), t>`` so
    that the *only* controlled variable between this baseline and
    ``CliffordCompGCN`` is the composition operator, not the decoder.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 composition: str = "corr", num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.composition = composition
        self._phi = _VECTOR_COMPOSITIONS[composition]

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(2 * num_relations + 1, embedding_dim)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        self.layers = nn.ModuleList([
            CompGCNLayer(embedding_dim, composition=composition, dropout=dropout) for _ in range(num_layers)
        ])

    def encode(self, edge_index: torch.LongTensor,
               edge_type: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ent = self.entity_embeddings.weight
        rel = self.relation_embeddings.weight
        for layer in self.layers:
            ent, rel = layer(ent, rel, edge_index, edge_type, self.num_relations)
        return ent, rel

    def _decode(self, ent, rel, h_idx, r_idx, t_idx=None):
        h = ent.index_select(0, h_idx)
        r = rel.index_select(0, r_idx)
        z = self._phi(h, r)
        if t_idx is not None:
            t = ent.index_select(0, t_idx)
            return (z * t).sum(dim=-1)
        return z @ ent.t()

    def forward_triples(self, edge_index, edge_type, triples):
        ent, rel = self.encode(edge_index, edge_type)
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        return self._decode(ent, rel, h, r, t)

    def forward_k_vs_all(self, edge_index, edge_type, hr):
        ent, rel = self.encode(edge_index, edge_type)
        h, r = hr[:, 0], hr[:, 1]
        return self._decode(ent, rel, h, r, t_idx=None)


# --------------------------------------------------------------------------- #
#  BaseKGE-compatible wrappers -- these plug directly into the EXISTING       #
#  dicee CLI / DICE_Trainer / Evaluator pipeline (dicee/scripts/run.py,       #
#  dicee/static_funcs.py::MODEL_REGISTRY, dicee/evaluation/evaluator.py).     #
#  No new training loop, negative sampler, or evaluation code is needed --    #
#  everything below reuses the existing infrastructure.                      #
# --------------------------------------------------------------------------- #
class _GraphBaseKGE(BaseKGE):
    """Shared plumbing for graph-based encoders inside the ``BaseKGE`` API.

    Builds the CompGCN-style message-passing graph once at construction time
    from ``args['train_set']`` (attached by
    :meth:`dicee.executer.Execute._update_args_from_kg`), and overrides
    ``forward_triples``/``forward_k_vs_all`` to match the ``(x) -> scores``
    signature expected by :class:`BaseKGE`, :class:`BaseKGELightning`, and
    :mod:`dicee.evaluation.evaluator` -- i.e. no bespoke training/eval code.
    """

    def _init_graph(self):
        train_set = self.args.get("train_set", None)
        if train_set is None:
            # Unit tests / KGE(...) inference construct models without a
            # live train_set; fall back to an all-self-loop graph so the
            # model is at least constructible and runnable.
            edge_index, edge_type = build_compgcn_graph(
                torch.zeros((0, 3), dtype=torch.long), self.num_entities, self.num_relations
            )
        else:
            triples = torch.as_tensor(train_set, dtype=torch.long)
            edge_index, edge_type = build_compgcn_graph(triples, self.num_entities, self.num_relations)
        self.register_buffer("_edge_index", edge_index)
        self.register_buffer("_edge_type", edge_type)


class CliffordCompGCN(_GraphBaseKGE):
    """CompGCN with the vector composition operator replaced by the Clifford
    geometric product, registered in ``MODEL_REGISTRY`` and usable via
    ``--model CliffordCompGCN`` exactly like any other ``dicee`` model.

    ``args`` recognises the extra keys ``p``, ``q``, ``r`` (Clifford
    signature, default ``1, 1, 0``) and ``num_gcn_layers`` (default ``1``),
    on top of the standard ``BaseKGE`` args.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "CliffordCompGCN"
        p = int(self.args.get("p", 1) or 1)
        q = int(self.args.get("q", 1) or 1)
        r = int(self.args.get("r", 0) or 0)
        num_layers = int(self.args.get("num_gcn_layers", 1) or 1)
        # Replace BaseKGE's plain embedding tables with the Clifford-GNN core
        # (which owns its own entity/relation embeddings + message-passing
        # layers under the FullDeCaL blade convention D = 2^(p+q+r) * re).
        del self.entity_embeddings
        del self.relation_embeddings
        self.core = _CliffordCompGCNCore(
            self.num_entities, self.num_relations, self.embedding_dim, p=p, q=q, r=r, num_layers=num_layers
        )
        self._init_graph()

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.core.forward_triples(self._edge_index, self._edge_type, x)

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.core.forward_k_vs_all(self._edge_index, self._edge_type, x)


class FullCliffordGNN(_GraphBaseKGE):
    """CliffordCompGCN extended with Clifford/grade-aware message and update
    transformations, registered in ``MODEL_REGISTRY`` and usable via
    ``--model FullCliffordGNN`` exactly like any other ``dicee`` model.

    Whereas :class:`CliffordCompGCN` performs the Clifford geometric-product
    composition ``phi(h_u, h_r) = h_u (x) h_r`` but then applies conventional
    unconstrained ``nn.Linear(D, D)`` transformations to the flattened
    Clifford coefficients, ``FullCliffordGNN`` replaces those transformations
    with grade-aware :class:`CliffordLinear` modules that act independently
    on each grade ``<H>_k`` of the multivector ``H = sum_{k=0}^{n} <H>_k``:

        Phi(H) = sum_{k=0}^{n} Phi_k(<H>_k)

    Everything else (Clifford signature convention, blade layout, geometric
    product, sum aggregation, Full-DeCaL decoder) is identical to
    ``CliffordCompGCN`` -- see :class:`_FullCliffordGNNCore`.

    ``args`` recognises the same extra keys as ``CliffordCompGCN``: ``p``,
    ``q``, ``r`` (Clifford signature, default ``1, 1, 0``) and
    ``num_gcn_layers`` (default ``1``).
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "FullCliffordGNN"
        p = int(self.args.get("p", 1) or 1)
        q = int(self.args.get("q", 1) or 1)
        r = int(self.args.get("r", 0) or 0)
        num_layers = int(self.args.get("num_gcn_layers", 1) or 1)
        del self.entity_embeddings
        del self.relation_embeddings
        self.core = _FullCliffordGNNCore(
            self.num_entities, self.num_relations, self.embedding_dim, p=p, q=q, r=r, num_layers=num_layers
        )
        self._init_graph()

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.core.forward_triples(self._edge_index, self._edge_type, x)

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.core.forward_k_vs_all(self._edge_index, self._edge_type, x)


class CompGCN(_GraphBaseKGE):
    """Vanilla CompGCN baseline (sub/mult/corr composition), registered in
    ``MODEL_REGISTRY`` and usable via ``--model CompGCN``.

    ``args`` recognises the extra keys ``composition``
    (``"sub"|"mult"|"corr"``, default ``"corr"``) and ``num_gcn_layers``
    (default ``1``).
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "CompGCN"
        composition = self.args.get("composition", "corr") or "corr"
        num_layers = int(self.args.get("num_gcn_layers", 1) or 1)
        del self.entity_embeddings
        del self.relation_embeddings
        self.core = _CompGCNCore(
            self.num_entities, self.num_relations, self.embedding_dim,
            composition=composition, num_layers=num_layers
        )
        self._init_graph()

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.core.forward_triples(self._edge_index, self._edge_type, x)

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.core.forward_k_vs_all(self._edge_index, self._edge_type, x)
