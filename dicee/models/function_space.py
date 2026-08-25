import logging

import numpy as np
import torch

from .base_model import BaseKGE

logger = logging.getLogger(__name__)


class FMult(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'FMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = 50
        # self.gamma = torch.rand(self.k, self.num_sample) [0,1) uniform=> worse results
        self.gamma = torch.randn(self.k, self.num_sample)  # N(0,1)
        # Lazy import
        from scipy.special import roots_legendre
        roots, weights = roots_legendre(self.num_sample)
        self.roots = torch.from_numpy(roots).repeat(self.k, 1).float()  # shape self.k by self.n
        self.weights = torch.from_numpy(weights).reshape(1, -1).float()  # shape 1 by self.n


    def compute_func(self, weights: torch.FloatTensor, x) -> torch.FloatTensor:
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Forward Pass
        out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
        out2 = w2 @ out1
        return out2  # no non-linearity => better results

    def chain_func(self, weights, x: torch.FloatTensor):
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Perform the forward pass
        out1 = torch.tanh(torch.bmm(w1, x))
        out2 = torch.bmm(w2, out1)
        return out2

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        # (2) Compute NNs on \Gamma
        # Logits via FDistMult...
        # h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # r_x = self.compute_func(rel_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # out = h_x * r_x * t_x  # batch, \mathbb{R}^k, |gamma|
        # (2) Compute NNs on \Gamma
        self.gamma=self.gamma.to(head_ent_emb.device)

        h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.chain_func(weights=rel_ent_emb, x=h_x)  # batch, \mathbb{R}^k, |\Gamma|
        # (3) Compute |\Gamma| predictions
        out = torch.sum(r_h_x * t_x, dim=1)  # batch, |gamma| #
        # (4) Average (3) over \Gamma
        out = torch.mean(out, dim=1)  # batch
        return out

class GFMult(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'GFMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = 250
        # Lazy import
        from scipy.special import roots_legendre
        roots, weights = roots_legendre(self.num_sample)
        self.roots = torch.from_numpy(roots).repeat(self.k, 1).float()  # shape self.k by self.n
        self.weights = torch.from_numpy(weights).reshape(1, -1).float()  # shape 1 by self.n

    def compute_func(self, weights: torch.FloatTensor, x) -> torch.FloatTensor:
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Forward Pass
        out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
        out2 = w2 @ out1
        return out2  # no non-linearity => better results

    def chain_func(self, weights, x: torch.FloatTensor):
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Perform the forward pass
        out1 = torch.tanh(torch.bmm(w1, x))
        out2 = torch.bmm(w2, out1)
        return out2

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        # (2) Compute NNs on \Gamma
        self.roots=self.roots.to(head_ent_emb.device)
        self.weights=self.weights.to(head_ent_emb.device)

        h_x = self.compute_func(head_ent_emb, x=self.roots)  # batch, \mathbb{R}^k, |\Gamma|
        t_x = self.compute_func(tail_ent_emb, x=self.roots)  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.chain_func(weights=rel_ent_emb, x=h_x)  # batch, \mathbb{R}^k, |\Gamma|
        # (3) Compute |\Gamma| predictions.
        out = torch.sum(r_h_x * t_x, dim=1)*self.weights  # batch, |gamma| #
        # (4) Average (3) over \Gamma
        out = torch.mean(out, dim=1)  # batch
        return out


class FMult2(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'FMult2'
        self.n_layers = 3
        tuned_embedding_dim = False
        while int(np.sqrt((self.embedding_dim - 1) / self.n_layers)) != np.sqrt(
                (self.embedding_dim - 1) / self.n_layers):
            self.embedding_dim += 1
            tuned_embedding_dim = True
        if tuned_embedding_dim:
            logger.warning(f"Embedding dimension reset to {self.embedding_dim} to fit model architecture!")
        self.k = int(np.sqrt((self.embedding_dim - 1) // self.n_layers))
        self.n = 50
        self.a, self.b = -1.0, 1.0
        # self.score_func = "vtp" # "vector triple product"
        # self.score_func = "trilinear"
        self.score_func = "compositional"
        # self.score_func = "full-compositional"
        # self.discrete_points = torch.linspace(self.a, self.b, steps=self.n)
        self.discrete_points = torch.linspace(self.a, self.b, steps=self.n).repeat(self.k, 1)

        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def build_func(self, Vec):
        n = len(Vec)
        # (1) Construct self.n_layers layered neural network
        W = list(torch.hsplit(Vec[:, :-1], self.n_layers))
        # (2) Reshape weights of the layers
        for i, w in enumerate(W):
            W[i] = w.reshape(n, self.k, self.k)
        return W, Vec[:, -1]

    def build_chain_funcs(self, list_Vec):
        list_W = []
        list_b = []
        for Vec in list_Vec:
            W_, b = self.build_func(Vec)
            list_W.append(W_)
            list_b.append(b)

        W = list_W[-1][1:]
        for i in range(len(list_W) - 1):
            for j, w in enumerate(list_W[i]):
                if i == 0 and j == 0:
                    W_temp = w
                else:
                    W_temp = w @ W_temp
            W_temp = W_temp + list_b[i].reshape(-1, 1, 1)
        W_temp = list_W[-1][0] @ W_temp / ((len(list_Vec) - 1) * w.shape[1])
        W.insert(0, W_temp)
        return W, list_b[-1]

    def compute_func(self, W, b, x) -> torch.FloatTensor:
        out = W[0] @ x
        for i, w in enumerate(W[1:]):
            if i % 2 == 0:  # no non-linearity => better results
                out = out + torch.tanh(w @ out)
            else:
                out = out + w @ out
        return out + b.reshape(-1, 1, 1)

    def function(self, list_W, list_b):
        def f(x):
            if len(list_W) == 1:
                return self.compute_func(list_W[0], list_b[0], x)
            score = self.compute_func(list_W[0], list_b[0], x)
            for W, b in zip(list_W[1:], list_b[1:]):
                score = score * self.compute_func(W, b, x)
            return score

        return f

    def trapezoid(self, list_W, list_b):
        return torch.trapezoid(self.function(list_W, list_b)(self.discrete_points), x=self.discrete_points, dim=-1).sum(
            dim=-1)

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        if self.discrete_points.device != head_ent_emb.device:
            self.discrete_points = self.discrete_points.to(head_ent_emb.device)
        if self.score_func == "vtp":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = -self.trapezoid([t_W], [t_b]) * self.trapezoid([h_W, r_W], [h_b, r_b]) + self.trapezoid([r_W], [
                r_b]) * self.trapezoid([t_W, h_W], [t_b, h_b])
        elif self.score_func == "compositional":
            t_W, t_b = self.build_func(tail_ent_emb)
            chain_W, chain_b = self.build_chain_funcs([head_ent_emb, rel_emb])
            out = self.trapezoid([chain_W, t_W], [chain_b, t_b])
        elif self.score_func == "full-compositional":
            chain_W, chain_b = self.build_chain_funcs([head_ent_emb, rel_emb, tail_ent_emb])
            out = self.trapezoid([chain_W], [chain_b])
        elif self.score_func == "trilinear":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = self.trapezoid([h_W, r_W, t_W], [h_b, r_b, t_b])
        return out


class LFMult1(BaseKGE):

    r'''Embedding with trigonometric functions. We represent all entities and relations in the complex number space as:
      f(x) = \sum_{k=0}^{k=d-1}wk e^{kix}. and use the three differents scoring function as in the paper to evaluate the score'''

    def __init__(self,args):
        super().__init__(args)
        self.name = 'LFMult1'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)

    def forward_triples(self, idx_triple): # idx_triplet = (h_idx, r_idx, t_idx) #change this to the forward_triples

        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)

        score = self.vtp_score(head_ent_emb,rel_emb,tail_ent_emb)

        return score

    def tri_score(self,h,r,t):

        i_range, j_range, k_range = torch.meshgrid(torch.arange(self.embedding_dim),torch.arange(self.embedding_dim),torch.arange(self.embedding_dim))
        eps = 10**-6   #for stability reason
        cond = i_range + j_range == k_range

        s1 = torch.sum(torch.where(~cond, torch.zeros_like(~cond),  h[:, i_range] * r[:, j_range] * t[:, k_range]),dim=(-3,-2,-1)) # sum on i+j = k

        s2 = torch.sum(torch.where(cond, torch.zeros_like(cond), torch.sin(i_range + j_range - k_range) \
                                * h[:, i_range] * r[:, j_range] * t[:, k_range] /(eps+i_range + j_range - k_range)),dim=(-3,-2,-1))# sum on i+j != k
        s = s1 + s2 # combine the two sums.
        return s

    def vtp_score(self,h,r,t):

        i_range, j_range = torch.meshgrid(torch.arange(self.embedding_dim),torch.arange(self.embedding_dim))
        eps = 10**-6   #for stability reason
        cond = i_range == j_range

        p1 = torch.sum(torch.where(cond, torch.zeros_like(cond), torch.sin(i_range - j_range) \
                                * h[:, i_range] * t[:, j_range] /(eps+i_range - j_range)),dim=(-3,-2,-1)) \
                                    + torch.sum(h[:, i_range] * t[:, i_range],dim=(-3,-2,-1))# sum on i != j
        i_1 = torch.arange(1,self.embedding_dim)
        p2 = torch.sum(r[:, i_1] * torch.sin(i_1)/(i_1) ,dim=-1) + r[:,0]

        s1 = p1*p2

        p3 = torch.sum(torch.where(cond, torch.zeros_like(cond), torch.sin(i_range - j_range) \
                                * r[:, i_range] * t[:, j_range] /(eps+i_range - j_range)),dim=(-3,-2,-1)) \
                                    + torch.sum(r[:, i_range] * t[:, i_range],dim=(-3,-2,-1))# sum on i != j

        p4 = torch.sum(h[:, i_1] * torch.sin(i_1)/(i_1) ,dim=-1) + h[:,0]
        s2 = p3*p4


        s = s1 - s2 # combine the two sums.
        return s

class LFMult(BaseKGE):
    r"""Learnable Function Multiplication for KGE.

    Each entity/relation embedding is interpreted as the parameters of
    **m independent single-input neural networks** evaluated over a shared
    grid of ``n_quad`` quadrature points in [0, 1].  The triple score is the
    numerical integral of the point-wise product of the three functions:

    .. math::

        f(h, r, t) = \int_0^1 \sum_{k=1}^m f_k^h(x)\, f_k^r(x)\, f_k^t(x)\, dx

    The function at channel *k* is a depth-``degree`` network:

    * ``degree=0``: :math:`f_k(x) = w_k \cdot x`  (linear)
    * ``degree=1``: :math:`f_k(x) = \tanh(w_k x + b_k)`  (1-layer)
    * ``degree≥2``: successive tanh layers, one per extra block

    **Embedding layout** — the ``embedding_dim``-dimensional vector is split
    evenly into ``degree+1`` blocks of size ``m = embedding_dim // (degree+1)``.
    Block 0 is the slope, block 1 the bias, blocks 2…degree add depth.

    **KvsAll scalability** — ``forward_k_vs_all`` avoids the cubic O(B·|E|·m)
    loop by separating head×relation from tails:

    .. math::

        f(h, r, t_e) = \sum_j w_j \underbrace{\left[\sum_k f_k^h(x_j)
        f_k^r(x_j)\right]}_{\text{HR}_{b,j}}\; \underbrace{\left[\sum_k
        f_k^{t_e}(x_j)\right]}_{\text{T}_{e,j}}

    This is a single matrix multiply :math:`\mathbf{HR}_w \, \mathbf{T}^\top`
    of shape ``(B, n_quad) × (n_quad, |E|) = (B, |E|)``.

    Parameters
    ----------
    args : dict
        ``degree`` (int, default 1) — network depth per channel.
        ``n_quad`` (int, default 100) — quadrature points; reduce for large |E|.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'LFMult'

        self.degree = int(self.args.get("degree", 1))
        self.n_quad = int(self.args.get("n_quad", 100))

        assert self.embedding_dim % (self.degree + 1) == 0, (
            f"LFMult: embedding_dim={self.embedding_dim} must be divisible by "
            f"(degree+1)={self.degree + 1}."
        )
        self.m = self.embedding_dim // (self.degree + 1)

        self.entity_embeddings   = torch.nn.Embedding(self.num_entities,  self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)

        # Quadrature grid — registered as a buffer so it moves with .to(device).
        x = torch.linspace(0.0, 1.0, self.n_quad)
        self.register_buffer("x_values", x)

        # Precompute trapezoid weights once (uniform spacing, so dx is constant).
        dx = x[1] - x[0]
        trap_w = torch.ones(self.n_quad) * dx
        trap_w[0]  *= 0.5
        trap_w[-1] *= 0.5
        self.register_buffer("trap_weights", trap_w)   # (n_quad,)

        logger.info(
            f"LFMult | m={self.m}  degree={self.degree}  n_quad={self.n_quad}"
        )

    # ── Core: evaluate embeddings as functions at quadrature points ────────

    def _eval(self, emb: torch.Tensor) -> torch.Tensor:
        """Map a batch of embeddings to function evaluations at quadrature points.

        Each embedding ``e ∈ ℝ^D`` parameterises *m* networks
        ``f_k : [0,1] → ℝ``.  With ``D = m · (degree+1)`` the embedding is
        split into ``(degree+1)`` blocks of size ``m``.

        Parameters
        ----------
        emb : (B, D)

        Returns
        -------
        (B, m, n_quad) — ``out[b, k, j] = f_k(x_j)``
        """
        B = emb.size(0)
        # blocks[b, d, k] = d-th parameter of channel k for sample b
        blocks = emb.view(B, self.degree + 1, self.m)  # (B, 1+deg, m)
        # x broadcast shape: (1, 1, n_quad)
        x = self.x_values.view(1, 1, -1)

        # First block: slope.  out = w0 * x  →  (B, m, n_quad)
        out = blocks[:, 0, :].unsqueeze(2) * x   # (B, m, 1) × (1, 1, n_quad)

        # Each subsequent block adds a tanh layer:
        # out ← tanh(out + w_d)
        for d in range(1, self.degree + 1):
            wd = blocks[:, d, :].unsqueeze(2)     # (B, m, 1)
            out = torch.tanh(out + wd)

        return out   # (B, m, n_quad)

    # ── Scoring ────────────────────────────────────────────────────────────

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        """NegSample / trilinear scoring.

        .. math::

            f(h,r,t) = \\int_0^1 \\sum_k f_k^h(x)\\, f_k^r(x)\\, f_k^t(x)\\, dx

        Parameters
        ----------
        idx_triple : (B, 3)

        Returns
        -------
        (B,)
        """
        h_emb, r_emb, t_emb = self.get_triple_representation(idx_triple)

        h = self._eval(h_emb)   # (B, m, n_quad)
        r = self._eval(r_emb)   # (B, m, n_quad)
        t = self._eval(t_emb)   # (B, m, n_quad)

        # Point-wise product summed over channels → (B, n_quad)
        integrand = (h * r * t).sum(dim=1)

        # Trapezoid integration: dot with precomputed weights
        return integrand @ self.trap_weights   # (B,)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        """KvsAll scoring — scores every entity as tail in one matrix multiply.

        Derivation
        ----------
        .. math::

            f(h,r,t_e) &= \\int_0^1 \\underbrace{\\sum_k f_k^h(x)f_k^r(x)}_{\\text{HR}(x)}
                          \\underbrace{\\sum_k f_k^{t_e}(x)}_{\\text{T}_e(x)}\\, dx \\\\
                       &\\approx \\sum_j w_j \\, \\text{HR}_{b,j} \\, \\text{T}_{e,j}
                        = \\mathbf{HR}_w \\, \\mathbf{T}^\\top

        where :math:`\\mathbf{HR}_w \\in \\mathbb{R}^{B \\times n_q}` has the
        trapezoid weights folded in, and :math:`\\mathbf{T} \\in
        \\mathbb{R}^{|E| \\times n_q}`.

        **Complexity**: O(B·m·n_q) + O(|E|·m·n_q) + O(B·|E|·n_q) — the last
        term dominates but n_q is small (default 100), making it tractable
        for datasets up to ~10k entities.  For larger KGs reduce ``n_quad``.

        Parameters
        ----------
        x : (B, 2)  integer [head_idx, relation_idx]

        Returns
        -------
        (B, |E|)
        """
        h_emb, r_emb = self.get_head_relation_representation(x)

        h = self._eval(h_emb)   # (B, m, n_quad)
        r = self._eval(r_emb)   # (B, m, n_quad)

        # HR[b, j] = Σ_k f_k^h(x_j) * f_k^r(x_j)
        HR = (h * r).sum(dim=1)   # (B, n_quad)

        # Evaluate ALL entity embeddings at quadrature points
        T_all = self._eval(self.entity_embeddings.weight)   # (|E|, m, n_quad)
        T_sum = T_all.sum(dim=1)                            # (|E|, n_quad)

        # Fold trapezoid weights into HR: HR_w[b, j] = w_j * HR[b, j]
        HR_w = HR * self.trap_weights.unsqueeze(0)   # (B, n_quad)

        # score[b, e] = Σ_j HR_w[b,j] * T_sum[e,j]
        return HR_w @ T_sum.t()   # (B, |E|)
