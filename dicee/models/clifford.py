import logging

import torch

from .base_model import BaseKGE

logger = logging.getLogger(__name__)


class Keci(BaseKGE):
    """Keci: Knowledge Graph Embedding via Clifford Algebra.

    Embeds entities and relations as multi-vectors in the Clifford algebra
    Cl_{p,q}(R^d) and scores triples via the Clifford product.  The algebra
    is parameterised by two non-negative integers *p* and *q*:

    * ``p = 0, q = 0`` — reduces to a standard bilinear (DistMult-like) model.
    * ``p = 0, q = 1`` — equivalent to ComplEx.
    * Larger ``p`` and ``q`` capture higher-order geometric interactions.

    The embedding dimension must satisfy ``embedding_dim % (p + q + 1) == 0``;
    the resulting quotient is stored as ``self.r``.

    Parameters
    ----------
    args : dict
        Configuration dictionary.  Recognised keys (beyond those in
        :class:`BaseKGE`): ``p`` (int, default 0) and ``q`` (int, default 0).

    References
    ----------
    Demir et al., *Clifford Embeddings — A Generalized Approach for Embedding
    in Normed Algebras*, ECML 2023.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Keci'
        self.p = self.args.get("p", 0)
        self.q = self.args.get("q", 0)
        if self.p is None:
            self.p = 0
        if self.q is None:
            self.q = 0
        self.r = self.embedding_dim / (self.p + self.q + 1)
        try:
            assert self.r.is_integer()
        except AssertionError:
            raise AssertionError(f'r = embedding_dim / (p + q+ 1) must be a whole number\n'
                                 f'Currently {self.r}={self.embedding_dim} / ({self.p}+ {self.q} +1)')
        self.r = int(self.r)
        self.requires_grad_for_interactions = True
        # Initialize parameters for dimension scaling
        # TODO:Do we need coefficients for the real part ?
        if self.p > 0:
            self.p_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.p)
            torch.nn.init.zeros_(self.p_coefficients.weight)
        if self.q > 0:
            self.q_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.q)
            torch.nn.init.zeros_(self.q_coefficients.weight)

    def compute_sigma_pp(self, hp, rp):
        r"""
        Compute  sigma_{pp} = \sum_{i=1}^{p-1} \sum_{k=i+1}^p (h_i r_k - h_k r_i) e_i e_k

        sigma_{pp} captures the interactions between along p bases
        For instance, let p e_1, e_2, e_3, we compute interactions between e_1 e_2, e_1 e_3 , and e_2 e_3
        This can be implemented with a nested two for loops

                        results = []
                        for i in range(p - 1):
                            for k in range(i + 1, p):
                                results.append(hp[:, :, i] * rp[:, :, k] - hp[:, :, k] * rp[:, :, i])
                        sigma_pp = torch.stack(results, dim=2)
                        assert sigma_pp.shape == (b, r, int((p * (p - 1)) / 2))

        Yet, this computation would be quite inefficient. Instead, we compute interactions along all p,
        e.g., e1e1, e1e2, e1e3,
              e2e1, e2e2, e2e3,
              e3e1, e3e2, e3e3
        Then select the triangular matrix without diagonals: e1e2, e1e3, e2e3.
        """
        # Compute indexes for the upper triangle of p by p matrix
        indices = torch.triu_indices(self.p, self.p, offset=1)
        # Compute p by p operations
        sigma_pp = torch.einsum('nrp,nrx->nrpx', hp, rp) - torch.einsum('nrx,nrp->nrpx', hp, rp)
        sigma_pp = sigma_pp[:, :, indices[0], indices[1]]
        return sigma_pp

    def compute_sigma_qq(self, hq, rq):
        r"""
        Compute  sigma_{qq} = \sum_{j=1}^{p+q-1} \sum_{k=j+1}^{p+q} (h_j r_k - h_k r_j) e_j e_k
        sigma_{q} captures the interactions between along q bases
        For instance, let q e_1, e_2, e_3, we compute interactions between e_1 e_2, e_1 e_3 , and e_2 e_3
        This can be implemented with a nested two for loops

                        results = []
                        for j in range(q - 1):
                            for k in range(j + 1, q):
                                results.append(hq[:, :, j] * rq[:, :, k] - hq[:, :, k] * rq[:, :, j])
                        sigma_qq = torch.stack(results, dim=2)
                        assert sigma_qq.shape == (b, r, int((q * (q - 1)) / 2))

        Yet, this computation would be quite inefficient. Instead, we compute interactions along all p,
        e.g., e1e1, e1e2, e1e3,
              e2e1, e2e2, e2e3,
              e3e1, e3e2, e3e3
        Then select the triangular matrix without diagonals: e1e2, e1e3, e2e3.
        """
        # Compute indexes for the upper triangle of p by p matrix
        if self.q > 1:
            indices = torch.triu_indices(self.q, self.q, offset=1)
            # Compute p by p operations
            sigma_qq = torch.einsum('nrp,nrx->nrpx', hq, rq) - torch.einsum('nrx,nrp->nrpx', hq, rq)
            sigma_qq = sigma_qq[:, :, indices[0], indices[1]]
        else:
            sigma_qq = torch.zeros((len(hq), self.r, int((self.q * (self.q - 1)) / 2)), device=hq.device)

        return sigma_qq

    def compute_sigma_pq(self, *, hp, hq, rp, rq):
        r"""
        \sum_{i=1}^{p} \sum_{j=p+1}^{p+q} (h_i r_j - h_j r_i) e_i e_j

        results = []
        sigma_pq = torch.zeros(b, r, p, q)
        for i in range(p):
            for j in range(q):
                sigma_pq[:, :, i, j] = hp[:, :, i] * rq[:, :, j] - hq[:, :, j] * rp[:, :, i]
        print(sigma_pq.shape)

        """
        sigma_pq = torch.einsum('nrp,nrq->nrpq', hp, rq) - torch.einsum('nrq,nrp->nrpq', hq, rp)
        assert sigma_pq.shape[1:] == (self.r, self.p, self.q)
        return sigma_pq

    def apply_coefficients(self, hp, hq, rp, rq):
        """ Multiplying a base vector with its scalar coefficient """
        if self.p > 0:
            hp = hp * self.p_coefficients.weight
            rp = rp * self.p_coefficients.weight
        if self.q > 0:
            hq = hq * self.q_coefficients.weight
            rq = rq * self.q_coefficients.weight
        return hp, hq, rp, rq

    def clifford_multiplication(self, h0, hp, hq, r0, rp, rq):
        """ Compute our CL multiplication

        h = h_0 + \\sum_{i=1}^p h_i e_i + \\sum_{j=p+1}^{p+q} h_j e_j
        r = r_0 + \\sum_{i=1}^p r_i e_i + \\sum_{j=p+1}^{p+q} r_j e_j

        ei ^2 = +1     for i =< i =< p
        ej ^2 = -1     for p < j =< p+q
        ei ej = -eje1  for i \neq j

        h r =   sigma_0 + sigma_p + sigma_q + sigma_{pp} + sigma_{q}+ sigma_{pq}
        where
                (1) sigma_0 = h_0 r_0 + \\sum_{i=1}^p (h_0 r_i) e_i - \\sum_{j=p+1}^{p+q} (h_j r_j) e_j

                (2) sigma_p = \\sum_{i=1}^p (h_0 r_i + h_i r_0) e_i

                (3) sigma_q = \\sum_{j=p+1}^{p+q} (h_0 r_j + h_j r_0) e_j

                (4) sigma_{pp} = \\sum_{i=1}^{p-1} \\sum_{k=i+1}^p (h_i r_k - h_k r_i) e_i e_k

                (5) sigma_{qq} = \\sum_{j=1}^{p+q-1} \\sum_{k=j+1}^{p+q} (h_j r_k - h_k r_j) e_j e_k

                (6) sigma_{pq} = \\sum_{i=1}^{p} \\sum_{j=p+1}^{p+q} (h_i r_j - h_j r_i) e_i e_j

        """
        n = len(h0)
        assert h0.shape == (n, self.r) == r0.shape == (n, self.r)
        assert hp.shape == (n, self.r, self.p) == rp.shape == (n, self.r, self.p)
        assert hq.shape == (n, self.r, self.q) == rq.shape == (n, self.r, self.q)
        # (1)
        sigma_0 = h0 * r0 + torch.sum(hp * rp, dim=2) - torch.sum(hq * rq, dim=2)
        assert sigma_0.shape == (n, self.r)
        # (2)
        sigma_p = torch.einsum('nr,nrp->nrp', h0, rp) + torch.einsum('nr,nrp->nrp', r0, hp)
        assert sigma_p.shape == (n, self.r, self.p)
        # (3)
        sigma_q = torch.einsum('nr,nrq->nrq', h0, rq) + torch.einsum('nr,nrq->nrq', r0, hq)
        # (4)
        sigma_pp = self.compute_sigma_pp(hp, rp)
        # (5)
        sigma_qq = self.compute_sigma_qq(hq, rq)
        # (6)
        sigma_pq = torch.einsum('bkp,bkq->bkpq', hp, rq) - torch.einsum('bkp,bkq->bkpq', rp, hq)
        assert sigma_pq.shape == (n, self.r, self.p, self.q)

        return sigma_0, sigma_p, sigma_q, sigma_pp, sigma_qq, sigma_pq

    def construct_cl_multivector(self, x: torch.FloatTensor, r: int, p: int, q: int) -> tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Split a flat embedding vector into the three Clifford components.

        Given an embedding ``x`` of dimension ``d = r + r*p + r*q``, returns
        the scalar part ``a0``, the *p*-blade part ``ap``, and the *q*-blade
        part ``aq``.

        Parameters
        ----------
        x : torch.FloatTensor
            Shape ``(batch_size, d)``.
        r : int
            Scalar block size (``embedding_dim // (p + q + 1)``).
        p : int
            Number of positive-signature basis elements.
        q : int
            Number of negative-signature basis elements.

        Returns
        -------
        a0 : torch.FloatTensor
            Shape ``(batch_size, r)`` — scalar (grade-0) part.
        ap : torch.FloatTensor
            Shape ``(batch_size, r, p)`` — positive-blade part.
        aq : torch.FloatTensor
            Shape ``(batch_size, r, q)`` — negative-blade part.
        """
        batch_size, d = x.shape
        # (1) A_{n \times k}: take the first k columns
        a0 = x[:, :r].view(batch_size, r)
        # (2) B_{n \times p}, C_{n \times q}: take the self.k * self.p columns after the k. column
        if p > 0:
            ap = x[:, r: r + (r * p)].view(batch_size, r, p)
        else:
            ap = torch.zeros((batch_size, r, p), device=self.device)
        if q > 0:
            # (3) B_{n \times p}, C_{n \times q}: take the last self.r * self.q .
            aq = x[:, -(r * q):].view(batch_size, r, q)
        else:
            aq = torch.zeros((batch_size, r, q), device=self.device)
        return a0, ap, aq

    def forward_k_vs_with_explicit(self, x: torch.Tensor) -> torch.FloatTensor:
        """KvsAll scoring using an explicit loop over sigma_pp/qq/pq terms.

        Functionally equivalent to :meth:`forward_k_vs_all` but computes the
        higher-order interaction terms (sigma_pp, sigma_qq, sigma_pq) with
        explicit nested loops rather than einsum contractions.  Kept for
        reference and correctness verification.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, 2)`` integer tensor ``[head_idx, relation_idx]``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_entities)`` score matrix.
        """
        n = len(x)
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq = self.construct_cl_multivector(head_ent_emb, r=self.r, p=self.p, q=self.q)
        r0, rp, rq = self.construct_cl_multivector(rel_ent_emb, r=self.r, p=self.p, q=self.q)
        E = self.entity_embeddings.weight

        # Clifford mul.
        sigma_0 = h0 * r0 + torch.sum(hp * rp, dim=2) - torch.sum(hq * rq, dim=2)
        sigma_p = torch.einsum('nr,nrp->nrp', h0, rp) + torch.einsum('nrp, nr->nrp', hp, r0)
        sigma_q = torch.einsum('nr,nrq->nrq', h0, rq) + torch.einsum('nrq, nr->nrq', hq, r0)

        t0 = E[:, :self.r]

        score_sigma_0 = sigma_0 @ t0.transpose(1, 0)
        if self.p > 0:
            tp = E[:, self.r: self.r + (self.r * self.p)].view(self.num_entities, self.r, self.p)
            score_sigma_p = torch.einsum('bkp,ekp->be', sigma_p, tp)
        else:
            score_sigma_p = 0
        if self.q > 0:
            tq = E[:, -(self.r * self.q):].view(self.num_entities, self.r, self.q)
            score_sigma_q = torch.einsum('bkp,ekp->be', sigma_q, tq)
        else:
            score_sigma_q = 0

        # Compute sigma_pp sigma_qq and sigma_pq
        if self.p > 1:
            results = []
            for i in range(self.p - 1):
                for k in range(i + 1, self.p):
                    results.append(hp[:, :, i] * rp[:, :, k] - hp[:, :, k] * rp[:, :, i])
            sigma_pp = torch.stack(results, dim=2)
            assert sigma_pp.shape == (n, self.r, int((self.p * (self.p - 1)) / 2))
            sigma_pp = torch.sum(sigma_pp, dim=[1, 2]).view(n, 1)
            del results
        else:
            sigma_pp = 0

        if self.q > 1:
            results = []
            for j in range(self.q - 1):
                for k in range(j + 1, self.q):
                    results.append(hq[:, :, j] * rq[:, :, k] - hq[:, :, k] * rq[:, :, j])
            sigma_qq = torch.stack(results, dim=2)
            del results
            assert sigma_qq.shape == (n, self.r, int((self.q * (self.q - 1)) / 2))
            sigma_qq = torch.sum(sigma_qq, dim=[1, 2]).view(n, 1)
        else:
            sigma_qq = 0

        if self.p >= 1 and self.q >= 1:
            sigma_pq = torch.zeros(n, self.r, self.p, self.q)
            for i in range(self.p):
                for j in range(self.q):
                    sigma_pq[:, :, i, j] = hp[:, :, i] * rq[:, :, j] - hq[:, :, j] * rp[:, :, i]
            sigma_pq = torch.sum(sigma_pq, dim=[1, 2, 3]).view(n, 1)
        else:
            sigma_pq = 0

        return score_sigma_0 + score_sigma_p + score_sigma_q + sigma_pp + sigma_qq + sigma_pq

    def k_vs_all_score(self, bpe_head_ent_emb: torch.FloatTensor,
                       bpe_rel_ent_emb: torch.FloatTensor,
                       E: torch.FloatTensor) -> torch.FloatTensor:
        """Compute Clifford-product scores for a head/relation batch vs. all entities.

        Decomposes the head-entity and relation embeddings into Clifford
        multi-vectors, performs the Cl_{p,q} product, and inner-products the
        result against the entity embedding matrix *E*.

        Parameters
        ----------
        bpe_head_ent_emb : torch.FloatTensor
            Head-entity embeddings, shape ``(batch_size, embedding_dim)``.
        bpe_rel_ent_emb : torch.FloatTensor
            Relation embeddings, shape ``(batch_size, embedding_dim)``.
        E : torch.FloatTensor
            All entity embeddings, shape ``(num_entities, embedding_dim)``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_entities)`` score matrix.
        """
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq = self.construct_cl_multivector(bpe_head_ent_emb, r=self.r, p=self.p, q=self.q)
        r0, rp, rq = self.construct_cl_multivector(bpe_rel_ent_emb, r=self.r, p=self.p, q=self.q)

        hp, hq, rp, rq = self.apply_coefficients(hp, hq, rp, rq)
        # (3.1) Extract real part
        t0 = E[:, :self.r]

        num_entities = len(E)
        # (4) Compute a triple score based on interactions described by the basis 1. Eq. 20
        h0r0t0 = torch.einsum('br,er->be', h0 * r0, t0)

        # (5) Compute a triple score based on interactions described by the bases of p {e_1, ..., e_p}. Eq. 21
        if self.p > 0:
            tp = E[:, self.r: self.r + (self.r * self.p)].view(num_entities, self.r, self.p)
            hp_rp_t0 = torch.einsum('brp, er  -> be', hp * rp, t0)
            h0_rp_tp = torch.einsum('brp, erp -> be', torch.einsum('br,  brp -> brp', h0, rp), tp)
            hp_r0_tp = torch.einsum('brp, erp -> be', torch.einsum('brp, br  -> brp', hp, r0), tp)
            score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
        else:
            score_p = 0

        # (5) Compute a triple score based on interactions described by the bases of q {e_{p+1}, ..., e_{p+q}}. Eq. 22
        if self.q > 0:
            tq = E[:, -(self.r * self.q):].view(num_entities, self.r, self.q)
            h0_rq_tq = torch.einsum('brq, erq -> be', torch.einsum('br,  brq -> brq', h0, rq), tq)
            hq_r0_tq = torch.einsum('brq, erq -> be', torch.einsum('brq, br  -> brq', hq, r0), tq)
            hq_rq_t0 = torch.einsum('brq, er  -> be', hq * rq, t0)
            score_q = h0_rq_tq + hq_r0_tq - hq_rq_t0
        else:
            score_q = 0

        if self.p >= 2:
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_qq = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pq = 0
        return h0r0t0 + score_p + score_q + sigma_pp + sigma_qq + sigma_pq

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        r"""
        Kvsall training

        (1) Retrieve real-valued embedding vectors for heads and relations \mathbb{R}^d .
        (2) Construct head entity and relation embeddings according to Cl_{p,q}(\mathbb{R}^d) .
        (3) Perform Cl multiplication
        (4) Inner product of (3) and all entity embeddings

        forward_k_vs_with_explicit and this funcitons are identical
        Parameter
        ---------
        x: torch.LongTensor with (n,2) shape
        Returns
        -------
        torch.FloatTensor with (n, |E|) shape
        """
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)

        # (3) Extract all entity embeddings
        E = self.entity_embeddings.weight
        return self.k_vs_all_score(head_ent_emb, rel_ent_emb, E)

    def construct_batch_selected_cl_multivector(self, x: torch.FloatTensor, r: int, p: int, q: int) -> tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Split a batched, *k*-selected embedding tensor into Clifford components.

        A variant of :meth:`construct_cl_multivector` for tensors that have an
        extra *k* dimension (e.g. when scoring against *k* sampled targets).

        Parameters
        ----------
        x : torch.FloatTensor
            Shape ``(batch_size, k, d)``.
        r : int
            Scalar block size.
        p : int
            Number of positive-signature basis elements.
        q : int
            Number of negative-signature basis elements.

        Returns
        -------
        a0 : torch.FloatTensor
            Shape ``(batch_size, k, r)``.
        ap : torch.FloatTensor
            Shape ``(batch_size, k, r, p)``.
        aq : torch.FloatTensor
            Shape ``(batch_size, k, r, q)``.
        """
        batch_size, k, d = x.shape

        # (1) Take the first m columns for each k
        a0 = x[:, :, :r].view(batch_size, k, r)

        # (2) B_{n \times p}, C_{n \times q}: take the self.k * self.p columns after the k. column
        if p > 0:
            ap = x[:, :, r: r + (r * p)].view(batch_size, k, r, p)
        else:
            ap = torch.zeros((batch_size, k, r, p), device=self.device)
        if q > 0:
            # (3) B_{n \times p}, C_{n \times q}: take the last self.r * self.q .
            aq = x[:, :, -(r * q):].view(batch_size, k, r, q)
        else:
            aq = torch.zeros((batch_size, k, r, q), device=self.device)
        return a0, ap, aq

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor) -> torch.FloatTensor:
        """
        Parameter
        ---------
        x: torch.LongTensor with (n,2) shape

        target_entity_idx: torch.LongTensor with (n, k ) shape k denotes the selected number of examples.

        Returns
        -------
        torch.FloatTensor with (n, k) shape
        """
        # (1) Retrieve real-valued embedding vectors.
        # (b, d), (b, d)
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Construct multi-vector embeddings in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        # (b, m), (b, m, p), (b, m, q)
        h0, hp, hq = self.construct_cl_multivector(head_ent_emb, r=self.r, p=self.p, q=self.q)
        # (b, m), (b, m, p), (b, m, q)
        r0, rp, rq = self.construct_cl_multivector(rel_ent_emb, r=self.r, p=self.p, q=self.q)
        hp, hq, rp, rq = self.apply_coefficients(hp, hq, rp, rq)


        # (3) (b, k, d) Retrieve real-valued embedding vectors of selected entities.
        E = self.entity_embeddings(target_entity_idx)
        # (4) Construct multi-vector embeddings in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        # (b, k, m), (b, k, m, p), (b, k, m, q)
        t0, tp, tq = self.construct_batch_selected_cl_multivector(E, r=self.r, p=self.p, q=self.q)

        # (4) Batch vector matrix multiplications
        # Equivalent computations
        #                           h0*r0@t0.transpose(1,2)
        #                           torch.einsum('bm, bmk -> bk', h0 * r0, t0.transpose(1, 2))
        #                           torch.einsum('bm, bkm -> bk', h0 * r0, t0)
        h0r0t0 = torch.einsum('bm, bkm -> bk', h0 * r0, t0)

        # (5) Compute a triple score based on interactions described by the bases of p {e_1, ..., e_p}. Eq. 21
        if self.p > 0:
            raise NotImplementedError("Sample with p>0 for Keci not implemented")
            """
            # Second term in Eq.16
            hp_rp_t0 = torch.einsum('brp, br  -> b', hp * rp, t0)
            # Eq. 17
            # b=e
            h0_rp_tp = torch.einsum('brp, erp -> b', torch.einsum('br,  brp -> brp', h0, rp), tp)
            hp_r0_tp = torch.einsum('brp, erp -> b', torch.einsum('brp, br  -> brp', hp, r0), tp)
            score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
            """
        else:
            score_p = 0

        # (6) Compute a triple score based on interactions described by the bases of q {e_{p+1}, ..., e_{p+q}}. Eq. 22
        if self.q > 0:
            # \sum_{j=p+1}^{p+q} (h_j r_j t_0) : Third parth of the in Eq 16.
            # Equivalent computation
            # torch.einsum('bmq, bkm -> bk', hq*rq, t0) => (hq * rq).transpose(1,2) @ t0.transpose(1,2)
            hq_rq_t0 = torch.einsum('bmq, bkm -> bk', hq * rq, t0)

            # Eq. 18. Batch elementwise matrix matrix multiplication: bmq -> bkmq
            rq_tq=torch.einsum('bmq, bkmq -> bkmq', rq, tq)
            h0_rq_tq = torch.einsum('bm, bkmq  -> bk', h0, rq_tq)
            hq_tq=torch.einsum('bmq, bkmq -> bkmq',hq, tq)
            r0_hq_tq = torch.einsum('bm, bkmq  -> bk', r0, hq_tq)
            score_q = - hq_rq_t0 + (h0_rq_tq + r0_hq_tq)
        else:
            score_q = 0

        if self.p >= 2:
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_qq = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pq = 0
        return h0r0t0 + score_p + score_q + sigma_pp + sigma_qq + sigma_pq



    def score(self, h, r, t):
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq = self.construct_cl_multivector(h, r=self.r, p=self.p, q=self.q)
        r0, rp, rq = self.construct_cl_multivector(r, r=self.r, p=self.p, q=self.q)
        t0, tp, tq = self.construct_cl_multivector(t, r=self.r, p=self.p, q=self.q)

        if self.q > 0:
            self.q_coefficients = self.q_coefficients.to(h0.device, non_blocking=True)

        hp, hq, rp, rq = self.apply_coefficients(hp, hq, rp, rq)
        # (4) Compute a triple score based on interactions described by the basis 1. Eq. 20
        h0r0t0 = torch.einsum('br, br -> b', h0 * r0, t0)

        # (5) Compute a triple score based on interactions described by the bases of p {e_1, ..., e_p}. Eq. 21
        if self.p > 0:
            # Second term in Eq.16
            hp_rp_t0 = torch.einsum('brp, br  -> b', hp * rp, t0)
            # Eq. 17
            # b=e
            h0_rp_tp = torch.einsum('brp, erp -> b', torch.einsum('br,  brp -> brp', h0, rp), tp)
            hp_r0_tp = torch.einsum('brp, erp -> b', torch.einsum('brp, br  -> brp', hp, r0), tp)

            score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
        else:
            score_p = 0

        # (5) Compute a triple score based on interactions described by the bases of q {e_{p+1}, ..., e_{p+q}}. Eq. 22
        if self.q > 0:
            # Third item in Eq 16.
            hq_rq_t0 = torch.einsum('brq, br  -> b', hq * rq, t0)
            # Eq. 18.
            h0_rq_tq = torch.einsum('br, brq  -> b', h0, rq * tq)
            r0_hq_tq = torch.einsum('br, brq  -> b', r0, hq * tq)
            score_q = - hq_rq_t0 + (h0_rq_tq + r0_hq_tq)
        else:
            score_q = 0

        if self.p >= 2:
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_qq = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pq = 0
        return h0r0t0 + score_p + score_q + sigma_pp + sigma_qq + sigma_pq

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        """

        Parameter
        ---------
        x: torch.LongTensor with (n,3) shape

        Returns
        -------
        torch.FloatTensor with (n) shape
        """
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq = self.construct_cl_multivector(head_ent_emb, r=self.r, p=self.p, q=self.q)
        r0, rp, rq = self.construct_cl_multivector(rel_ent_emb, r=self.r, p=self.p, q=self.q)
        t0, tp, tq = self.construct_cl_multivector(tail_ent_emb, r=self.r, p=self.p, q=self.q)
        hp, hq, rp, rq = self.apply_coefficients( hp, hq, rp, rq)
        # (4) Compute a triple score based on interactions described by the basis 1. Eq. 20
        h0r0t0 = torch.einsum('br, br -> b', h0 * r0, t0)

        # (5) Compute a triple score based on interactions described by the bases of p {e_1, ..., e_p}. Eq. 21
        if self.p > 0:
            # Second term in Eq.16
            hp_rp_t0 = torch.einsum('brp, br  -> b', hp * rp, t0)
            # Eq. 17
            # b=e
            h0_rp_tp = torch.einsum('brp, erp -> b', torch.einsum('br,  brp -> brp', h0, rp), tp)
            hp_r0_tp = torch.einsum('brp, erp -> b', torch.einsum('brp, br  -> brp', hp, r0), tp)

            score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
        else:
            score_p = 0

        # (5) Compute a triple score based on interactions described by the bases of q {e_{p+1}, ..., e_{p+q}}. Eq. 22
        if self.q > 0:
            # Third item in Eq 16.
            hq_rq_t0 = torch.einsum('brq, br  -> b', hq * rq, t0)
            # Eq. 18.
            h0_rq_tq = torch.einsum('br, brq  -> b', h0, rq * tq)
            r0_hq_tq = torch.einsum('br, brq  -> b', r0, hq * tq)
            score_q = - hq_rq_t0 + (h0_rq_tq + r0_hq_tq)
        else:
            score_q = 0

        if self.p >= 2:
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).squeeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).squeeze(-1)
        else:
            sigma_qq = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).squeeze(-1)
        else:
            sigma_pq = 0
        return h0r0t0 + score_p + score_q + sigma_pp + sigma_qq + sigma_pq


class KeciTransformer(Keci):
    """
    Keci with Transformer architecture.

    Concatenates h0, hp, hq, r0, rp, rq into a single embedding vector and processes through transformer.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'KeciTransformer'

        # Boolean flag to include clifford multiplication in embedding
        self.use_clifford_mul = self.args.get("use_clifford_mul", False)

        # Input dimension:
        # Original: h0 (r) + hp (r*p) + hq (r*q) + r0 (r) + rp (r*p) + rq (r*q) = 2 * embedding_dim
        # Clifford multiplication: sigma_0 (r) + sigma_p (r*p) + sigma_q (r*q) + sigma_pp (r*(p*(p-1)/2)) + sigma_qq (r*(q*(q-1)/2)) + sigma_pq (r*p*q)
        original_dim = 2 * self.embedding_dim
        if self.use_clifford_mul:
            clifford_dim = self.r + self.r * self.p + self.r * self.q + \
                           self.r * int((self.p * (self.p - 1)) / 2) + \
                           self.r * int((self.q * (self.q - 1)) / 2) + \
                           self.r * self.p * self.q
            self.input_dim = original_dim + clifford_dim
        else:
            self.input_dim = original_dim

        # Transformer configuration
        n_layer = self.args.get("n_layer", 4)
        dropout = self.args.get("dropout", 0.0)
        bias = self.args.get("bias", False)

        # Calculate valid n_head: must divide input_dim evenly
        # Use user-specified n_head if valid, otherwise find largest valid divisor <= 4
        requested_n_head = self.args.get("n_head", 4)
        if self.input_dim % requested_n_head == 0:
            n_head = requested_n_head
        else:
            # Find largest divisor of input_dim that is <= requested_n_head and >= 1
            n_head = 1
            for h in range(1, requested_n_head + 1):
                if self.input_dim % h == 0:
                    n_head = h
        # Sequence length is 1 (single embedding vector treated as one token)
        self.seq_len = 1

        # Transformer components
        self.transformer = torch.nn.ModuleDict(dict(
            wpe=torch.nn.Embedding(self.seq_len, self.input_dim),  # positional embeddings
            drop=torch.nn.Dropout(dropout),
            h=torch.nn.ModuleList([TransformerBlock(self.input_dim, n_head, dropout, bias) for _ in range(n_layer)]),
            ln_f=torch.nn.LayerNorm(self.input_dim, elementwise_affine=not bias),
        ))
        # Output projection: maps to number of entities for scoring
        self.lm_head = torch.nn.Linear(self.input_dim, self.num_entities, bias=False)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Kvsall training

        Parameter
        ---------
        x: torch.LongTensor with (n,2) shape

        Returns
        -------
        torch.FloatTensor with (n, |E|) shape
        """
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)

        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq = self.construct_cl_multivector(head_ent_emb, r=self.r, p=self.p, q=self.q)
        r0, rp, rq = self.construct_cl_multivector(rel_ent_emb, r=self.r, p=self.p, q=self.q)

        # (3) Flatten base embeddings
        # h0: (n, r), hp: (n, r, p), hq: (n, r, q), r0: (n, r), rp: (n, r, p), rq: (n, r, q)
        batch_size = h0.shape[0]

        hp_flat = hp.view(batch_size, -1)  # (n, r*p)
        hq_flat = hq.view(batch_size, -1)  # (n, r*q)
        rp_flat = rp.view(batch_size, -1)  # (n, r*p)
        rq_flat = rq.view(batch_size, -1)  # (n, r*q)

        if self.use_clifford_mul:
            # Compute clifford multiplication
            sigma_0, sigma_p, sigma_q, sigma_pp, sigma_qq, sigma_pq = self.clifford_multiplication(h0, hp, hq, r0, rp, rq)

            # Flatten clifford multiplication results
            sigma_p_flat = sigma_p.view(batch_size, -1)  # (n, r*p)
            sigma_q_flat = sigma_q.view(batch_size, -1)  # (n, r*q)
            sigma_pp_flat = sigma_pp.view(batch_size, -1)  # (n, r*p*(p-1)/2)
            sigma_qq_flat = sigma_qq.view(batch_size, -1)  # (n, r*q*(q-1)/2)
            sigma_pq_flat = sigma_pq.view(batch_size, -1)  # (n, r*p*q)

            # Concatenate all embeddings including clifford multiplication
            x_emb = torch.cat([h0, hp_flat, hq_flat, r0, rp_flat, rq_flat,
                              sigma_0, sigma_p_flat, sigma_q_flat, sigma_pp_flat, sigma_qq_flat, sigma_pq_flat], dim=1)
        else:
            # Concatenate base embeddings only
            x_emb = torch.cat([h0, hp_flat, hq_flat, r0, rp_flat, rq_flat], dim=1)

        # (4) Reshape for transformer: (n, 1, input_dim)
        x_emb = x_emb.unsqueeze(1)

        # (5) Apply transformer
        device = x_emb.device
        pos = torch.arange(0, self.seq_len, dtype=torch.long, device=device)

        pos_emb = self.transformer.wpe(pos)
        x_emb = self.transformer.drop(x_emb + pos_emb)

        for block in self.transformer.h:
            x_emb = block(x_emb)
        x_emb = self.transformer.ln_f(x_emb)
        logits = self.lm_head(x_emb)  # (n, 1, num_entities)

        # (6) Squeeze and return logits directly as scores
        return logits.squeeze(1)  # (n, num_entities)


class TransformerBlock(torch.nn.Module):
    """A single transformer block with self-attention and MLP."""

    def __init__(self, n_embd, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(n_embd, elementwise_affine=not bias)
        self.attn = TransformerSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = torch.nn.LayerNorm(n_embd, elementwise_affine=not bias)
        self.mlp = TransformerMLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerSelfAttention(torch.nn.Module):
    """Multi-head self-attention for the Keci Transformer."""

    def __init__(self, n_embd, n_head, dropout=0.0, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = torch.nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Non-causal attention (bidirectional)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerMLP(torch.nn.Module):
    """MLP for the Keci Transformer."""

    def __init__(self, n_embd, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc = torch.nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CKeci(Keci):
    " Without learning dimension scaling"

    def __init__(self, args):
        super().__init__(args)
        self.name = 'CKeci'
        self.requires_grad_for_interactions = False
        logger.info(f'r:{self.r}\t p:{self.p}\t q:{self.q}')
        if self.p > 0:
            self.p_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.p,_freeze=self.requires_grad_for_interactions)
            torch.nn.init.ones_(self.p_coefficients.weight)
        if self.q > 0:
            self.q_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.q,_freeze=self.requires_grad_for_interactions)
            torch.nn.init.ones_(self.q_coefficients.weight)


class DeCaL(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'DeCaL'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.p = self.args.get("p", 0)
        self.q = self.args.get("q", 0)
        self.r = self.args.get("r", 0)
        self.re = int(self.embedding_dim / (self.r + self.p + self.q + 1))

        # Initialize parameters for dimension scaling
        if self.p > 0:
            self.p_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.p)
            torch.nn.init.zeros_(self.p_coefficients.weight)
        if self.q > 0:
            self.q_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.q)
            torch.nn.init.zeros_(self.q_coefficients.weight)
        if self.r > 0:
            self.r_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.r)
            torch.nn.init.zeros_(self.r_coefficients.weight)

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        """

        Parameter
        ---------
        x: torch.LongTensor with (n, ) shape

        Returns
        -------
        torch.FloatTensor with (n) shape
        """
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Construct multi-vector in Cl_{p,q,r} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq, hk = self.construct_cl_multivector(head_ent_emb, re=self.re, p=self.p, q=self.q, r=self.r)
        r0, rp, rq, rk = self.construct_cl_multivector(rel_ent_emb, re=self.re, p=self.p, q=self.q, r=self.r)
        t0, tp, tq, tk = self.construct_cl_multivector(tail_ent_emb, re=self.re, p=self.p, q=self.q, r=self.r)

        # h0, hp, hq, hk, h0, rp, rq, rk = self.apply_coefficients(h0, hp, hq, hk, h0, rp, rq,rk)

        # (4) Compute a triple score based on interactions described by the basis 1.
        h0r0t0 = torch.einsum('br, br -> b', h0 * r0, t0)

        # (5) Compute a triple score based on interactions described by the bases of p {e_1, ..., e_p}.
        if self.p > 0:
            # Second term in Eq.16
            hp_rp_t0 = torch.einsum('brp, br  -> b', hp * rp, t0)
            # Eq. 17
            # b=e
            h0_rp_tp = torch.einsum('brp, erp -> b', torch.einsum('br,  brp -> brp', h0, rp), tp)
            hp_r0_tp = torch.einsum('brp, erp -> b', torch.einsum('brp, br  -> brp', hp, r0), tp)

            score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
        else:
            score_p = 0

        # (5) Compute a triple score based on interactions described by the bases of q {e_{p+1}, ..., e_{p+q}}. Eq. 22
        if self.q > 0:
            # Third item in Eq 16.
            hq_rq_t0 = torch.einsum('brq, br  -> b', hq * rq, t0)
            # Eq. 18.
            h0_rq_tq = torch.einsum('br, brq  -> b', h0, rq * tq)
            r0_hq_tq = torch.einsum('br, brq  -> b', r0, hq * tq)
            score_q = - hq_rq_t0 + (h0_rq_tq + r0_hq_tq)
        else:
            score_q = 0

        if self.r > 0:
            # Eq. 18.
            h0_rk_tk = torch.einsum('br, brk  -> b', h0, rk * tk)
            r0_hk_tk = torch.einsum('br, brk  -> b', r0, hk * tk)
            score_r = (h0_rk_tk + r0_hk_tk)
        else:
            score_r = 0

        if self.p >= 2:
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).squeeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).squeeze(-1)
        else:
            sigma_qq = 0

        if self.r >= 2:
            sigma_rr = torch.sum(self.compute_sigma_rr(hk, rk), dim=[1, 2]).squeeze(-1)
        else:
            sigma_rr = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).squeeze(-1)
        else:
            sigma_pq = 0

        if self.p >= 2 and self.r >= 2:
            sigma_pr = torch.sum(self.compute_sigma_pr(hp=hp, hk=hk, rp=rp, rk=rk), dim=[1, 2, 3]).squeeze(-1)
        else:
            sigma_pr = 0
        if self.q >= 2 and self.r >= 2:
            sigma_qr = torch.sum(self.compute_sigma_qr(hq=hq, hk=hk, rq=rq, rk=rk), dim=[1, 2, 3]).squeeze(-1)
        else:
            sigma_qr = 0

        return h0r0t0 + score_p + score_q + score_r + sigma_pp + sigma_qq + sigma_rr + sigma_pq + sigma_qr + sigma_pr

    def cl_pqr(self, a:torch.tensor)->torch.tensor:

        ''' Input: tensor(batch_size, emb_dim) ---> output: tensor with 1+p+q+r components with size (batch_size, emb_dim/(1+p+q+r)) each.

        1) takes a tensor of size (batch_size, emb_dim), split it into 1 + p + q +r components, hence 1+p+q+r must be a divisor
        of the emb_dim.
        2) Return a list of the 1+p+q+r components vectors, each are tensors of size (batch_size, emb_dim/(1+p+q+r)) '''

        # num1 = 2**(p+q+r) #total number of vector in cl_pqr then after choose the first p+q+r+1 vectors
        num1 = 1 + self.p + self.q + self.r
        a1 = torch.hsplit(a, num1)

        return torch.stack(a1)

    def compute_sigmas_single(self, list_h_emb, list_r_emb, list_t_emb):

        r'''here we compute all the sums with no others vectors interaction taken with the scalar product with t, that is,

        .. math::

             s0 = h_0r_0t_0
             s1 = \sum_{i=1}^{p}h_ir_it_0
             s2 = \sum_{j=p+1}^{p+q}h_jr_jt_0
             s3 = \sum_{i=1}^{q}(h_0r_it_i + h_ir_0t_i)
             s4 = \sum_{i=p+1}^{p+q}(h_0r_it_i + h_ir_0t_i)
             s5 = \sum_{i=p+q+1}^{p+q+r}(h_0r_it_i + h_ir_0t_i)

        and return:

        .. math::

            sigma_0t = \sigma_0 \cdot t_0 = s0 + s1 -s2
            s3, s4 and s5


        '''

        p = self.p
        q = self.q
        r = self.r

        h_0 = list_h_emb[0]  # h_i = list_h_emb[i] similarly for r and t
        r_0 = list_r_emb[0]
        t_0 = list_t_emb[0]

        s0 = (h_0 * r_0 * t_0).sum(dim=1)

        s1 = (t_0 * (list_h_emb[1:p + 1] * list_r_emb[1:p + 1])).sum(dim=[-1, 0])

        s2 = (t_0 * (list_h_emb[p + 1:p + q + 1] * list_r_emb[p + 1:p + q + 1])).sum(dim=[-1, 0])

        s3 = (h_0 * (list_r_emb[1:p + 1] * list_t_emb[1:p + 1]) + r_0 * (
                    list_h_emb[1:p + 1] * list_t_emb[1:p + 1])).sum(dim=[-1, 0])

        s4 = (h_0 * (list_r_emb[p + 1:p + q + 1] * list_t_emb[p + 1:p + q + 1]) + r_0 * (
                    list_h_emb[p + 1:p + q + 1] * list_t_emb[p + 1:p + q + 1])).sum(dim=[-1, 0])

        s5 = (h_0 * (list_r_emb[p + q + 1:p + q + r + 1] * list_t_emb[p + q + 1:p + q + r + 1]) + r_0 * (
                    list_h_emb[p + q + 1:p + q + r + 1] * list_t_emb[p + q + 1:p + q + r + 1])).sum(dim=[-1, 0])

        sigma_0t = s0 + s1 - s2

        return sigma_0t, s3, s4, s5

    def compute_sigmas_multivect(self, list_h_emb, list_r_emb):

        r'''Here we compute and return all the sums with vectors interaction for the same and different bases.

           For same bases vectors interaction we have

           .. math::

                \sigma_pp = \sum_{i=1}^{p-1}\sum_{i'=i+1}^{p}(h_ir_{i'}-h_{i'}r_i) (models the interactions between e_i and e_i' for 1 <= i, i' <= p)
                \sigma_qq = \sum_{j=p+1}^{p+q-1}\sum_{j'=j+1}^{p+q}(h_jr_{j'}-h_{j'} (models the interactions between e_j and e_j' for p+1 <= j, j' <= p+q)
                \sigma_rr = \sum_{k=p+q+1}^{p+q+r-1}\sum_{k'=k+1}^{p}(h_kr_{k'}-h_{k'}r_k) (models the interactions between e_k and e_k' for p+q+1 <= k, k' <= p+q+r)

           For different base vector interactions, we have

            .. math::

                \sigma_pq = \sum_{i=1}^{p}\sum_{j=p+1}^{p+q}(h_ir_j - h_jr_i) (interactionsn between e_i and e_j for 1<=i <=p and p+1<= j <= p+q)
                \sigma_pr = \sum_{i=1}^{p}\sum_{k=p+q+1}^{p+q+r}(h_ir_k - h_kr_i) (interactionsn between e_i and e_k for 1<=i <=p and p+q+1<= k <= p+q+r)
                \sigma_qr = \sum_{j=p+1}^{p+q}\sum_{j=p+q+1}^{p+q+r}(h_jr_k - h_kr_j) (interactionsn between e_j and e_k for p+1 <= j <=p+q and p+q+1<= j <= p+q+r)

           '''

        p = self.p
        q = self.q
        r = self.r

        if p > 0:
            indices_i = torch.arange(1, p)
            sigma_pp = ((list_h_emb[indices_i] * list_r_emb[indices_i + 1].sum(dim=0)) - (
                        list_h_emb[indices_i + 1].sum(dim=0) * list_r_emb[indices_i])).sum(dim=[-1, 0])
        else:
            indices_i = []
            sigma_pp = 0
        if q > 0:
            indices_j = torch.arange(p + 1, p + q)
            sigma_qq = ((list_h_emb[indices_j] * list_r_emb[indices_j + 1].sum(dim=0)) - (
                        list_h_emb[indices_j + 1].sum(dim=0) * list_r_emb[indices_j])).sum(dim=[-1, 0])
        else:
            indices_j = []
            sigma_qq = 0
        if r > 0:
            indices_k = torch.arange(p + q + 1, p + q + r)
            sigma_rr = ((list_h_emb[indices_k] * list_r_emb[indices_k + 1].sum(dim=0)) - (
                        list_h_emb[indices_k + 1].sum(dim=0) * list_r_emb[indices_k])).sum(dim=[-1, 0])
        else:
            indices_k = []
            sigma_rr = 0

        sigma_pq = ((list_h_emb[indices_i] * list_r_emb[indices_j].sum(dim=0)) - (
                    list_h_emb[indices_j].sum(dim=0) * list_r_emb[indices_i])).sum(dim=[-1, 0])
        sigma_pr = ((list_h_emb[indices_i] * list_r_emb[indices_k].sum(dim=0)) - (
                    list_h_emb[indices_k].sum(dim=0) * list_r_emb[indices_i])).sum(dim=[-1, 0])
        sigma_qr = ((list_h_emb[indices_j] * list_r_emb[indices_k].sum(dim=0)) - (
                    list_h_emb[indices_k].sum(dim=0) * list_r_emb[indices_j])).sum(dim=[-1, 0])

        return sigma_pp, sigma_qq, sigma_rr, sigma_pq, sigma_pr, sigma_qr

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:

        r"""
            Kvsall training

            (1) Retrieve real-valued embedding vectors for heads and relations
            (2) Construct head entity and relation embeddings according to Cl_{p,q, r}(\mathbb{R}^d) .
            (3) Perform Cl multiplication
            (4) Inner product of (3) and all entity embeddings

            forward_k_vs_with_explicit and this funcitons are identical
            Parameter
            ---------
            x: torch.LongTensor with (n, ) shape
            Returns
            -------
            torch.FloatTensor with (n, |E|) shape
            """
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq, hk = self.construct_cl_multivector(head_ent_emb, re=self.re, p=self.p, q=self.q, r=self.r)
        r0, rp, rq, rk = self.construct_cl_multivector(rel_ent_emb, re=self.re, p=self.p, q=self.q, r=self.r)

        h0, hp, hq, hk, h0, rp, rq, rk = self.apply_coefficients(h0, hp, hq, hk, h0, rp, rq, rk)
        # (3) Extract all entity embeddings
        E = self.entity_embeddings.weight
        # (3.1) Extract real part
        t0 = E[:, :self.re]
        # (4) Compute a triple score based on interactions described by the basis 1.
        h0r0t0 = torch.einsum('br,er->be', h0 * r0, t0)

        # (5) Compute a triple score based on interactions described by the bases of p {e_1, ..., e_p}.
        if self.p > 0:
            tp = E[:, self.re: self.re + (self.re * self.p)].view(self.num_entities, self.re, self.p)
            hp_rp_t0 = torch.einsum('brp, er  -> be', hp * rp, t0)
            h0_rp_tp = torch.einsum('brp, erp -> be', torch.einsum('br,  brp -> brp', h0, rp), tp)
            hp_r0_tp = torch.einsum('brp, erp -> be', torch.einsum('brp, br  -> brp', hp, r0), tp)
            score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
        else:
            score_p = 0

        # (5) Compute a triple score based on interactions described by the bases of q {e_{p+1}, ..., e_{p+q}}.
        if self.q > 0:
            num = self.re + (self.re * self.p)
            tq = E[:, num:num + (self.re * self.q)].view(self.num_entities, self.re, self.q)
            h0_rq_tq = torch.einsum('brq, erq -> be', torch.einsum('br,  brq -> brq', h0, rq), tq)
            hq_r0_tq = torch.einsum('brq, erq -> be', torch.einsum('brq, br  -> brq', hq, r0), tq)
            hq_rq_t0 = torch.einsum('brq, er  -> be', hq * rq, t0)
            score_q = h0_rq_tq + hq_r0_tq - hq_rq_t0
        else:
            score_q = 0

        # (6) Compute a triple score based on interactions described by the bases of q {e_{p+q+1}, ..., e_{p+q+r}}.
        if self.r > 0:
            tk = E[:, -(self.re * self.r):].view(self.num_entities, self.re, self.r)
            h0_rk_tk = torch.einsum('brk, erk -> be', torch.einsum('br,  brk -> brk', h0, rk), tk)
            hk_r0_tk = torch.einsum('brk, erk -> be', torch.einsum('brk, br  -> brk', hk, r0), tk)
            # hq_rq_t0 = torch.einsum('brq, er  -> be', hq * rq, t0)
            score_r = h0_rk_tk + hk_r0_tk
        else:
            score_r = 0

        if self.p >= 2:
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_qq = 0

        if self.r >= 2:
            sigma_rr = torch.sum(self.compute_sigma_rr(hk, rk), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_rr = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pq = 0
        if self.p >= 2 and self.r >= 2:
            sigma_pr = torch.sum(self.compute_sigma_pr(hp=hp, hk=hk, rp=rp, rk=rk), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pr = 0
        if self.q >= 2 and self.r >= 2:
            sigma_qr = torch.sum(self.compute_sigma_qr(hq=hq, hk=hk, rq=rq, rk=rk), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_qr = 0

        return h0r0t0 + score_p + score_q + score_r + sigma_pp + sigma_qq + sigma_rr + sigma_pq + sigma_pr + sigma_qr

    def apply_coefficients(self, h0, hp, hq, hk, r0, rp, rq, rk):
        """ Multiplying a base vector with its scalar coefficient """
        if self.p > 0:
            hp = hp * self.p_coefficients.weight
            rp = rp * self.p_coefficients.weight
        if self.q > 0:
            hq = hq * self.q_coefficients.weight
            rq = rq * self.q_coefficients.weight
        if self.r > 0:
            hk = hk * self.r_coefficients.weight
            rk = rk * self.r_coefficients.weight
        return h0, hp, hq, hk, r0, rp, rq, rk

    def construct_cl_multivector(self, x: torch.FloatTensor, re: int, p: int, q: int, r: int) -> tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        r"""
        Construct a batch of multivectors Cl_{p,q,r}(\mathbb{R}^d)

        Parameter
        ---------
        x: torch.FloatTensor with (n,d) shape

        Returns
        -------
        a0: torch.FloatTensor
        ap: torch.FloatTensor
        aq: torch.FloatTensor
        ar: torch.FloatTensor
        """
        batch_size, d = x.shape
        # (1) A_{n \times k}: take the first k columns
        a0 = x[:, :re].view(batch_size, re)
        # (2) B_{n \times p}, C_{n \times q}: take the self.k * self.p columns after the k. column
        if p > 0:
            ap = x[:, re: re + (re * p)].view(batch_size, re, p)
        else:
            ap = torch.zeros((batch_size, re, p), device=self.device)
        if q > 0:
            # (3) B_{n \times p}, C_{n \times q}: take the last self.r * self.q .
            aq = x[:, re + (re * p):re + (re * p) + (re * q):].view(batch_size, re, q)
        else:
            aq = torch.zeros((batch_size, re, q), device=self.device)
        if r > 0:
            # (3) B_{n \times p}, C_{n \times q}: take the last self.r * self.q .
            ar = x[:, -(re * r):].view(batch_size, re, r)
        else:
            ar = torch.zeros((batch_size, re, r), device=self.device)
        return a0, ap, aq, ar

    def compute_sigma_pp(self, hp, rp):
        r"""
        Compute
        .. math::

            \sigma_{p,p}^* = \sum_{i=1}^{p-1}\sum_{i'=i+1}^{p}(x_iy_{i'}-x_{i'}y_i)

        \sigma_{pp} captures the interactions between along p bases
        For instance, let p e_1, e_2, e_3, we compute interactions between e_1 e_2, e_1 e_3 , and e_2 e_3
        This can be implemented with a nested two for loops

                        results = []
                        for i in range(p - 1):
                            for k in range(i + 1, p):
                                results.append(hp[:, :, i] * rp[:, :, k] - hp[:, :, k] * rp[:, :, i])
                        sigma_pp = torch.stack(results, dim=2)
                        assert sigma_pp.shape == (b, r, int((p * (p - 1)) / 2))

        Yet, this computation would be quite inefficient. Instead, we compute interactions along all p,
        e.g., e1e1, e1e2, e1e3,
              e2e1, e2e2, e2e3,
              e3e1, e3e2, e3e3
        Then select the triangular matrix without diagonals: e1e2, e1e3, e2e3.
        """
        # Compute indexes for the upper triangle of p by p matrix
        indices = torch.triu_indices(self.p, self.p, offset=1)
        # Compute p by p operations
        sigma_pp = torch.einsum('nrp,nrx->nrpx', hp, rp) - torch.einsum('nrx,nrp->nrpx', hp, rp)
        sigma_pp = sigma_pp[:, :, indices[0], indices[1]]
        return sigma_pp

    def compute_sigma_qq(self, hq, rq):
        r"""
        Compute

        .. math::

            \sigma_{q,q}^* = \sum_{j=p+1}^{p+q-1}\sum_{j'=j+1}^{p+q}(x_jy_{j'}-x_{j'}y_j) Eq. 16

        sigma_{q} captures the interactions between along q bases
        For instance, let q e_1, e_2, e_3, we compute interactions between e_1 e_2, e_1 e_3 , and e_2 e_3
        This can be implemented with a nested two for loops

                        results = []
                        for j in range(q - 1):
                            for k in range(j + 1, q):
                                results.append(hq[:, :, j] * rq[:, :, k] - hq[:, :, k] * rq[:, :, j])
                        sigma_qq = torch.stack(results, dim=2)
                        assert sigma_qq.shape == (b, r, int((q * (q - 1)) / 2))

        Yet, this computation would be quite inefficient. Instead, we compute interactions along all p,
        e.g., e1e1, e1e2, e1e3,
              e2e1, e2e2, e2e3,
              e3e1, e3e2, e3e3
        Then select the triangular matrix without diagonals: e1e2, e1e3, e2e3.
        """
        # Compute indexes for the upper triangle of p by p matrix
        if self.q > 1:
            indices = torch.triu_indices(self.q, self.q, offset=1)
            # Compute p by p operations
            sigma_qq = torch.einsum('nrp,nrx->nrpx', hq, rq) - torch.einsum('nrx,nrp->nrpx', hq, rq)
            sigma_qq = sigma_qq[:, :, indices[0], indices[1]]

        else:
            sigma_qq = torch.zeros((len(hq), self.re, int((self.q * (self.q - 1)) / 2)))

        return sigma_qq

    def compute_sigma_rr(self, hk, rk):
        r"""
        .. math::

            \sigma_{r,r}^* = \sum_{k=p+q+1}^{p+q+r-1}\sum_{k'=k+1}^{p}(x_ky_{k'}-x_{k'}y_k)

        """
        # Compute indexes for the upper triangle of p by p matrix
        if self.r > 1:
            indices = torch.triu_indices(self.r, self.r, offset=1)
            # Compute p by p operations
            sigma_rr = torch.einsum('nrp,nrx->nrpx', hk, rk) - torch.einsum('nrx,nrp->nrpx', hk, rk)
            sigma_rr = sigma_rr[:, :, indices[0], indices[1]]
        else:
            sigma_rr = torch.zeros((len(hk), self.re, int((self.r * (self.r - 1)) / 2)))

        return sigma_rr

    def compute_sigma_pq(self, *, hp, hq, rp, rq):
        r"""
        Compute

        .. math::

            \sum_{i=1}^{p} \sum_{j=p+1}^{p+q} (h_i r_j - h_j r_i) e_i e_j

        results = []
        sigma_pq = torch.zeros(b, r, p, q)
        for i in range(p):
            for j in range(q):
                sigma_pq[:, :, i, j] = hp[:, :, i] * rq[:, :, j] - hq[:, :, j] * rp[:, :, i]
        print(sigma_pq.shape)

        """
        sigma_pq = torch.einsum('nrp,nrq->nrpq', hp, rq) - torch.einsum('nrq,nrp->nrpq', hq, rp)
        assert sigma_pq.shape[1:] == (self.re, self.p, self.q)
        return sigma_pq

    def compute_sigma_pr(self, *, hp, hk, rp, rk):
        r"""
        Compute

        .. math::

            \sum_{i=1}^{p} \sum_{j=p+1}^{p+q} (h_i r_j - h_j r_i) e_i e_j

        results = []
        sigma_pq = torch.zeros(b, r, p, q)
        for i in range(p):
            for j in range(q):
                sigma_pq[:, :, i, j] = hp[:, :, i] * rq[:, :, j] - hq[:, :, j] * rp[:, :, i]
        print(sigma_pq.shape)

        """
        sigma_pr = torch.einsum('nrp,nrk->nrpk', hp, rk) - torch.einsum('nrk,nrp->nrpk', hk, rp)
        assert sigma_pr.shape[1:] == (self.re, self.p, self.r)
        return sigma_pr

    def compute_sigma_qr(self, *, hq, hk, rq, rk):
        r"""
        .. math::

            \sum_{i=1}^{p} \sum_{j=p+1}^{p+q} (h_i r_j - h_j r_i) e_i e_j

        results = []
        sigma_pq = torch.zeros(b, r, p, q)
        for i in range(p):
            for j in range(q):
                sigma_pq[:, :, i, j] = hp[:, :, i] * rq[:, :, j] - hq[:, :, j] * rp[:, :, i]
        print(sigma_pq.shape)

        """
        sigma_qr = torch.einsum('nrq,nrk->nrqk', hq, rk) - torch.einsum('nrk,nrq->nrqk', hk, rq)
        assert sigma_qr.shape[1:] == (self.re, self.q, self.r)
        return sigma_qr


# ─────────────────────────────────────────────────────────────────────────────
#  FullDeCaL – Full Clifford Knowledge Graph Embedding
#
#  Two modes controlled by  --auto_signature:
#
#  FIXED MODE  (default, --auto_signature not set)
#  ------------------------------------------------
#  User provides --p / --q / --r.  The signature η is fixed to
#      +1  (first p generators),  −1  (next q),  0  (last r).
#  The coefficient table is precomputed once at __init__ as a frozen buffer
#  → zero overhead per forward pass.
#  Use this when you already know the right algebra for your dataset.
#
#  AUTO MODE  (--auto_signature)
#  ------------------------------
#  User provides only --embedding_dim.  n is derived automatically as
#      n = floor(log2(embedding_dim) / 2)
#  giving a balanced split between number of blades (d=2^n) and per-blade
#  width (re = embedding_dim // d).  Examples:
#      dim=64  → n=3, d=8,  re=8
#      dim=128 → n=3, d=8,  re=16
#      dim=256 → n=4, d=16, re=16
#  η_1…η_n are free nn.Parameters initialized randomly to ±1 + small noise,
#  so that gradients are non-zero from the start and convergence is
#  dataset-dependent.  The coefficient table is rebuilt each forward pass
#  (fully differentiable).  After training, `model.learned_signature()`
#  reveals which Cl_{p,q,r} the model converged to — no search needed.
# ─────────────────────────────────────────────────────────────────────────────

def _build_sign_table(n: int):
    """
    Precompute the reordering-sign table σ(I,J) and structural index tables
    for the n-generator Clifford algebra (2^n blades, bitmask ordering).

    Returns
    -------
    sign_table        : FloatTensor (d, d)  – σ(I,J) ∈ {+1, −1}
    K_table           : LongTensor  (d, d)  – output blade  K = I △ J
    intersection_table: LongTensor  (d, d)  – I ∩ J  (bitmask)
    bits              : FloatTensor (d, n)  – bits[K, k] = (K >> k) & 1
    """
    d = 1 << n
    sign_table         = torch.zeros(d, d, dtype=torch.float32)
    K_table            = torch.zeros(d, d, dtype=torch.long)
    intersection_table = torch.zeros(d, d, dtype=torch.long)

    for I in range(d):
        for J in range(d):
            swaps = 0
            for i in range(n):
                if I & (1 << i):
                    swaps += bin(J & ((1 << i) - 1)).count('1')
            sign_table[I, J]         = -1.0 if swaps % 2 else 1.0
            K_table[I, J]            = I ^ J
            intersection_table[I, J] = I & J

    idx  = torch.arange(d, dtype=torch.long)
    bits = ((idx.unsqueeze(1) >> torch.arange(n, dtype=torch.long).unsqueeze(0)) & 1).float()
    return sign_table, K_table, intersection_table, bits


def _auto_n_from_dim(embedding_dim: int, min_re: int = 4) -> int:
    """Derive n for FullDeCaL auto mode from embedding_dim.

    Maximises n subject to ``re = embedding_dim / 2^n >= min_re``.
    This gives the largest signature search space (any Cl_{p,q,r} with
    p+q+r <= n is reachable) while keeping per-blade width >= min_re.

    Examples with min_re=4::

        dim=32  → n=3, d=8,  re=4
        dim=64  → n=4, d=16, re=4
        dim=128 → n=5, d=32, re=4
        dim=256 → n=6, d=64, re=4
    """
    import math
    n = max(1, int(math.log2(embedding_dim)) - int(math.log2(min_re)))
    while embedding_dim % (1 << n) != 0:
        n -= 1
    return n


class FullDeCaL(BaseKGE):
    """Full Clifford KGE – fixed or auto Clifford signature.

    Scoring function
    ----------------
        f(h, r, t) = Σ_{I,J}  h_I r_J  σ(I,J)  [∏_{k∈I∩J} η_k]  t_{I△J}

    Two modes
    ---------
    **Fixed mode** (default)::

        dicee ... --model FullDeCaL --p 1 --q 1 --r 0 --embedding_dim 64

        Signature η is frozen from (p,q,r).  Coefficient table precomputed
        once at init → fastest possible forward pass.

    **Auto mode**::

        dicee ... --model FullDeCaL --auto_signature --embedding_dim 64

        Only embedding_dim is needed.  n is derived automatically as the
        largest value satisfying re = embedding_dim / 2^n >= 4
        (e.g. dim=64 → n=4, d=16, re=4; dim=128 → n=5, d=32, re=4).
        η_1…η_n are free nn.Parameters.  Because η_k = tanh(η̃_k), any
        generator can converge to +1, −1, or 0 (null), so the effective
        algebra is Cl_{p,q,r} for any p+q+r <= n — a fully continuous
        search over the signature space.

    Embedding layout
    ----------------
    embedding_dim = 2^n · re.  Embeddings are split into 2^n blocks of size
    re, one per blade in bitmask order (0=scalar, 1=e₁, 2=e₂, 3=e₁₂, …).
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = 'FullDeCaL'

        self._auto = bool(self.args.get('auto_signature', False))

        if self._auto:
            # ── Auto mode ────────────────────────────────────────────────
            # n is the largest value s.t. re = embedding_dim / 2^n >= 4.
            # This maximises the signature search space: any Cl_{p,q,r}
            # with p+q+r <= n is reachable because each η_k = tanh(η̃_k)
            # can freely converge to +1 (positive), −1 (negative), or 0
            # (null), making "p+q+r <= n" a continuous relaxation.
            #
            # Initialization: N(0, 1.5) gives ~35% of generators near 0
            # (easily null), ~45% softly typed, ~20% strongly typed,
            # avoiding saturation traps that prevent mixed-signature finds.
            n = _auto_n_from_dim(self.embedding_dim)
            self.p, self.q, self.r = 0, 0, n   # bookkeeping only
            eta_raw_init = torch.randn(n) * 1# N(0, 1.5)
        else:
            self.p = int(self.args.get('p', 1))
            self.q = int(self.args.get('q', 1))
            self.r = int(self.args.get('r', 0))
            n = self.p + self.q + self.r
            eta_init = torch.tensor(
                [1.0] * self.p + [-1.0] * self.q + [0.0] * self.r,
                dtype=torch.float32,
            )

        d = 1 << n #(d = 2^n = number of blades in Cl_{p,q,r})
        assert self.embedding_dim % d == 0, (
            f"FullDeCaL requires embedding_dim ({self.embedding_dim}) "
            f"divisible by 2^n = 2^{n} = {d}."
        )
        self.n  = n
        self.d  = d
        self.re = self.embedding_dim // d

        self.entity_embeddings   = torch.nn.Embedding(self.num_entities,  self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)

        # ── Structural buffers (always fixed, device-portable) ───────────
        sign_table, K_table, intersection_table, bits = _build_sign_table(n)
        self.register_buffer('_sign_table',         sign_table)          # (d,d)
        self.register_buffer('_K_table',            K_table)             # (d,d)
        self.register_buffer('_intersection_table', intersection_table)  # (d,d)
        self.register_buffer('_bits',               bits)                # (d,n)

        if self._auto:
            # Learned: η_k = tanh(η_raw_k) for each of the n generators.
            # α_K = ∏_{k∈K} η_k built in _coeff_table(); α[∅]=1 by construction.
            self.eta_raw = torch.nn.Parameter(eta_raw_init)   # (n,)
        else:
            # Fixed: precompute the coefficient table once as a frozen buffer
            eta_blade = (bits * eta_init + (1.0 - bits)).prod(dim=1)  # (d,)
            coeff = sign_table * eta_blade[intersection_table]         # (d,d)
            self.register_buffer('_coeff_fixed', coeff)

        mode_str = (
            f"auto (n={n}, d={d}, re={self.re}, η[{n}] learned via tanh)"
            if self._auto else
            f"fixed (p={self.p}, q={self.q}, r={self.r})"
        )
        logger.info(f"FullDeCaL mode: {mode_str}  |  embedding_dim={self.embedding_dim}")

    # ------------------------------------------------------------------ #
    #  Coefficient table (auto: differentiable; fixed: cached buffer)     #
    # ------------------------------------------------------------------ #

    def _coeff_table(self) -> torch.Tensor:
        if self._auto:
            # η_k = tanh(η_raw_k) ∈ (-1, 1)
            eta = torch.tanh(self.eta_raw)   # (n,)
            # α_K = ∏_{k∈K} η_k  via bit-mask product:
            #   bits[K, k] = 1 if generator k is in blade K, else 0
            #   eta_or_1[K, k] = η_k if k ∈ K, else 1
            #   α[K] = ∏_k eta_or_1[K, k]
            eta_or_1 = 1.0 - self._bits + self._bits * eta.unsqueeze(0)  # (d, n)
            alpha    = eta_or_1.prod(dim=1)                               # (d,)
            # coeff[I,J] = σ(I,J) · α[I∩J]
            return self._sign_table * alpha[self._intersection_table]
        else:
            return self._coeff_fixed  # precomputed frozen buffer

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _split(self, emb: torch.Tensor) -> torch.Tensor:
        """(B, d·re)  →  (B, d, re)"""
        return emb.view(emb.size(0), self.d, self.re)

    def _geo_product(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Batched geometric product z = h ⊛ r.  Returns (B, d, re)."""
        B     = h.size(0)
        d, re = self.d, self.re
        coeff = self._coeff_table()   # (d, d)

        hr = (torch.einsum('bie,bje->bije', h, r)
              * coeff.unsqueeze(0).unsqueeze(-1))   # (B, d, d, re)

        z       = torch.zeros(B, d, re, device=h.device, dtype=h.dtype)
        K_flat  = self._K_table.reshape(-1)
        hr_flat = hr.reshape(B, d * d, re)
        z.scatter_add_(
            1,
            K_flat.unsqueeze(0).unsqueeze(-1).expand(B, -1, re),
            hr_flat,
        )
        return z

    # ------------------------------------------------------------------ #
    #  KvsAll  –  (B, 2) → (B, |E|)                                      #
    # ------------------------------------------------------------------ #

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        h = self._split(head_ent_emb)
        r = self._split(rel_ent_emb)
        z = self._geo_product(h, r)
        T = self._split(self.entity_embeddings.weight)
        return torch.einsum('bkr,ekr->be', z, T)

    # ------------------------------------------------------------------ #
    #  NegSample  –  (B, 3) → (B,)                                       #
    # ------------------------------------------------------------------ #

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        h = self._split(head_ent_emb)
        r = self._split(rel_ent_emb)
        t = self._split(tail_ent_emb)
        z = self._geo_product(h, r)
        return torch.einsum('bkr,bkr->b', z, t)

    def learned_signature(self) -> dict:
        """Return the learned geometry for auto mode, or fixed spec for fixed mode.

        In auto mode, reports the learned α values (one per intersection-bitmask
        subset K) and estimates the best-fit Clifford signature by reading off
        η_k = α[{k}] (the singleton-subset coefficients, which equal η_k in any
        true Clifford algebra).

        Logs a human-readable summary.
        """
        if not self._auto:
            info = {
                'mode': 'fixed',
                'p': self.p, 'q': self.q, 'r': self.r,
                'n': self.n, 'd': self.d, 're': self.re,
            }
            logger.info(
                f'FullDeCaL fixed signature: Cl_{{{self.p},{self.q},{self.r}}}  '
                f'(n={self.n}, d={self.d}, re={self.re})'
            )
            return info

        eta_raw = self.eta_raw.detach().cpu()
        eta = torch.tanh(eta_raw).tolist()

        # Reconstruct full alpha vector for logging
        eta_t   = torch.tanh(eta_raw)           # (n,)
        bits    = self._bits.cpu()              # (d, n)
        eta_or_1 = 1.0 - bits + bits * eta_t.unsqueeze(0)  # (d, n)
        alpha   = eta_or_1.prod(dim=1).tolist()             # (d,)

        # Classify generators by sign; null if |\u03b7_k| < 0.15
        threshold = 0.1
        p = sum(1 for v in eta if v >  threshold)
        q = sum(1 for v in eta if v < -threshold)
        r = self.n - p - q

        # Format \u03b1 values grouped by grade
        grade_strs = []
        for K in range(self.d):
            bits_set = bin(K).count('1')
            v = alpha[K]
            grade_strs.append(f'  \u03b1[{K:0{self.n}b}] (grade {bits_set}) = {v:+.4f}')

        eta_labels = []
        for v in eta:
            if   v >  threshold: eta_labels.append(f'{v:+.4f}(+)')
            elif v < -threshold: eta_labels.append(f'{v:+.4f}(-)')
            else:                eta_labels.append(f'{v:+.4f}(~0)')

        logger.info(
            f'FullDeCaL learned geometry (n={self.n}, d={self.d}, re={self.re}):\n'
            f'  Learned η (tanh-bounded generator signatures):\n'
            + '\n'.join(f'  η[{k+1}] = {v:+.4f}  (raw={float(eta_raw[k]):+.4f})' for k, v in enumerate(eta))
            + '\n'
            f'  Derived α (blade intersection coefficients):\n'
            + '\n'.join(grade_strs) + '\n'
            f'  Best-fit η = [{", ".join(eta_labels)}]\n'
            f'  Best-fit algebra: Cl_{{{p},{q},{r}}}  '
            f'(p={p} positive, q={q} negative, r={r} null)'
        )
        return {
            'mode':       'auto',
            'eta':        eta,
            'eta_raw':    eta_raw.tolist(),
            'alpha':      alpha,
            'inferred_p': p, 'inferred_q': q, 'inferred_r': r,
            'n': self.n, 'd': self.d, 're': self.re,
        }
