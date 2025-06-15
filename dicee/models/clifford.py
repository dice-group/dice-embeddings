from .base_model import BaseKGE
import torch
from torch.nn import functional as F
from torch import nn


class CMult(BaseKGE):
    """
    Cl_(0,0) => Real Numbers


    Cl_(0,1) =>
                A multivector \mathbf{a} = a_0 + a_1 e_1
                A multivector \mathbf{b} = b_0 + b_1 e_1

                multiplication is isomorphic to the product of two complex numbers

                \mathbf{a} \times \mathbf{b} = a_0 b_0 + a_0b_1 e1 + a_1 b_1 e_1 e_1
                                             = (a_0 b_0 - a_1 b_1) + (a_0 b_1 + a_1 b_0) e_1
    Cl_(2,0) =>
                A multivector \mathbf{a} = a_0 + a_1 e_1 + a_2 e_2 + a_{12} e_1 e_2
                A multivector \mathbf{b} = b_0 + b_1 e_1 + b_2 e_2 + b_{12} e_1 e_2

                \mathbf{a} \times \mathbf{b} = a_0b_0 + a_0b_1 e_1 + a_0b_2e_2 + a_0 b_12 e_1 e_2
                                            + a_1 b_0 e_1 + a_1b_1 e_1_e1 ..

    Cl_(0,2) => Quaternions
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'CMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.p = self.args['p']
        self.q = self.args['q']
        if self.p is None:
            self.p = 0
        if self.q is None:
            self.q = 0
        print(f'\tp:{self.p}\tq:{self.q}')

    def clifford_mul(self, x: torch.FloatTensor, y: torch.FloatTensor, p: int, q: int) -> tuple:
        """
        Clifford multiplication Cl_{p,q} (\mathbb{R})

        ei ^2 = +1     for i =< i =< p
        ej ^2 = -1     for p < j =< p+q
        ei ej = -eje1  for i \neq j

        Parameter
        ---------
        x: torch.FloatTensor with (n,d) shape

        y: torch.FloatTensor with (n,d) shape

        p: a non-negative integer p>= 0
        q: a non-negative integer q>= 0



        Returns
        -------

        """

        if p == q == 0:
            return x * y
        elif (p == 1 and q == 0) or (p == 0 and q == 1):
            # {1,e1} e_i^2 = +1 for i
            a0, a1 = torch.hsplit(x, 2)
            b0, b1 = torch.hsplit(y, 2)
            if p == 1 and q == 0:
                ab0 = a0 * b0 + a1 * b1
                ab1 = a0 * b1 + a1 * b0
            else:
                ab0 = a0 * b0 - a1 * b1
                ab1 = a0 * b1 + a1 * b0
            return ab0, ab1
        elif (p == 2 and q == 0) or (p == 0 and q == 2):
            a0, a1, a2, a12 = torch.hsplit(x, 4)
            b0, b1, b2, b12 = torch.hsplit(y, 4)
            if p == 2 and q == 0:
                ab0 = a0 * b0 + a1 * b1 + a2 * b2 - a12 * b12
                ab1 = a0 * b1 + a1 * b0 - a2 * b12 + a12 * b2
                ab2 = a0 * b2 + a1 * b12 + a2 * b0 - a12 * b1
                ab12 = a0 * b12 + a1 * b2 - a2 * b1 + a12 * b0
            else:
                ab0 = a0 * b0 - a1 * b1 - a2 * b2 - a12 * b12
                ab1 = a0 * b1 + a1 * b0 + a2 * b12 - a12 * b2
                ab2 = a0 * b2 - a1 * b12 + a2 * b0 + a12 * b1
                ab12 = a0 * b12 + a1 * b2 - a2 * b1 + a12 * b0
            return ab0, ab1, ab2, ab12
        elif p == 1 and q == 1:
            a0, a1, a2, a12 = torch.hsplit(x, 4)
            b0, b1, b2, b12 = torch.hsplit(y, 4)

            ab0 = a0 * b0 + a1 * b1 - a2 * b2 + a12 * b12
            ab1 = a0 * b1 + a1 * b0 + a2 * b12 - a12 * b2
            ab2 = a0 * b2 + a1 * b12 + a2 * b0 - a12 * b1
            ab12 = a0 * b12 + a1 * b2 - a2 * b1 + a12 * b0
            return ab0, ab1, ab2, ab12
        elif p == 3 and q == 0:
            # cl3,0 no 0,3
            a0, a1, a2, a3, a12, a13, a23, a123 = torch.hsplit(x, 8)
            b0, b1, b2, b3, b12, b13, b23, b123 = torch.hsplit(y, 8)

            ab0 = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 - a12 * b12 - a13 * b13 - a23 * b23 - a123 * b123
            ab1 = a0 * b1 + a1 * b0 - a2 * b12 - a3 * b13 + a12 * b2 + a13 * b3 - a23 * b123 - a123 * b23
            ab2 = a0 * b2 + a1 * b12 + a2 * b0 - a3 * b23 - a12 * b1 + a13 * b123 + a23 * b3 + a123 * b13
            ab3 = a0 * b3 + a1 * b13 + a2 * b23 + a3 * b0 - a12 * b123 - a13 * b1 - a23 * b2 - a123 * b12
            ab12 = a0 * b12 + a1 * b2 - a2 * b1 + a3 * b123 + a12 * b0 - a13 * b23 + a23 * b13 + a123 * b3
            ab13 = a0 * b13 + a1 * b3 - a2 * b123 - a3 * b1 + a12 * b23 + a13 * b0 - a23 * b12 - a123 * b2
            ab23 = a0 * b23 + a1 * b123 + a2 * b3 - a3 * b2 - a12 * b13 - a13 * b12 + a23 * b0 + a123 * b1
            ab123 = a0 * b123 + a1 * b23 - a2 * b13 + a3 * b12 + a12 * b3 - a13 * b2 + a23 * b1 + a123 * b0
            return ab0, ab1, ab2, ab3, ab12, ab13, ab23, ab123
        else:
            raise NotImplementedError

    def score(self, head_ent_emb, rel_ent_emb, tail_ent_emb):
        ab = self.clifford_mul(x=head_ent_emb, y=rel_ent_emb, p=self.p, q=self.q)

        if self.p == self.q == 0:
            return torch.einsum('bd,bd->b', ab, tail_ent_emb)
        elif (self.p == 1 and self.q == 0) or (self.p == 0 and self.q == 1):
            ab0, ab1 = ab
            c0, c1 = torch.hsplit(tail_ent_emb, 2)
            return torch.einsum('bd,bd->b', ab0, c0) + torch.einsum('bd,bd->b', ab1, c1)
        elif (self.p == 2 and self.q == 0) or (self.p == 0 and self.q == 2):
            ab0, ab1, ab2, ab12 = ab
            c0, c1, c2, c12 = torch.hsplit(tail_ent_emb, 4)
            return torch.einsum('bd,bd->b', ab0, c0) \
                + torch.einsum('bd,bd->b', ab1, c1) \
                + torch.einsum('bd,bd->b', ab2, c2) \
                + torch.einsum('bd,bd->b', ab12, c12)
        else:
            raise NotImplementedError

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Compute batch triple scores

        Parameter
        ---------
        x: torch.LongTensor with shape n by 3


        Returns
        -------
        torch.LongTensor with shape n

        """

        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        ab = self.clifford_mul(x=head_ent_emb, y=rel_ent_emb, p=self.p, q=self.q)

        if self.p == self.q == 0:
            return torch.einsum('bd,bd->b', ab, tail_ent_emb)
        elif (self.p == 1 and self.q == 0) or (self.p == 0 and self.q == 1):
            ab0, ab1 = ab
            c0, c1 = torch.hsplit(tail_ent_emb, 2)
            return torch.einsum('bd,bd->b', ab0, c0) + torch.einsum('bd,bd->b', ab1, c1)
        elif (self.p == 2 and self.q == 0) or (self.p == 0 and self.q == 2):
            ab0, ab1, ab2, ab12 = ab
            c0, c1, c2, c12 = torch.hsplit(tail_ent_emb, 4)
            return torch.einsum('bd,bd->b', ab0, c0) \
                + torch.einsum('bd,bd->b', ab1, c1) \
                + torch.einsum('bd,bd->b', ab2, c2) \
                + torch.einsum('bd,bd->b', ab12, c12)
        else:
            raise NotImplementedError

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Compute batch KvsAll triple scores

        Parameter
        ---------
        x: torch.LongTensor with shape n by 3


        Returns
        -------
        torch.LongTensor with shape n

        """
        # (1) Retrieve embedding vectors of heads and relations.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) CL multiply (1).
        ab = self.clifford_mul(x=head_ent_emb, y=rel_ent_emb, p=self.p, q=self.q)
        # (3) Inner product of (2) and all entity embeddings.
        if self.p == self.q == 0:
            return torch.mm(ab, self.entity_embeddings.weight.transpose(1, 0))
        elif (self.p == 1 and self.q == 0) or (self.p == 0 and self.q == 1):
            ab0, ab1 = ab
            c0, c1 = torch.hsplit(self.entity_embeddings.weight, 2)
            return torch.mm(ab0, c0.transpose(1, 0)) + torch.mm(ab1, c1.transpose(1, 0))
        elif (self.p == 2 and self.q == 0) or (self.p == 0 and self.q == 2):
            ab0, ab1, ab2, ab12 = ab
            c0, c1, c2, c12 = torch.hsplit(self.entity_embeddings.weight, 4)
            return torch.mm(ab0, c0.transpose(1, 0)) + \
                torch.mm(ab1, c1.transpose(1, 0)) + torch.mm(ab2, c2.transpose(1, 0)) + torch.mm(
                    ab12, c12.transpose(1, 0))
        elif self.p == 3 and self.q == 0:

            ab0, ab1, ab2, ab3, ab12, ab13, ab23, ab123 = ab
            c0, c1, c2, c3, c12, c13, c23, c123 = torch.hsplit(self.entity_embeddings.weight, 8)

            return torch.mm(ab0, c0.transpose(1, 0)) \
                + torch.mm(ab1, c1.transpose(1, 0)) \
                + torch.mm(ab2, c2.transpose(1, 0)) \
                + torch.mm(ab3, c3.transpose(1, 0)) + \
                torch.mm(ab12, c3.transpose(1, 0)) + torch.mm(ab13, c13.transpose(1, 0)) \
                + torch.mm(ab23, c23.transpose(1, 0)) + torch.mm(ab123, c123.transpose(1, 0))
        elif self.p == 1 and self.q == 1:
            ab0, ab1, ab2, ab12 = ab
            c0, c1, c2, c12 = torch.hsplit(self.entity_embeddings.weight, 4)
            return torch.mm(ab0, c0.transpose(1, 0)) + torch.mm(ab1, c1.transpose(1, 0)) + \
                torch.mm(ab2, c2.transpose(1, 0)) + torch.mm(
                    ab12, c12.transpose(1, 0))

        else:
            raise NotImplementedError


class Keci(BaseKGE):
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
        if self.p > 0:
            self.p_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.p)
            torch.nn.init.zeros_(self.p_coefficients.weight)
        if self.q > 0:
            self.q_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.q)
            torch.nn.init.zeros_(self.q_coefficients.weight)

    def compute_sigma_pp(self, hp, rp):
        """
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
        """
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
            sigma_qq = torch.zeros((len(hq), self.r, int((self.q * (self.q - 1)) / 2)))

        return sigma_qq

    def compute_sigma_pq(self, *, hp, hq, rp, rq):
        """
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

    def apply_coefficients(self, h0, hp, hq, r0, rp, rq):
        """ Multiplying a base vector with its scalar coefficient """
        if self.p > 0:
            hp = hp * self.p_coefficients.weight
            rp = rp * self.p_coefficients.weight
        if self.q > 0:
            hq = hq * self.q_coefficients.weight
            rq = rq * self.q_coefficients.weight
        return h0, hp, hq, r0, rp, rq

    def clifford_multiplication(self, h0, hp, hq, r0, rp, rq):
        """ Compute our CL multiplication

        h = h_0 + \sum_{i=1}^p h_i e_i + \sum_{j=p+1}^{p+q} h_j e_j
        r = r_0 + \sum_{i=1}^p r_i e_i + \sum_{j=p+1}^{p+q} r_j e_j

        ei ^2 = +1     for i =< i =< p
        ej ^2 = -1     for p < j =< p+q
        ei ej = -eje1  for i \neq j

        h r =   sigma_0 + sigma_p + sigma_q + sigma_{pp} + sigma_{q}+ sigma_{pq}
        where
                (1) sigma_0 = h_0 r_0 + \sum_{i=1}^p (h_0 r_i) e_i - \sum_{j=p+1}^{p+q} (h_j r_j) e_j

                (2) sigma_p = \sum_{i=1}^p (h_0 r_i + h_i r_0) e_i

                (3) sigma_q = \sum_{j=p+1}^{p+q} (h_0 r_j + h_j r_0) e_j

                (4) sigma_{pp} = \sum_{i=1}^{p-1} \sum_{k=i+1}^p (h_i r_k - h_k r_i) e_i e_k

                (5) sigma_{qq} = \sum_{j=1}^{p+q-1} \sum_{k=j+1}^{p+q} (h_j r_k - h_k r_j) e_j e_k

                (6) sigma_{pq} = \sum_{i=1}^{p} \sum_{j=p+1}^{p+q} (h_i r_j - h_j r_i) e_i e_j

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
        """
        Construct a batch of multivectors Cl_{p,q}(\mathbb{R}^d)

        Parameter
        ---------
        x: torch.FloatTensor with (n,d) shape

        Returns
        -------
        a0: torch.FloatTensor with (n,r) shape
        ap: torch.FloatTensor with (n,r,p) shape
        aq: torch.FloatTensor with (n,r,q) shape
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

    def forward_k_vs_with_explicit(self, x: torch.Tensor):
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

    def k_vs_all_score(self, bpe_head_ent_emb, bpe_rel_ent_emb, E):
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq = self.construct_cl_multivector(bpe_head_ent_emb, r=self.r, p=self.p, q=self.q)
        r0, rp, rq = self.construct_cl_multivector(bpe_rel_ent_emb, r=self.r, p=self.p, q=self.q)

        h0, hp, hq, h0, rp, rq = self.apply_coefficients(h0, hp, hq, h0, rp, rq)
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
        """
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

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor) -> torch.FloatTensor:
        """
        Kvsall training

        (1) Retrieve real-valued embedding vectors for heads and relations \mathbb{R}^d .
        (2) Construct head entity and relation embeddings according to Cl_{p,q}(\mathbb{R}^d) .
        (3) Perform Cl multiplication
        (4) Inner product of (3) and all entity embeddings

        Parameter
        ---------
        x: torch.LongTensor with (n,2) shape

        Returns
        -------
        torch.FloatTensor with (n, |E|) shape
        """
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_emb = self.get_head_relation_representation(x)
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities.
        a0, ap, aq = self.construct_cl_multivector(head_ent_emb, r=self.r, p=self.p, q=self.q)
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for relations.
        b0, bp, bq = self.construct_cl_multivector(rel_emb, r=self.r, p=self.p, q=self.q)

        # (4) Clifford multiplication of (2) and (3).
        # AB_pp, AB_qq, AB_pq
        # AB_0, AB_p, AB_q, AB_pp, AB_qq, AB_pq = self.clifford_mul_reduced_interactions(a0, ap, aq, b0, bp, bq)
        AB_0, AB_p, AB_q, AB_pp, AB_qq, AB_pq = self.clifford_mul(a0, ap, aq, b0, bp, bq)

        # b e r
        selected_tail_entity_embeddings = self.entity_embeddings(target_entity_idx)
        # (7) Inner product of AB_0 and a0 of all entities.
        A_score = torch.einsum('br,ber->be', AB_0, selected_tail_entity_embeddings[:, :self.r])

        # (8) Inner product of AB_p and ap of all entities.
        if self.p > 0:
            B_score = torch.einsum('brp,berp->be', AB_p,
                                   selected_tail_entity_embeddings[:, self.r: self.r + (self.r * self.p)]
                                   .view(self.num_entities, self.r, self.p))
        else:
            B_score = 0
        # (9) Inner product of AB_q and aq of all entities.
        if self.q > 0:
            C_score = torch.einsum('brq,berq->be', AB_q,
                                   selected_tail_entity_embeddings[:, -(self.r * self.q):]
                                   .view(self.num_entities, self.r, self.q))
        else:
            C_score = 0
        # (10) Aggregate (7,8,9).
        A_B_C_score = A_score + B_score + C_score
        # (11) Compute inner products of AB_pp, AB_qq, AB_pq and respective identity matrices of all entities.
        D_E_F_score = (torch.einsum('brpp->b', AB_pp) + torch.einsum('brqq->b', AB_qq) + torch.einsum('brpq->b', AB_pq))
        D_E_F_score = D_E_F_score.view(len(head_ent_emb), 1)
        # (12) Score
        return A_B_C_score + D_E_F_score

    def score(self, h, r, t):
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities and relations
        h0, hp, hq = self.construct_cl_multivector(h, r=self.r, p=self.p, q=self.q)
        r0, rp, rq = self.construct_cl_multivector(r, r=self.r, p=self.p, q=self.q)
        t0, tp, tq = self.construct_cl_multivector(t, r=self.r, p=self.p, q=self.q)

        if self.q > 0:
            self.q_coefficients = self.q_coefficients.to(h0.device, non_blocking=True)

        h0, hp, hq, h0, rp, rq = self.apply_coefficients(h0, hp, hq, h0, rp, rq)
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
        h0, hp, hq, h0, rp, rq = self.apply_coefficients(h0, hp, hq, h0, rp, rq)
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


class KeciBase(Keci):
    " Without learning dimension scaling"

    def __init__(self, args):
        super().__init__(args)
        self.name = 'KeciBase'
        self.requires_grad_for_interactions = False
        print(f'r:{self.r}\t p:{self.p}\t q:{self.q}')
        if self.p > 0:
            self.p_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.p)
            torch.nn.init.ones_(self.p_coefficients.weight)
        if self.q > 0:
            self.q_coefficients = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.q)
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
            sigma_pp = torch.sum(self.compute_sigma_pp(hp, rp), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_pp = 0

        if self.q >= 2:
            sigma_qq = torch.sum(self.compute_sigma_qq(hq, rq), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_qq = 0

        if self.r >= 2:
            sigma_rr = torch.sum(self.compute_sigma_qq(hk, rk), dim=[1, 2]).unsqueeze(-1)
        else:
            sigma_rr = 0

        if self.p >= 2 and self.q >= 2:
            sigma_pq = torch.sum(self.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pq = 0

        if self.p >= 2 and self.r >= 2:
            sigma_pr = torch.sum(self.compute_sigma_pq(hp=hp, hk=hk, rp=rp, rk=rk), dim=[1, 2, 3]).unsqueeze(-1)
        else:
            sigma_pr = 0
        if self.q >= 2 and self.r >= 2:
            sigma_qr = torch.sum(self.compute_sigma_pq(hq=hq, hk=hk, rq=rq, rk=rk), dim=[1, 2, 3]).unsqueeze(-1)
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

        '''here we compute all the sums with no others vectors interaction taken with the scalar product with t, that is,
        
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

        '''Here we compute and return all the sums with vectors interaction for the same and different bases.

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

        """
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
        """
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
        """
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
        """
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
        """
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
        """
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
        """
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
        """
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
