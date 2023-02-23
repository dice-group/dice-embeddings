from .base_model import BaseKGE
import torch


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
            return torch.mm(ab0, c0.transpose(1, 0)) + torch.mm(ab1, c1.transpose(1, 0)) + torch.mm(ab2, c2.transpose(1,
                                                                                                                      0)) + torch.mm(
                ab12, c12.transpose(1, 0))
        elif self.p == 3 and self.q == 0:

            ab0, ab1, ab2, ab3, ab12, ab13, ab23, ab123 = ab
            c0, c1, c2, c3, c12, c13, c23, c123 = torch.hsplit(self.entity_embeddings.weight, 8)

            return torch.mm(ab0, c0.transpose(1, 0)) \
                   + torch.mm(ab1, c1.transpose(1, 0)) \
                   + torch.mm(ab2, c2.transpose(1, 0)) \
                   + torch.mm(ab3, c3.transpose(1, 0)) + torch.mm(ab12, c3.transpose(1, 0)) + torch.mm(ab13,
                                                                                                       c13.transpose(1,
                                                                                                                     0)) \
                   + torch.mm(ab23, c23.transpose(1, 0)) + torch.mm(ab123, c123.transpose(1, 0))
        elif self.p == 1 and self.q == 1:
            ab0, ab1, ab2, ab12 = ab
            c0, c1, c2, c12 = torch.hsplit(self.entity_embeddings.weight, 4)
            return torch.mm(ab0, c0.transpose(1, 0)) + torch.mm(ab1, c1.transpose(1, 0)) + torch.mm(ab2, c2.transpose(1,
                                                                                                                      0)) + torch.mm(
                ab12, c12.transpose(1, 0))

        else:
            raise NotImplementedError


class CLf(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'CLf'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.p = self.args['p']
        self.q = self.args['q']
        if self.p is None:
            self.p = 0
        if self.q is None:
            self.q = 0
        self.r = self.embedding_dim / (self.p + self.q + 1)
        print(f'r:{self.r}\tp:{self.p}\tq:{self.q}')
        try:
            assert self.r.is_integer()
        except AssertionError:
            print(f'r = embedding_dim / (p + q+ 1) must be a whole number\n'
                  f'Currently {self.r}={self.embedding_dim} / ({self.p}+ {self.q} +1)')
            print(f'r is corrected to {int(self.r)}')
        self.r = int(self.r)

    def clifford_mul(self, a0, ap, aq, b0, bp, bq):
        """ Compute our CL multiplication

        a = a_0 + \sum_i=1 ^p ap_i  v_i + \sum_j=1 ^q aq_j  u_j
        b = b_0 + \sum_i=1 ^p bp_i  v_i + \sum_j=1 ^q bq_j  u_j,

        ei ^2 = +1     for i =< i =< p
        ej ^2 = -1     for p < j =< p+q
        ei ej = -eje1  for i \neq j

        \mathbf{a}\mathbf{b} =   AB_0 + AB_p + AB_q + AB_{p,p}+ AB_{q,q} + AB_{p,q}
        where
                (1) AB_0      = a_0b_0 + \sum\limits_{i=1}^p ap_i bp_i - \sum\limits_{j=1}^q aq_j bq_j
                (2) AB_p      = \sum_{i=1}^p a_0bp_i + b_0 ap_i  v_i
                (3) AB_q      = \sum\limits_{j=1}^q a_0bq_j  + b_0 aq_j  u_j
                (4) AB_{p,p}  = \sum\limits_{i=1}^{p-1} \sum\limits_{k=i+1}^p ap_i bp_k - ap_k bp_i  v_i v_k
                (5) AB_{q,q}  = \sum\limits_{j=1}^{q-1} \sum\limits_{k=j+1}^q aq_j bp_k - aq_k bq_j  u_i u_k
                (6) AB_{p,q}  = \sum\limits_{i=1}^p \sum\limits_{j=1}^q ap_i bq_j - aq_j bp_i  v_i u_j.

        """
        n = len(a0)
        assert a0.shape == (n, self.r) == b0.shape == (n, self.r)
        assert ap.shape == (n, self.r, self.p) == bp.shape == (n, self.r, self.p)
        assert aq.shape == (n, self.r, self.q) == bq.shape == (n, self.r, self.q)

        AB_0: torch.FloatTensor
        # AB_0.shape: torch.Size([batch_size, r])
        # (1) AB_0 = a_0b_0 + \sum_{i=1}^p ap_i bp_i - \sum_{j=1}^q aq_j bq_j ,e.g. p=q=0, hadamard product
        AB_0 = torch.einsum('nr,nr->nr', a0, b0) \
               + torch.einsum('nrp,nrp->nr', ap, bp) \
               - torch.einsum('nrq,nrq->nr', aq, bq)
        assert AB_0.shape == (n, self.r)

        # (2) AB_p = \sum_{i=1}^p a_0 bp_i + b_0 ap_i  v_i
        # (2.1) \sum_{i=1}^p a_0 bp_i : multiply each column vector of r by p matrix (ap) with r-dimensional vector (a_0)
        # (2.2) \sum_{i=1}^p b_0 ap_i : multiply each column vector of r by p matrix (ap) with r-dimensional vector (b_0)
        # (2.3) Sum (2.1) and (2.2)
        # equiv. => a0.view(n, self.r, 1) * bp + b0.view(n, self.r, 1) * ap
        AB_p = torch.einsum('nr,nrp->nrp', a0, bp) + torch.einsum('nr,nrp->nrp', b0, ap)
        assert AB_p.shape == (n, self.r, self.p)

        # (3) AB_q = \sum_{j=1}^q a_0 bq_j  + b_0 aq_j  u_j
        # (3.1) \sum_{i=1}^q a_0 bq_i : multiply each column vector of r by q matrix (ap) with r-dimensional vector (a_0)
        # (3.2) \sum_{i=1}^q b_0 aq_i : multiply each column vector of r by q matrix (ap) with r-dimensional vector (b_0)
        # (3.3) Sum (3.1) and (3.2)
        # equiv. => a0.view(n, self.r, 1) * bq + b0.view(n, self.r, 1) * aq
        AB_q = torch.einsum('nr,nrq->nrq', a0, bq) + torch.einsum('nr,nrq->nrq', b0, aq)

        # (4) AB_{p,p}  = \sum_{i=1}^{p} \sum_{k=1}^p ap_i bp_k - ap_k bp_i  v_i v_k
        AB_pp = torch.einsum('nrp,nrx->nrpx', ap, bp) - torch.einsum('nrp,nrx->nrpx', bp, ap)
        assert AB_pp.shape == (n, self.r, self.p, self.p)
        # (5) AB_{q,q}  = \sum_{j=1}^{q} \sum_{j+1}^q aq_j bp_k - aq_k bq_j  u_i u_k
        AB_qq = torch.einsum('nrq,nrx->nrqx', aq, bq) - torch.einsum('nrq,nrx->nrqx', bq, aq)
        assert AB_qq.shape == (n, self.r, self.q, self.q)
        # (6) AB_{p,q}  = \sum_{i=1}^p \sum_{j=1}^q ap_i bq_j - aq_j bp_i  v_i u_j.
        AB_pq = torch.einsum('bkp,bkq->bkpq', ap, bq) - torch.einsum('bkp,bkq->bkpq', bp, aq)
        assert AB_pq.shape == (n, self.r, self.p, self.q)
        return AB_0, AB_p, AB_q, AB_pp, AB_qq, AB_pq

    def clifford_mul_reduced_interactions(self, a0, ap, aq, b0, bp, bq):
        """ Compute our CL multiplication

        a = a_0 + \sum_i=1 ^p ap_i  v_i + \sum_j=1 ^q aq_j  u_j
        b = b_0 + \sum_i=1 ^p bp_i  v_i + \sum_j=1 ^q bq_j  u_j,

        ei ^2 = +1     for i =< i =< p
        ej ^2 = -1     for p < j =< p+q
        ei ej = -eje1  for i \neq j

        \mathbf{a}\mathbf{b} =   AB_0 + AB_p + AB_q + AB_{p,p}+ AB_{q,q} + AB_{p,q}
        where
                (1) AB_0      = a_0b_0 + \sum\limits_{i=1}^p ap_i bp_i - \sum\limits_{j=1}^q aq_j bq_j
                (2) AB_p      = \sum_{i=1}^p a_0bp_i + b_0 ap_i  v_i
                (3) AB_q      = \sum\limits_{j=1}^q a_0bq_j  + b_0 aq_j  u_j
                (4) AB_{p,p}  = \sum\limits_{i=1}^{p-1} \sum\limits_{k=i+1}^p ap_i bp_k - ap_k bp_i  v_i v_k
                (5) AB_{q,q}  = \sum\limits_{j=1}^{q-1} \sum\limits_{k=j+1}^q aq_j bp_k - aq_k bq_j  u_i u_k
                (6) AB_{p,q}  = \sum\limits_{i=1}^p \sum\limits_{j=1}^q ap_i bq_j - aq_j bp_i  v_i u_j.

        """
        n = len(a0)
        assert a0.shape == (n, self.r) == b0.shape == (n, self.r)
        assert ap.shape == (n, self.r, self.p) == bp.shape == (n, self.r, self.p)
        assert aq.shape == (n, self.r, self.q) == bq.shape == (n, self.r, self.q)

        AB_0: torch.FloatTensor
        # AB_0.shape: torch.Size([batch_size, r])
        # (1) AB_0 = a_0b_0 + \sum_{i=1}^p ap_i bp_i - \sum_{j=1}^q aq_j bq_j ,e.g. p=q=0, hadamard product
        AB_0 = torch.einsum('nr,nr->nr', a0, b0) \
               + torch.einsum('nrp,nrp->nr', ap, bp) \
               - torch.einsum('nrq,nrq->nr', aq, bq)
        assert AB_0.shape == (n, self.r)

        # (2) AB_p = \sum_{i=1}^p a_0 bp_i + b_0 ap_i  v_i
        # (2.1) \sum_{i=1}^p a_0 bp_i : multiply each column vector of r by p matrix (ap) with r-dimensional vector (a_0)
        # (2.2) \sum_{i=1}^p b_0 ap_i : multiply each column vector of r by p matrix (ap) with r-dimensional vector (b_0)
        # (2.3) Sum (2.1) and (2.2)
        # equiv. => a0.view(n, self.r, 1) * bp + b0.view(n, self.r, 1) * ap
        AB_p = torch.einsum('nr,nrp->nrp', a0, bp) + torch.einsum('nr,nrp->nrp', b0, ap)
        assert AB_p.shape == (n, self.r, self.p)

        # (3) AB_q = \sum_{j=1}^q a_0 bq_j  + b_0 aq_j  u_j
        # (3.1) \sum_{i=1}^q a_0 bq_i : multiply each column vector of r by q matrix (ap) with r-dimensional vector (a_0)
        # (3.2) \sum_{i=1}^q b_0 aq_i : multiply each column vector of r by q matrix (ap) with r-dimensional vector (b_0)
        # (3.3) Sum (3.1) and (3.2)
        # equiv. => a0.view(n, self.r, 1) * bq + b0.view(n, self.r, 1) * aq
        AB_q = torch.einsum('nr,nrq->nrq', a0, bq) + torch.einsum('nr,nrq->nrq', b0, aq)

        # (4) AB_{p,p}  = \sum_{i=1}^{p-1} \sum_{k=i+1}^p ap_i bp_k - ap_k bp_i  v_i v_k
        # if p=2, then => (ap_1 bp_2 - ap_2 bp_1) v_1 v_2
        results = []
        for i in range(self.p - 1):
            for k in range(i + 1, self.p):
                x = ap[:, :, i] * bp[:, :, k] - ap[:, :, k] * ap[:, :, i]
                results.append(x.view(n, self.r).unsqueeze(-1))
        AB_pp = torch.stack(results, dim=2)

        # (5) AB_{q,q}  = \sum_{j=1}^{q-1} \sum_{k=j+1}^q aq_j bp_k - aq_k bq_j  u_i u_k
        # Explanation written in (4) holds for (5)
        results = []
        for i in range(self.q - 1):
            for k in range(i + 1, self.q):
                x = ap[:, :, i] * bp[:, :, k] - ap[:, :, k] * ap[:, :, i]
                results.append(x.view(n, self.r).unsqueeze(-1))
        AB_qq = torch.stack(results, dim=2)

        # (6) AB_{p,q}  = \sum_{i=1}^p \sum_{j=1}^q ap_i bq_j - aq_j bp_i  v_i u_j.
        AB_pq = torch.einsum('bkp,bkq->bkpq', ap, bq) - torch.einsum('bkp,bkq->bkpq', bp, aq)
        assert AB_pq.shape == (n, self.r, self.p, self.q)

        return AB_0, AB_p, AB_q, AB_pp, AB_qq, AB_pq

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
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
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for head entities.
        a0, ap, aq = self.construct_cl_multivector(head_ent_emb, r=self.r, p=self.p, q=self.q)
        # (2) Construct multi-vector in Cl_{p,q} (\mathbb{R}^d) for relations.
        b0, bp, bq = self.construct_cl_multivector(rel_ent_emb, r=self.r, p=self.p, q=self.q)

        # (4) Clifford multiplication of (2) and (3).
        # AB_pp, AB_qq, AB_pq
        AB_0, AB_p, AB_q, AB_pp, AB_qq, AB_pq = self.clifford_mul(a0, ap, aq, b0, bp, bq)

        # (7) Inner product of AB_0 and a0 of all entities.
        A_score = torch.einsum('bk,ek->be', AB_0, self.entity_embeddings.weight[:, :self.r])
        # (8) Inner product of AB_p and ap of all entities.
        if self.p > 0:
            B_score = torch.einsum('bkp,ekp->be', AB_p,
                                   self.entity_embeddings.weight[:, self.r: self.r + (self.r * self.p)]
                                   .view(self.num_entities, self.r, self.p))
        else:
            B_score = 0
        # (9) Inner product of AB_q and aq of all entities.
        if self.q > 0:
            C_score = torch.einsum('bkq,ekq->be', AB_q,
                                   self.entity_embeddings.weight[:, -(self.r * self.q):]
                                   .view(self.num_entities, self.r, self.q))
        else:
            C_score = 0
        # (10) Aggregate (7,8,9).
        A_B_C_score = A_score + B_score + C_score
        # (11) Compute inner products of AB_pp, AB_qq, AB_pq and respective identity matrices of all entities.
        D_E_F_score = (torch.einsum('bkpp->b', AB_pp) + torch.einsum('bkqq->b', AB_qq) + torch.einsum('bkpq->b',AB_pq))
        D_E_F_score=D_E_F_score.view(len(head_ent_emb), 1)
        # (12) Score
        return A_B_C_score + D_E_F_score

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

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Construct k-dimensional vector in CL_{p,q} for head entities.
        A_head, B_head, C_head = self.construct_cl_vector(head_ent_emb)
        # (3) Construct n dimensional vector in CL_{p,q} for relations.
        A_rel, B_rel, C_rel = self.construct_cl_vector(rel_ent_emb)
        # (4) Construct n dimensional vector in CL_{p,q} for tail entities.
        A_tail, B_tail, C_tail = self.construct_cl_vector(tail_ent_emb)

        # (5) Clifford multiplication of (2) and (3)
        A, B, C, D, E, F = self.clifford_mul_with_einsum(A_head, B_head, C_head, A_rel, B_rel, C_rel)
        # (6) Inner product of (5) and (4)
        A_score = torch.einsum('bk,bk->b', A_tail, A)
        B_score = torch.einsum('bkp,bkp->b', B_tail, B)
        C_score = torch.einsum('bkq,bkq->b', C_tail, C)
        D_score = torch.einsum('bkpp->b', D)
        E_score = torch.einsum('bkqq->b', E)
        F_score = torch.einsum('bkpq->b', F)
        return A_score + B_score + C_score + D_score + E_score + F_score
