from .base_model import BaseKGE
import torch


def clifford_mul(x: torch.FloatTensor, y: torch.FloatTensor, p: int, q: int) -> tuple:
    """
    Clifford multiplication Cl_{p,q} (\mathbb{R})

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
        # (1) Elementwise multiplication CL_0,0(\mathbb{R}) is isomorphic to \mathbb{R}
        return x * y
    elif p == 1 and q == 0:
        # (2) Elementwise multiplication CL_0,0(\mathbb{R}) is isomorphic to ? \mathbb{R}(2)
        a0, a1 = torch.hsplit(x, 2)
        b0, b1 = torch.hsplit(y, 2)
        ab0 = a0 * b0 + a1 * b1
        ab1 = a0 * b1 + a1 * b0
        return ab0, ab1
    elif p == 0 and q == 1:
        # (2) Elementwise multiplication CL_0,0(\mathbb{R}) is isomorphic to \mathbb{C}
        a0, a1 = torch.hsplit(x, 2)
        b0, b1 = torch.hsplit(y, 2)

        ab0 = a0 * b0 - a1 * b1
        ab1 = a0 * b1 + a1 * b0
        return ab0, ab1
    elif p == 2 and q == 0:
        # (2) Elementwise multiplication CL_0,0(\mathbb{R}) is isomorphic to ?\mathbb{C}
        a0, a1, a2, a12 = torch.hsplit(x, 4)
        b0, b1, b2, b12 = torch.hsplit(y, 4)
        # (2) multiplication
        ab0 = a0 * b0 + a1 * b1 + a2 * b2 - a12 * b12
        ab1 = a0 * b1 + a1 * b0 - a2 * b12 + a12 * b2
        ab2 = a0 * b2 + a1 * b12 + a2 * b0 - a12 * b1
        ab12 = a0 * b12 + a1 * b2 - a2 * b1 + a12 * b0
        return ab0, ab1, ab2, ab12
    elif p == 0 and q == 2:
        # (2) Elementwise multiplication CL_0,0(\mathbb{R}) is isomorphic to \mathbb{H}
        a0, a1, a2, a12 = torch.hsplit(x, 4)
        b0, b1, b2, b12 = torch.hsplit(y, 4)

        ab0 = a0 * b0 - a1 * b1 - a2 * b2 - a12 * b12
        ab1 = a0 * b1 + a1 * b0 + a2 * b12 - a12 * b2
        ab2 = a0 * b2 - a1 * b12 + a2 * b0 + a12 * b1
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

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        ab = clifford_mul(x=head_ent_emb, y=rel_ent_emb, p=self.p, q=self.q)

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
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # CL mult of
        ab = clifford_mul(x=head_ent_emb, y=rel_ent_emb, p=self.p, q=self.q)

        if self.p == self.q == 0:
            return torch.mm(ab, self.entity_embeddings.weight.transpose(1, 0))
            # produces different results
            # return torch.einsum('bd,kd->bk', ab, Emb_all)

        elif (self.p == 1 and self.q == 0) or (self.p == 0 and self.q == 1):
            ab0, ab1 = ab
            c0, c1 = torch.hsplit(self.entity_embeddings.weight, 2)
            # return torch.mm(ab0, c0.transpose(1, 0))+torch.mm(ab1, c1.transpose(1, 0))
            return torch.einsum('bd,kd->bk', ab0, c0) + torch.einsum('bd,kd->bk', ab1, c1)
        elif (self.p == 2 and self.q == 0) or (self.p == 0 and self.q == 2):
            ab0, ab1, ab2, ab12 = ab
            c0, c1, c2, c12 = torch.hsplit(self.entity_embeddings.weight, 4)
            """
            # Slightly slower
            return torch.einsum('bd,kd->bk', ab0, c0) \
                   + torch.einsum('bd,kd->bk', ab1, c1) \
                   + torch.einsum('bd,kd->bk', ab2, c2) \
                   + torch.einsum('bd,kd->bk', ab12, c12)
            """
            return torch.mm(ab0, c0.transpose(1, 0)) + torch.mm(ab1, c1.transpose(1, 0)) + torch.mm(ab2, c2.transpose(1, 0)) + torch.mm(ab12, c12.transpose(1, 0))
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

        else:
            raise NotImplementedError


class CLf(BaseKGE):
    """Clifford:Embedding Space Search in Clifford Algebras


    h = A_{d \times 1}, B_{d \times p}, C_{d \times q}

    r = A'_{d \times 1}, B'_{d \times p}, C'_{d \times q}

    t = A''_{d \times 1}, B''_{d \times p}, C_{d \times q}

    """

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
        self.k = self.embedding_dim / (self.p + self.q + 1)
        print(f'k:{self.k}\tp:{self.p}\tq:{self.q}')
        try:
            assert self.k.is_integer()
        except AssertionError:
            raise AssertionError(f'k= embedding_dim / (p + q+ 1) must be a whole number\n'
                                 f'Currently {self.k}={self.embedding_dim} / ({self.p}+ {self.q} +1)')
        self.k = int(self.k)

    def construct_cl_vector(self, batch_x: torch.FloatTensor) -> tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """ Convert a batch of d-dimensional real valued vectors to a batch of k-dimensional Clifford vectors
        batch_x: n \times d

        batch_x_cl: A_{n \times k,1}, B_{n \times k \times p}, C_{n \times k \times q}
        """
        batch_size, d = batch_x.shape
        # (1) A_{n \times k}: take the first k columns
        A = batch_x[:, :self.k].view(batch_size, self.k)
        # (2) B_{n \times p}, C_{n \times q}: take the self.k * self.p columns after the k. column
        if self.p > 0:
            B = batch_x[:, self.k: self.k + (self.k * self.p)].view(batch_size, self.k, self.p)
        else:
            B = torch.zeros((batch_size, self.k, self.p))

        if self.q > 0:
            # (3) B_{n \times p}, C_{n \times q}: take the last self.k * self.q .
            C = batch_x[:, -(self.k * self.q):].view(batch_size, self.k, self.q)
        else:
            C = torch.zeros((batch_size, self.k, self.q))

        return A, B, C

    def clifford_mul_with_einsum(self, A_h, B_h, C_h, A_r, B_r, C_r):
        """ Compute CL multiplication """
        # (1) Compute A: batch_size (b), A \in \mathbb{R}^k
        A = torch.einsum('bk,bk->bk', A_h, A_r) \
            + torch.einsum('bkp,bkp->bk', B_h, B_r) \
            - torch.einsum('bkq,bkq->bk', C_h, C_r)
        # (2) Compute B: batch_size (b), B \in \mathbb{R}^{k \times p}
        B = torch.einsum('bk,bkp->bkp', A_h, B_r) + torch.einsum('bk,bkp->bkp', A_r, B_h)
        # (3) Compute C: batch_size (b), C \in \mathbb{R}^{k \times q}
        C = torch.einsum('bk,bkq->bkq', A_h, C_r) + torch.einsum('bk,bkq->bkq', A_r, C_h)

        # (4) Compute D: batch_size (b), D \in \mathbb{R}^{k \times  p \times p}
        """
        x = B_h.transpose(1, 2)
        b, p, k = x.shape
        results = []
        for i in range(k):
            a = x[:, :, i].view(len(x), self.p, 1)
            b = B_r[:, i, :].view(len(x), 1, self.p)
            results.append(a @ b)
        results = torch.stack(results, dim=1)
        """
        D = torch.einsum('bkp,bkl->bkpl', B_h, B_r) - torch.einsum('bkp,bkl->bkpl', B_r, B_h)
        # (5) Compute E: batch_size (b), E \in \mathbb{R}^{k \times  q \times q}
        E = torch.einsum('bkq,bkl->bkql', C_h, C_r) - torch.einsum('bkq,bkl->bkql', C_r, C_h)
        # (5) Compute F: batch_size (b), E \in \mathbb{R}^{k \times  p \times q}
        F = torch.einsum('bkp,bkq->bkpq', B_h, C_r) - torch.einsum('bkp,bkq->bkpq', B_r, C_h)

        return A, B, C, D, E, F

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

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Construct k-dimensional vector in CL_{p,q} for head entities.
        A_head, B_head, C_head = self.construct_cl_vector(head_ent_emb)
        # (3) Construct n dimensional vector in CL_{p,q} for relations.
        A_rel, B_rel, C_rel = self.construct_cl_vector(rel_ent_emb)
        # (5) Clifford multiplication of (2) and (3).
        A, B, C, D, E, F = self.clifford_mul_with_einsum(A_head, B_head, C_head, A_rel, B_rel, C_rel)

        # (6) Extract embeddings for all entities.
        Emb_all = self.entity_embeddings.weight
        # (7) Compute A
        A_all = Emb_all[:, :self.k]
        A_score = torch.einsum('bk,ek->be', A, A_all)
        # (8) Compute B
        if self.p > 0:
            B_all = Emb_all[:, self.k: self.k + (self.k * self.p)].view(self.num_entities, self.k, self.p)
            B_score = torch.einsum('bkp,ekp->be', B, B_all)
        else:
            B_score = 0
        # (9) Compute C
        if self.q > 0:
            C_all = Emb_all[:, -(self.k * self.q):].view(self.num_entities, self.k, self.q)
            C_score = torch.einsum('bkq,ekq->be', C, C_all)
        else:
            C_score = 0
        # (10) Aggregate (7,8,9)
        A_B_C_score = A_score + B_score + C_score
        # (11) Compute and Aggregate D,E,F
        D_E_F_score = torch.einsum('bkpp->b', D) + torch.einsum('bkqq->b', E) + torch.einsum('bkpq->b', F)
        D_E_F_score = D_E_F_score.view(len(D_E_F_score), 1)
        # (12) Score
        return A_B_C_score + D_E_F_score
