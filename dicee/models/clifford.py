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
        self.k = self.embedding_dim / (self.p + self.q + 1)
        print(f'k:{self.k}\tp:{self.p}\tq:{self.q}')
        try:
            assert self.k.is_integer()
        except AssertionError:
            raise AssertionError(f'k= embedding_dim / (p + q+ 1) must be a whole number\n'
                                 f'Currently {self.k}={self.embedding_dim} / ({self.p}+ {self.q} +1)')
        self.k = int(self.k)

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Construct k-dimensional vector in CL_{p,q} for head entities.

        # (1) A_{n \times k}: take the first k columns
        if self.p == self.q == 0:
            # DistMult
            return torch.einsum('bd,bd,bd->b', head_ent_emb, rel_ent_emb, tail_ent_emb)
        elif self.p == 0 and self.q == 1:
            # ComplEx
            batch_size, d = head_ent_emb.shape
            # Complex representations
            head_0 = head_ent_emb[:, :self.k].view(batch_size, self.k)
            head_1 = head_ent_emb[:, -self.k:].view(batch_size, self.k)
            rel_0 = rel_ent_emb[:, :self.k].view(batch_size, self.k)
            rel_1 = rel_ent_emb[:, -self.k:].view(batch_size, self.k)
            tail_0 = tail_ent_emb[:, :self.k].view(batch_size, self.k)
            tail_1 = tail_ent_emb[:, -self.k:].view(batch_size, self.k)

            # Complex multiplication
            rel = torch.einsum('bd,bd->bd', head_0, rel_0) - torch.einsum('bd,bd->bd', head_1, rel_1)
            imag = torch.einsum('bd,bd->bd', head_0, rel_1) + torch.einsum('bd,bd->bd', head_1, rel_0)

            return torch.einsum('bd,bd->b', rel, tail_0) + torch.einsum('bd,bd->b', imag, tail_1)
        elif self.p == 0 and self.q == 2:
            # Quaternions
            # Eq. 39
            # a = a_0 + a_1 e_1 + a_2 e_2 + a_{12} e1 e2
            a0, a1, a2, a12 = torch.hsplit(head_ent_emb, 4)
            b0, b1, b2, b12 = torch.hsplit(rel_ent_emb, 4)

            ab0 = a0 * b0 - a1 * b1 - a2 * b2 - a12 * b12
            ab1 = a0 * b1 + a1 * b0 + a2 * b12 - a12 * b2
            ab2 = a0 * b2 - a1 * b12 + a2 * b0 + a12 * b1
            ab12 = a0 * b12 - a1 * b2 + a12 * b0
            ab = torch.cat((ab0, ab1, ab2, ab12), dim=1)
            return torch.einsum('bd, bd -> b', ab, tail_ent_emb)
        else:
            raise NotImplementedError

        exit(1)

        a0b0 = torch.einsum('bd,bd->b', head_a0, rel_a0)

        print(head_a0[0])
        print(head_a1[0])
        print(head_a2[0])

        exit(1)
        print(a0.shape)
        print(a1.shape)
        print(a2.shape)
        exit(1)
        # (5)mult (2) and (3)
        a0b0 = torch.einsum('bd,bd->b', a0, b0)
        a0b1 = torch.einsum('bd,bd->b', a0, b1)
        a1b0 = torch.einsum('bd,bd->b', a1, b0)
        a1b1 = torch.einsum('bd,bd->b', a1, b1)

        z = (a0b0 - a1b1) + (a0b1 + a1b0)

        print(z.shape)
        exit(1)
        print(A_head)
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
