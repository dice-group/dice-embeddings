from .base_model import BaseKGE
import torch
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
        # TODO:Do we need coefficients for the real part ?
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

    def construct_batch_selected_cl_multivector(self, x: torch.FloatTensor, r: int, p: int, q: int) -> tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Construct a batch of batchs multivectors Cl_{p,q}(\mathbb{R}^d)

        Parameter
        ---------
        x: torch.FloatTensor with (n,k, d) shape

        Returns
        -------
        a0: torch.FloatTensor with (n,k, m) shape
        ap: torch.FloatTensor with (n,k, m, p) shape
        aq: torch.FloatTensor with (n,k, m, q) shape
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


class CKeci(Keci):
    " Without learning dimension scaling"

    def __init__(self, args):
        super().__init__(args)
        self.name = 'CKeci'
        self.requires_grad_for_interactions = False
        print(f'r:{self.r}\t p:{self.p}\t q:{self.q}')
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
