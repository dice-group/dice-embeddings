import torch
from .base_model import BaseKGE


class DualE(BaseKGE):
    """Dual Quaternion Knowledge Graph Embeddings (https://ojs.aaai.org/index.php/AAAI/article/download/16850/16657)"""
    def __init__(self, args):
        super().__init__(args)
        self.name = 'DualE'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.num_ent = self.num_entities


    
    def _omult(self, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, c_0, c_1, c_2, c_3, d_0, d_1, d_2, d_3):
        """Calculate the Dual Hamiltonian product"""

        h_0=a_0*c_0-a_1*c_1-a_2*c_2-a_3*c_3
        h1_0=a_0*d_0+b_0*c_0-a_1*d_1-b_1*c_1-a_2*d_2-b_2*c_2-a_3*d_3-b_3*c_3
        h_1=a_0*c_1+a_1*c_0+a_2*c_3-a_3*c_2
        h1_1=a_0*d_1+b_0*c_1+a_1*d_0+b_1*c_0+a_2*d_3+b_2*c_3-a_3*d_2-b_3*c_2
        h_2=a_0*c_2-a_1*c_3+a_2*c_0+a_3*c_1
        h1_2=a_0*d_2+b_0*c_2-a_1*d_3-b_1*c_3+a_2*d_0+b_2*c_0+a_3*d_1+b_3*c_1
        h_3=a_0*c_3+a_1*c_2-a_2*c_1+a_3*c_0
        h1_3=a_0*d_3+b_0*c_3+a_1*d_2+b_1*c_2-a_2*d_1-b_2*c_1+a_3*d_0+b_3*c_0

        return  (h_0,h_1,h_2,h_3,h1_0,h1_1,h1_2,h1_3)

    
    def _onorm(self,r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
        """Normalization of relationship embedding
        
        Inputs
        --------
        Real and Imaginary parts of the Relation embeddings 
        
        .. math::
        
            W_r = (c,d)
            c = (r_1, r_2, r_3, r_4)
            d = (r_5, r_6, r_7, r_8)

        .. math::

            \bar{d} = d -  \frac{<d,c>}{<c,c>} c
            c' = \frac{c}{\|c\|} = \frac{c_0 + c_1i + c_2j + c_3k}{c_0^2 + c_1^2 + c_2^2 + c_3^2}

        
        Outputs
        --------
        Normalized Real and Imaginary parts of the Relation embeddings

        .. math::

            W_r' = (c', \bar{d})
        """

        denominator_0 = r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
        denominator_1 = torch.sqrt(denominator_0)
        #denominator_2 = torch.sqrt(r_5 ** 2 + r_6 ** 2 + r_7 ** 2 + r_8 ** 2)
        deno_cross = r_5 * r_1 + r_6 * r_2 + r_7 * r_3 + r_8 * r_4

        r_5 = r_5 - deno_cross / denominator_0 * r_1
        r_6 = r_6 - deno_cross / denominator_0 * r_2
        r_7 = r_7 - deno_cross / denominator_0 * r_3
        r_8 = r_8 - deno_cross / denominator_0 * r_4

        r_1 = r_1 / denominator_1
        r_2 = r_2 / denominator_1
        r_3 = r_3 / denominator_1
        r_4 = r_4 / denominator_1
        #r_5 = r_5 / denominator_2
        #r_6 = r_6 / denominator_2
        #r_7 = r_7 / denominator_2
        #r_8 = r_8 / denominator_2
        return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    
    def _calc(self, e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )->torch.tensor:
        
        """Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity ref(Eq.8)
        \phi(h,r,t) = <a'_h, a_t> + <b'_h, b_t> + <c'_h, c_t> + <d'_h, d_t> 
        
        Inputs:
        ----------
        (Tensors) Real and imaginary parts of the head, relation and tail embeddings

        Output: inner product of the head entity and the relationship Hamiltonian product and the tail entity"""

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)


        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   +  o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        return -torch.sum(score_r, -1)
    
    def kvsall_score(self, e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )->torch.tensor:
        """KvsAll scoring function
        
            Input
            ---------
            x: torch.LongTensor with (n, ) shape

            Output
            -------
            torch.FloatTensor with (n) shape
        """

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)


        score_r = torch.mm(o_1, e_1_t) + torch.mm(o_2 ,e_2_t) + torch.mm(o_3, e_3_t) + torch.mm(o_4, e_4_t)\
            +  torch.mm(o_5, e_5_t) + torch.mm(o_6, e_6_t) + torch.mm(o_7, e_7_t) +torch.mm( o_8 , e_8_t)
        
        return -score_r

    
    def forward_triples(self, idx_triple:torch.tensor)-> torch.tensor:
        """Negative Sampling forward pass:

        Input
        ---------
        x: torch.LongTensor with (n, ) shape

        Output
        -------
        torch.FloatTensor with (n) shape
        """

        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)


        e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h = torch.hsplit(head_ent_emb, 8)

        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t = torch.hsplit(tail_ent_emb, 8)
        
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = torch.hsplit(rel_emb, 8)

        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        
       
        return score 
    


    def forward_k_vs_all(self,x):

        """KvsAll forward pass
        
        Input
        ---------
        x: torch.LongTensor with (n, ) shape

        Output
        -------
        torch.FloatTensor with (n) shape
    
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        
        e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h = torch.hsplit(head_ent_emb, 8)

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = torch.hsplit(rel_ent_emb, 8)

        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t = torch.hsplit(self.entity_embeddings.weight, 8)

        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t = self.T(e_1_t), self.T(e_2_t), self.T(e_3_t),\
            self.T(e_4_t), self.T(e_5_t), self.T(e_6_t), self.T(e_7_t), self.T(e_8_t)

        score = self.kvsall_score(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )


        return score
    
    def T(self, x:torch.tensor)->torch.tensor:
        """ Transpose function

            Input: Tensor with shape (nxm)
            Output: Tensor with shape (mxn)"""
      
        return x.transpose(1, 0)


    

