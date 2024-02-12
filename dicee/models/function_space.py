from .base_model import BaseKGE
import torch
import numpy as np

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
            print(f"\n\n*****Embedding dimension reset to {self.embedding_dim} to fit model architecture!*****\n")
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
    

class LFMult1(BaseKGE): #this consider the score:  score = <hr,t> = \int Re(hr \bar t), gives similar result as the trilinear score
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
    
class LFMult(BaseKGE): # embedding with polynomials 
    def __init__(self,args):
        super().__init__(args)
        self.name = 'LFMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.x_values = torch.linspace(0, 1, 100)
        self.degree = self.args.get("degree",0)
        self.m = int(self.embedding_dim/(1+self.degree))

    def forward_triples(self, idx_triple): # idx_triplet = (h_idx, r_idx, t_idx) #change this to the forward_triples

        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)

        coeff_head, coeff_rel, coeff_tail = self.construct_multi_coeff(head_ent_emb), self.construct_multi_coeff(rel_emb), self.construct_multi_coeff(tail_ent_emb)


        #score = self.tri_score(coeff_head,coeff_rel,coeff_tail)

        score = self.tri_score(coeff_head,coeff_rel,coeff_tail)
        
        score = score.reshape(-1,self.m).sum(dim=1)
        
        return score 
    
    def construct_multi_coeff(self, x):

        coeffs = torch.hsplit(x,self.degree + 1)
        coeffs = torch.stack(coeffs,dim=1)

        return coeffs.transpose(1,2)




    def tri_score(self, coeff_h, coeff_r, coeff_t):

        '''this part implement the trilinear scoring techniques: 

        score(h,r,t) = \int_{0}{1} h(x)r(x)t(x) dx = \sum_{i,j,k = 0}^{d-1} \dfrac{a_i*b_j*c_k}{1+(i+j+k)%d} 

        1. generate the range for i,j and k from [0 d-1]

        2. perform
        \dfrac{a_i*b_j*c_k}{1+(i+j+k)%d} in parallel for every batch

        3. take the sum over each batch
        
        '''
        
        i_range, j_range, k_range = torch.meshgrid(torch.arange(self.degree+1),torch.arange(self.degree+1),torch.arange(self.degree+1))
        terms = 1 / (1 + i_range + j_range + k_range) 

        
        weighted_terms = terms.unsqueeze(0)*coeff_h.view(-1, 1, self.degree+1, 1) *coeff_r.view(-1, self.degree+1, 1, 1) * coeff_t.view(-1, 1, 1,self.degree+1)
        result = torch.sum(weighted_terms, dim=[-3,-2,-1])
    
        return result
    
    def vtp_score(self, h, r, t):
            
        '''this part implement the vector triple product scoring techniques: 

        score(h,r,t) = \int_{0}{1} h(x)r(x)t(x) dx = \sum_{i,j,k = 0}^{d-1} \dfrac{a_i*c_j*b_k - b_i*c_j*a_k}{(1+(i+j)%d)(1+k)} 

        1. generate the range for i,j and k from [0 d-1]

        2. Compute the first and second terms of the sum

        3.  Multiply with then denominator and take the sum
        
        4. take the sum over each batch
        
        '''
            
        i_range, j_range, k_range = torch.meshgrid(torch.arange(self.embedding_dim),torch.arange(self.embedding_dim),torch.arange(self.embedding_dim))

        # terms = 1 / (1 + (i_range + j_range)%self.embedding_dim) / (1+ k_range) # with modulo

        terms = 1 / (1 + i_range + j_range) / (1+ k_range)   #without dthe modulo


        terms1 = h.view(-1, 1, self.embedding_dim, 1) * t.view(-1, self.embedding_dim, 1, 1) * r.view(-1, 1, 1,self.embedding_dim)
        terms2 = r.view(-1, 1, self.embedding_dim, 1) * t.view(-1, self.embedding_dim, 1, 1) * h.view(-1, 1, 1,self.embedding_dim)

        weighted_terms = terms * (terms1-terms2)
        
        result = torch.sum(weighted_terms, dim=[-3,-2,-1])

        return result

    def comp_func(self,h,r,t): 
        '''this part implement the function composition scoring techniques:'''

        degree = torch.arange(self.embedding_dim, dtype=torch.float32)

        r_emb = self.polynomial(r,self.x_values,degree) 

        t_emb = self.polynomial(t,self.x_values,degree) 

        hor = self.pop(h,r_emb,degree) 
        
        score = torch.trapz(hor*t_emb , self.x_values) #Computing the score with the trapezoid method

        return score

    def polynomial(self,coeff,x,degree):
        '''This function takes a matrix tensor of coefficients (coeff), a tensor vector of points x  and range of integer [0,1,...d]
            and return a vector tensor (coeff[0][0] + coeff[0][1]x +...+ coeff[0][d]x^d,
                                coeff[1][0] + coeff[1][1]x +...+ coeff[1][d]x^d)
                                        ....'''

        x_powers = x.unsqueeze(1) ** degree

        vect = torch.matmul(coeff,x_powers.T)

        return vect

        
    def pop(self,coeff,x,degree):
        '''This function allow us to evaluate the composition of two polynomes without for loops :) 
        it takes a matrix tensor of coefficients (coeff), a matrix tensor of points x  and range of integer [0,1,...d]
            and return a tensor (coeff[0][0] + coeff[0][1]x +...+ coeff[0][d]x^d,
                                coeff[1][0] + coeff[1][1]x +...+ coeff[1][d]x^d)
                                        ....'''
        x_powers = x.unsqueeze(2) ** degree

        Mat = (coeff.unsqueeze(1)*x_powers).sum(dim=-1)

        return Mat