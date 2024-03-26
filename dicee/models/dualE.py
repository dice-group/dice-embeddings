import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from .base_model import BaseKGE
import numpy as np
from numpy.random import RandomState



# class OMult(BaseKGE):
#     def __init__(self, args):
#         super().__init__(args)
#         self.name = 'OMult'

#     @staticmethod
#     def octonion_normalizer(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6,
#                             emb_rel_e7):
#         denominator = torch.sqrt(
#             emb_rel_e0 ** 2 + emb_rel_e1 ** 2 + emb_rel_e2 ** 2 + emb_rel_e3 ** 2 + emb_rel_e4 ** 2
#             + emb_rel_e5 ** 2 + emb_rel_e6 ** 2 + emb_rel_e7 ** 2)
#         y0 = emb_rel_e0 / denominator
#         y1 = emb_rel_e1 / denominator
#         y2 = emb_rel_e2 / denominator
#         y3 = emb_rel_e3 / denominator
#         y4 = emb_rel_e4 / denominator
#         y5 = emb_rel_e5 / denominator
#         y6 = emb_rel_e6 / denominator
#         y7 = emb_rel_e7 / denominator
#         return y0, y1, y2, y3, y4, y5, y6, y7

#     def score(self, head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor, tail_ent_emb: torch.FloatTensor):
#         # (2) Split (1) into real and imaginary parts.
#         emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7 = torch.hsplit(
#             head_ent_emb, 8)
#         emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = torch.hsplit(
#             rel_ent_emb,
#             8)
#         if isinstance(self.normalize_relation_embeddings, IdentityClass):
#             (emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
#              emb_rel_e5, emb_rel_e6, emb_rel_e7) = self.octonion_normalizer(emb_rel_e0,
#                                                                             emb_rel_e1, emb_rel_e2, emb_rel_e3,
#                                                                             emb_rel_e4, emb_rel_e5, emb_rel_e6,
#                                                                             emb_rel_e7)

#         emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 = torch.hsplit(
#             tail_ent_emb, 8)
#         # (3) Octonion Multiplication
#         e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
#             O_1=(
#                 emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
#             O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
#         # (4)
#         # (4.3) Inner product
#         e0_score = (e0 * emb_tail_e0).sum(dim=1)
#         e1_score = (e1 * emb_tail_e1).sum(dim=1)
#         e2_score = (e2 * emb_tail_e2).sum(dim=1)
#         e3_score = (e3 * emb_tail_e3).sum(dim=1)
#         e4_score = (e4 * emb_tail_e4).sum(dim=1)
#         e5_score = (e5 * emb_tail_e5).sum(dim=1)
#         e6_score = (e6 * emb_tail_e6).sum(dim=1)
#         e7_score = (e7 * emb_tail_e7).sum(dim=1)

#         return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score



class DualE(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'DualE'
        self.lmbda = 0.0

        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)

        # self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        # self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)

        # self.emb_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_3 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_4 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_5 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_6 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_7 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.emb_8 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.rel_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_3 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_4 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_5 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_6 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_7 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_8 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.criterion = nn.Softplus()
        # self.fc = nn.Linear(100, 50, bias=False)
        # self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        # self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        # self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)

        # self.init_weights()

    def init_weights(self):
        if True:
            r, i, j, k,r_1,i_1,j_1,k_1 = self.quaternion_init(self.config.entTotal, self.config.hidden_size)
            r, i, j, k,r_1,i_1,j_1,k_1 = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k),\
                                        torch.from_numpy(r_1), torch.from_numpy(i_1), torch.from_numpy(j_1), torch.from_numpy(k_1)
            self.emb_1.weight.data = r.type_as(self.emb_1.weight.data)
            self.emb_2.weight.data = i.type_as(self.emb_2.weight.data)
            self.emb_3.weight.data = j.type_as(self.emb_3.weight.data)
            self.emb_4.weight.data = k.type_as(self.emb_4.weight.data)
            self.emb_5.weight.data = r_1.type_as(self.emb_5.weight.data)
            self.emb_6.weight.data = i_1.type_as(self.emb_6.weight.data)
            self.emb_7.weight.data = j_1.type_as(self.emb_7.weight.data)
            self.emb_8.weight.data = k_1.type_as(self.emb_8.weight.data)

            s, x, y, z,s_1,x_1,y_1,z_1 = self.quaternion_init(self.config.entTotal, self.config.hidden_size)
            s, x, y, z,s_1,x_1,y_1,z_1 = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z), \
                                        torch.from_numpy(s_1), torch.from_numpy(x_1), torch.from_numpy(y_1), torch.from_numpy(z_1)
            self.rel_1.weight.data = s.type_as(self.rel_1.weight.data)
            self.rel_2.weight.data = x.type_as(self.rel_2.weight.data)
            self.rel_3.weight.data = y.type_as(self.rel_3.weight.data)
            self.rel_4.weight.data = z.type_as(self.rel_4.weight.data)
            self.rel_5.weight.data = s_1.type_as(self.rel_5.weight.data)
            self.rel_6.weight.data = x_1.type_as(self.rel_6.weight.data)
            self.rel_7.weight.data = y_1.type_as(self.rel_7.weight.data)
            self.rel_8.weight.data = z_1.type_as(self.rel_8.weight.data)
            nn.init.xavier_uniform_(self.rel_w.weight.data)
        else:
            nn.init.xavier_uniform_(self.emb_1.weight.data)
            nn.init.xavier_uniform_(self.emb_2.weight.data)
            nn.init.xavier_uniform_(self.emb_3.weight.data)
            nn.init.xavier_uniform_(self.emb_4.weight.data)
            nn.init.xavier_uniform_(self.emb_5.weight.data)
            nn.init.xavier_uniform_(self.emb_6.weight.data)
            nn.init.xavier_uniform_(self.emb_7.weight.data)
            nn.init.xavier_uniform_(self.emb_8.weight.data)
            nn.init.xavier_uniform_(self.rel_1.weight.data)
            nn.init.xavier_uniform_(self.rel_2.weight.data)
            nn.init.xavier_uniform_(self.rel_3.weight.data)
            nn.init.xavier_uniform_(self.rel_4.weight.data)
            nn.init.xavier_uniform_(self.rel_5.weight.data)
            nn.init.xavier_uniform_(self.rel_6.weight.data)
            nn.init.xavier_uniform_(self.rel_7.weight.data)
            nn.init.xavier_uniform_(self.rel_8.weight.data)



    #Calculate the Dual Hamiltonian product
    def _omult(self, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, c_0, c_1, c_2, c_3, d_0, d_1, d_2, d_3):

        h_0=a_0*c_0-a_1*c_1-a_2*c_2-a_3*c_3
        h1_0=a_0*d_0+b_0*c_0-a_1*d_1-b_1*c_1-a_2*d_2-b_2*c_2-a_3*d_3-b_3*c_3
        h_1=a_0*c_1+a_1*c_0+a_2*c_3-a_3*c_2
        h1_1=a_0*d_1+b_0*c_1+a_1*d_0+b_1*c_0+a_2*d_3+b_2*c_3-a_3*d_2-b_3*c_2
        h_2=a_0*c_2-a_1*c_3+a_2*c_0+a_3*c_1
        h1_2=a_0*d_2+b_0*c_2-a_1*d_3-b_1*c_3+a_2*d_0+b_2*c_0+a_3*d_1+b_3*c_1
        h_3=a_0*c_3+a_1*c_2-a_2*c_1+a_3*c_0
        h1_3=a_0*d_3+b_0*c_3+a_1*d_2+b_1*c_2-a_2*d_1-b_2*c_1+a_3*d_0+b_3*c_0

        return  (h_0,h_1,h_2,h_3,h1_0,h1_1,h1_2,h1_3)

    #Normalization of relationship embedding
    def _onorm(self,r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
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

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 ):

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)


        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   +  o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        return -torch.sum(score_r, -1)

    

    # def loss(self, score, regul, regul2):
    #     return (
    #         torch.mean(self.criterion(score * self.batch_y)) + self.lmbda * regul + self.lmbda * regul2
    #     )

    def forward_triples(self, idx_triple):

        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)

        e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h = torch.hsplit(head_ent_emb, 8)
        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t = torch.hsplit(tail_ent_emb, 8)
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = torch.hsplit(rel_emb, 8)

       

        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        
        regul = (torch.mean(torch.abs(e_1_h) ** 2)
                 + torch.mean(torch.abs(e_2_h) ** 2)
                 + torch.mean(torch.abs(e_3_h) ** 2)
                 + torch.mean(torch.abs(e_4_h) ** 2)
                 + torch.mean(torch.abs(e_5_h) ** 2)
                 + torch.mean(torch.abs(e_6_h) ** 2)
                 + torch.mean(torch.abs(e_7_h) ** 2)
                 + torch.mean(torch.abs(e_8_h) ** 2)
                 + torch.mean(torch.abs(e_1_t) ** 2)
                 + torch.mean(torch.abs(e_2_t) ** 2)
                 + torch.mean(torch.abs(e_3_t) ** 2)
                 + torch.mean(torch.abs(e_4_t) ** 2)
                 + torch.mean(torch.abs(e_5_t) ** 2)
                 + torch.mean(torch.abs(e_6_t) ** 2)
                 + torch.mean(torch.abs(e_7_t) ** 2)
                 + torch.mean(torch.abs(e_8_t) ** 2)
                 )
        regul2 = (torch.mean(torch.abs(r_1) ** 2)
                  + torch.mean(torch.abs(r_2) ** 2)
                  + torch.mean(torch.abs(r_3) ** 2)
                  + torch.mean(torch.abs(r_4) ** 2)
                  + torch.mean(torch.abs(r_5) ** 2)
                  + torch.mean(torch.abs(r_6) ** 2)
                  + torch.mean(torch.abs(r_7) ** 2)
                  + torch.mean(torch.abs(r_8) ** 2))

        return score #self.loss(score, regul, regul2)

    def predict(self):
        e_1_h = self.emb_1(self.batch_h)
        e_2_h = self.emb_2(self.batch_h)
        e_3_h = self.emb_3(self.batch_h)
        e_4_h = self.emb_4(self.batch_h)
        e_5_h = self.emb_5(self.batch_h)
        e_6_h = self.emb_6(self.batch_h)
        e_7_h = self.emb_7(self.batch_h)
        e_8_h = self.emb_8(self.batch_h)

        e_1_t = self.emb_1(self.batch_t)
        e_2_t = self.emb_2(self.batch_t)
        e_3_t = self.emb_3(self.batch_t)
        e_4_t = self.emb_4(self.batch_t)
        e_5_t = self.emb_5(self.batch_t)
        e_6_t = self.emb_6(self.batch_t)
        e_7_t = self.emb_7(self.batch_t)
        e_8_t = self.emb_8(self.batch_t)

        r_1 = self.rel_1(self.batch_r)
        r_2 = self.rel_2(self.batch_r)
        r_3 = self.rel_3(self.batch_r)
        r_4 = self.rel_4(self.batch_r)
        r_5 = self.rel_5(self.batch_r)
        r_6 = self.rel_6(self.batch_r)
        r_7 = self.rel_7(self.batch_r)
        r_8 = self.rel_8(self.batch_r)

        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        return score.cpu().data.numpy()




    def quaternion_init(self, in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(2020)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)


        # Calculate the three parts about t
        kernel_shape1 = (in_features, out_features)
        number_of_weights1 = np.prod(kernel_shape1)
        t_i = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_j = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_k = np.random.uniform(0.0, 1.0, number_of_weights1)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights1):
            norm1 = np.sqrt(t_i[i] ** 2 + t_j[i] ** 2 + t_k[i] ** 2) + 0.0001
            t_i[i] /= norm1
            t_j[i] /= norm1
            t_k[i] /= norm1
        t_i = t_i.reshape(kernel_shape1)
        t_j = t_j.reshape(kernel_shape1)
        t_k = t_k.reshape(kernel_shape1)
        tmp_t = rng.uniform(low=-s, high=s, size=kernel_shape1)


        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        phase1 = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape1)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        wt_i = tmp_t * t_i * np.sin(phase1)
        wt_j = tmp_t * t_j * np.sin(phase1)
        wt_k = tmp_t * t_k * np.sin(phase1)

        i_0=weight_r
        i_1=weight_i
        i_2=weight_j
        i_3=weight_k
        i_4=(-wt_i*weight_i-wt_j*weight_j-wt_k*weight_k)/2
        i_5=(wt_i*weight_r+wt_j*weight_k-wt_k*weight_j)/2
        i_6=(-wt_i*weight_k+wt_j*weight_r+wt_k*weight_i)/2
        i_7=(wt_i*weight_j-wt_j*weight_i+wt_k*weight_r)/2


        return (i_0,i_1,i_2,i_3,i_4,i_5,i_6,i_7)
