import torch
from .base_model import *
import numpy as np
from math import sqrt
from .static_funcs import quaternion_mul


class DistMult(BaseKGE):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    https://arxiv.org/abs/1412.6575"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'DistMult'
        # Adding this reduces performance in training and generalization
        self.hidden_normalizer = lambda x: x

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Compute the score
        return (self.hidden_dropout(self.hidden_normalizer(head_ent_emb * rel_ent_emb)) * tail_ent_emb).sum(dim=1)

    def forward_k_vs_all(self, x: torch.Tensor):
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_head_real * emb_rel_real)),
                        self.entity_embeddings.weight.transpose(1, 0))


class TransE(BaseKGE):
    """
    Translating Embeddings for Modeling
    Multi-relational Data
    https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'TransE'
        # Adding this reduces performance in training and generalization
        self.hidden_normalizer = lambda x: x
        self.loss = torch.nn.BCELoss()
        self._norm = 2
        self.margin = 1

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # Original || s+p - t|| true label > 0 distance, false label
        # Update: 1 - sigmoid(|| s+p -t ||) to work with BCE
        distance = torch.nn.functional.pairwise_distance(head_ent_emb + rel_ent_emb, tail_ent_emb, p=self._norm)
        scores = torch.sigmoid(distance + self.margin)
        return scores

    def forward_k_vs_all(self, x: torch.Tensor):
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        distance = torch.nn.functional.pairwise_distance(torch.unsqueeze(emb_head_real + emb_rel_real, 1), self.entity_embeddings.weight, p=self._norm)
        scores = torch.sigmoid(distance + self.margin)
        return scores


class Shallom(BaseKGE):
    """ A shallow neural model for relation prediction (https://arxiv.org/abs/2101.09090) """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Shallom'
        # Fixed
        shallom_width = int(2*self.embedding_dim)
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        self.shallom = nn.Sequential(nn.Dropout(self.input_dropout_rate),
                                     torch.nn.Linear(self.embedding_dim * 2, shallom_width),
                                     self.normalizer_class(shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(self.hidden_dropout_rate),
                                     torch.nn.Linear(shallom_width, self.num_relations))

    def get_embeddings(self) -> Tuple[np.ndarray, None]:
        return self.entity_embeddings.weight.data.detach(), None

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, e2_idx = x[:, 0], x[:, 1]
        emb_s, emb_o = self.entity_embeddings(e1_idx), self.entity_embeddings(e2_idx)
        return self.shallom(torch.cat((emb_s, emb_o), 1))

    def forward_triples(self, x):
        """

        :param x:
        :return:
        """

        n, d = x.shape
        assert d == 3
        scores_for_all_relations = self.forward_k_vs_all(x[:, [0, 2]])
        return scores_for_all_relations[:, x[:, 1]].flatten()


""" On going works"""

class CLf(BaseKGE):
    """Clifford:Embedding Space Search in Clifford Algebras"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'CLf'
        # Adding this reduces performance in training and generalization
        self.hidden_normalizer = lambda x: x

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)

        # (2) Formula for CL_{p,1}(\mathbb R) =>  a + \sum_i ^p b_i v_i + \sum_j ^q c_j u_j.
        # (3) a + bv + cu provided that p= 1 and q=1.
        # (4) Head embedding representation in CL (a + bv + cu)
        a, b, c = torch.hsplit(head_ent_emb, 3)
        # (5) Relation embedding representation in CL (a' + b'v + c'u).
        a_prime, b_prime, c_prime = torch.hsplit(rel_ent_emb, 3)
        # (6) Tail embedding representation in CL (a''' + b'''v + c'''u).
        a_3prime, b_3prime, c_3prime = torch.hsplit(tail_ent_emb, 3)
        # (7) Scoring function.
        score_vec = a_3prime * ((a * a_prime + b * b_prime) - (c * c_prime)) + b_3prime * (
                    a * b_prime + a_prime * b) + c_3prime * (a * c_prime + a_prime * c)
        return score_vec.sum(dim=1)

    def forward_k_vs_all(self, x: torch.Tensor):
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        print('Hello')
        raise NotImplementedError('Implement scoring function for KvsAll')

class DimAdaptiveDistMult(BaseKGE):

    def __init__(self, args):
        super().__init__(args)
        self.name = 'AdaptiveDistMult'
        # Init Embeddings
        self.current_embedding_dim = 1
        self.emb_ent_real = nn.Embedding(self.num_entities, self.current_embedding_dim)
        self.emb_rel_real = nn.Embedding(self.num_relations, self.current_embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        self.losses = []
        self.moving_average = 0
        self.moving_average_interval = 10
        self.add_dim_size = 1

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def forward_k_vs_all(self, x: torch.Tensor):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        # (1.2) Real embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        return torch.mm(emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1], x[:, 2]
        # (1)
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_tail_real = self.emb_ent_real(e2_idx)
        return (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)

    def training_epoch_end(self, training_step_outputs):

        if self.current_embedding_dim + self.add_dim_size < self.embedding_dim:
            epoch_loss = float(training_step_outputs[0]['loss'].detach())
            self.losses.append(epoch_loss)
            if len(self.losses) % self.moving_average_interval == 0:
                moving_average = sum(self.losses) / len(self.losses)
                self.losses.clear()
                diff = abs(moving_average - epoch_loss)

                if diff > epoch_loss * .1:
                    # do nothing
                    pass
                else:

                    """

                    # Either increase the embedding size or the multiplication
                    print('\nDouble the embedding size') 
                    # Leads to inferious results
                    x = nn.Embedding(self.num_entities, self.add_dim_size)
                    xavier_normal_(x.weight.data)
                    self.emb_ent_real.weight = nn.Parameter(
                        torch.cat((self.emb_ent_real.weight.detach(), x.weight.detach()), dim=1).data,
                        requires_grad=True)
                    x = nn.Embedding(self.num_relations, self.add_dim_size)
                    xavier_normal_(x.weight.data)
                    self.emb_rel_real.weight = nn.Parameter(
                        torch.cat((self.emb_rel_real.weight.detach(), x.weight.detach()), dim=1).data,
                        requires_grad=True)
                    del x
                    self.current_embedding_dim += self.add_dim_size
                    """


class KPDistMult(BaseKGE):
    """
    Named as KD-Rel-DistMult  in our paper
    """

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KPDistMult'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings # must have valid root
        # (1) Initialize embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, int(sqrt(args.embedding_dim)))
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        # (2) Initialize Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(args.hidden_dropout_rate)

        # (3) Initialize Batch Norms
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_real = torch.nn.BatchNorm1d(args.embedding_dim)

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]

        # (1) Retrieve  head entity embeddings and apply BN + DP
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        emb_rel_real = self.emb_rel_real(rel_idx)
        # (2) Retrieve  relation embeddings and apply kronecker_product
        emb_rel_real = batch_kronecker_product(emb_rel_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        # (3) Apply BN + DP on (2)
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(emb_rel_real))
        # (4) Compute scores
        return torch.mm(self.hidden_dropout(self.bn_hidden_real(emb_head_real * emb_rel_real)),
                        self.emb_ent_real.weight.transpose(1, 0))


class KronE(BaseKGE):
    """ Kronecker Decomposition applied on Entitiy and Relation Embedding matrices KP-DistMult """

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KronE'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings # must have valid root
        # (1) Initialize embeddings
        self.embedding_dim = int(sqrt(args.embedding_dim))
        self.embedding_dim_rel = int(sqrt(args.embedding_dim))
        self.emb_ent_real = nn.Embedding(args.num_entities, self.embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.embedding_dim_rel)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        # (2) Initialize Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        """
        # Linear transformation W is a by m by n matrix ,
        # where n is the kronecker product of h and r
        self.m = self.embedding_dim
        self.n = int((self.embedding_dim * self.embedding_dim_rel))
        # (2) With additional parameters
        self.m1, self.n1 = self.m, self.n // self.m
        self.A = nn.parameter.Parameter(torch.randn(self.m1, self.n1, requires_grad=True))

        self.m2, self.n2 = self.m // self.m1, self.n // self.n1
        self.B = nn.parameter.Parameter(torch.randn(self.m2, self.n2, requires_grad=True))
        """

        # (3) Initialize Batch Norms
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim_rel)

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def construct_entity_embeddings(self, e1_idx: torch.Tensor):
        emb_head = self.bn_ent_real(self.emb_ent_real(e1_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_head, emb_head).flatten(1)

    def construct_relation_embeddings(self, rel_idx):
        emb_rel = self.bn_rel_real(self.emb_rel_real(rel_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_rel, emb_rel).flatten(1)

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_head_real = self.input_dp_ent_real(self.construct_entity_embeddings(e1_idx))
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_rel_real = self.input_dp_rel_real(self.construct_relation_embeddings(rel_idx))
        # (3)
        # (3.1) Capture interactions via Hadamard Product (1) and (2);
        feature = emb_head_real * emb_rel_real
        n, dim = feature.shape
        n_rows = dim // self.embedding_dim
        feature = feature.reshape(n, n_rows, self.embedding_dim)
        # (6) Compute sum of logics Logits
        logits = torch.matmul(feature, self.emb_ent_real.weight.transpose(1, 0)).sum(dim=1)
        return logits


class KronELinear(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'KronELinear'
        # Init Embeddings # must have valid root
        # (1) Initialize embeddings
        self.entity_embedding_dim = args.embedding_dim
        self.rel_embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, self.entity_embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.rel_embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        # (2) Initialize Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)

        # Linear transformation W is a by mp by nq matrix
        # where
        # mp is the kronecker product of h and r and
        # nq is the entity_embedding_dim
        # W: X \otimes Z : W mp by nq
        # X m1 by n1
        # Z mp/m1 by nq/n1

        # output features
        mp = self.entity_embedding_dim
        # Input features
        nq = int((self.entity_embedding_dim ** 2))
        # (2) With additional parameters
        self.m1, self.n1 = mp // 4, nq // 4
        self.X = nn.parameter.Parameter(torch.randn(self.m1, self.n1, requires_grad=True))

        self.m2, self.n2 = mp // self.m1, nq // self.n1
        self.Z = nn.parameter.Parameter(torch.randn(self.m2, self.n2, requires_grad=True))

        # (3) Initialize Batch Norms
        self.bn_ent_real = torch.nn.BatchNorm1d(self.entity_embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.rel_embedding_dim)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def construct_entity_embeddings(self, e1_idx: torch.Tensor):
        emb_head = self.bn_ent_real(self.emb_ent_real(e1_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_head, emb_head).flatten(1)

    def construct_relation_embeddings(self, rel_idx):
        emb_rel = self.bn_rel_real(self.emb_rel_real(rel_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_rel, emb_rel).flatten(1)

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_head_real = self.input_dp_ent_real(self.construct_entity_embeddings(e1_idx))
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_rel_real = self.input_dp_rel_real(self.construct_relation_embeddings(rel_idx))
        # (3)
        # (3.1) Capture interactions via Hadamard Product (1) and (2);
        feature = emb_head_real + emb_rel_real
        feature = kronecker_linear_transformation(self.X, self.Z, feature)
        # (6) Compute sum of logics Logits
        logits = torch.matmul(feature, self.emb_ent_real.weight.transpose(1, 0))
        return logits


def batch_kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    res = res.reshape(siz0 + siz1)
    return res


def kronecker_linear_transformation(X, Z, x):
    """
    W:X\otimes Z: mp by nq matrix
      X :m1 by n1
      Z : mp/m1 by nq/n1

    1) R(x) nq/n1 by n1 matrix
    2) Z (1)
    Let a linear transformation defined by $W\ in R^{mp\times nq}$
    Let a matrix $A \in \mathbb{R}^{m_1  \times n_1} $ and
    a matrix $ B \in \mathbb{R}^{ \frac{mp}{m_1} \times \frac{nq}{n_1}}$.

    (A\otimes B)x=\mathcal{V}(B \; \mathcal{R}_{\frac{n}{n_1} \times n_1}(x) A^\top), \label{Eq:kronecker}
    \end{equation}
    where
    \begin{enumerate}
        \item $x \in \mathbb{R}^n$ represent input feature vector,
        \item $\mathcal{V}$ transforms a matrix to a vector by stacking its columns,
        \item $ \mathcal{R}_{ \frac{n}{n_1} \times n_1} $
        converts x to a $\frac{n}{n_1}$ by $n_1$ matrix by dividing the vector to columns of size $\frac{n}{n_1}$
        and concatenating the resulting columns together
    For more details, please see this wonderful paper
    KroneckerBERT: Learning Kronecker Decomposition for Pre-trained Language Models via Knowledge Distillation

    :type A: torch.Tensor
    :type B: torch.Tensor
    :type x: torch.Tensor

    :rtype: torch.Tensor
    """
    m1, n1 = X.shape
    mp_div_m1, nq_div_n1 = Z.shape
    n, dim = x.shape

    x = x.reshape(n, n1, nq_div_n1)  # x tranpose for the batch computation
    Zx = torch.matmul(x, Z).transpose(1, 2)
    out = torch.matmul(Zx, X.T)
    return out.flatten(1)
