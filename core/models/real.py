import torch

from .base_model import *
import numpy as np
from math import sqrt



class DistMult(BaseKGE):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    https://arxiv.org/abs/1412.6575"""

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'DistMult'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_real = torch.nn.BatchNorm1d(args.embedding_dim)

        self.hidden_dropout = torch.nn.Dropout(args.hidden_dropout_rate)

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        return torch.mm(self.hidden_dropout(self.bn_hidden_real(emb_head_real * emb_rel_real)),
                        self.emb_ent_real.weight.transpose(1, 0))

    def forward_triples(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor, e2_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute score of given triple
        :param e1_idx:
        :param rel_idx:
        :param e2_idx:
        :return:
        """
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))

        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))

        # (1.3) Complex embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx)
        return (self.hidden_dropout(self.bn_hidden_real(emb_head_real * emb_rel_real)) * emb_tail_real).sum(dim=1)






class Shallom(BaseKGE):
    """ A shallow neural model for relation prediction (https://arxiv.org/abs/2101.09090) """

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'Shallom'
        shallom_width = int(args.shallom_width_ratio_of_emb * args.embedding_dim)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.entity_embeddings = nn.Embedding(args.num_entities, args.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        self.shallom = nn.Sequential(nn.Dropout(args.input_dropout_rate),
                                     torch.nn.Linear(args.embedding_dim * 2, shallom_width),
                                     nn.BatchNorm1d(shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(args.hidden_dropout_rate),
                                     torch.nn.Linear(shallom_width, args.num_relations))

    def get_embeddings(self) -> Tuple[np.ndarray, None]:
        return self.entity_embeddings.weight.data.detach().numpy(), None

    def forward_k_vs_all(self, e1_idx, e2_idx):
        emb_s, emb_o = self.entity_embeddings(e1_idx), self.entity_embeddings(e2_idx)
        return self.shallom(torch.cat((emb_s, emb_o), 1))

""" On going works"""
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
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
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
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def construct_entity_embeddings(self, e1_idx: torch.Tensor):
        emb_head = self.bn_ent_real(self.emb_ent_real(e1_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_head, emb_head).flatten(1)

    def construct_relation_embeddings(self, rel_idx):
        emb_rel = self.bn_rel_real(self.emb_rel_real(rel_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_rel, emb_rel).flatten(1)

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
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
        super().__init__(args.learning_rate)
        self.name = 'KronELinear'
        self.loss = torch.nn.BCEWithLogitsLoss()
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
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def construct_entity_embeddings(self, e1_idx: torch.Tensor):
        emb_head = self.bn_ent_real(self.emb_ent_real(e1_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_head, emb_head).flatten(1)

    def construct_relation_embeddings(self, rel_idx):
        emb_rel = self.bn_rel_real(self.emb_rel_real(rel_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_rel, emb_rel).flatten(1)

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
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

