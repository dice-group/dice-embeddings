import torch

from .base_model import *

from math import sqrt


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


def kronecker_linear_transformation(A, B, x):
    """
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

    return


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

    def get_embeddings(self):
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


class KPDistMult(BaseKGE):

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

    def get_embeddings(self):
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
    """

    """

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KronE'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings # must have valid root
        # (1) Initialize embeddings
        self.embedding_dim = args.embedding_dim
        self.embedding_dim_rel = args.embedding_dim

        self.emb_ent_real = nn.Embedding(args.num_entities, self.embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.embedding_dim_rel)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        # (2) Initialize Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(args.hidden_dropout_rate)

        # (3) Initialize Batch Norms
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim_rel)
        self.bn_hidden_1 = torch.nn.BatchNorm1d(int(self.embedding_dim * self.embedding_dim_rel))
        self.bn_hidden_2 = torch.nn.BatchNorm1d(self.embedding_dim)

        # Linear transformation W is a by m by n matrix ,
        # where n is the kronecker product of h and r
        self.m = self.embedding_dim
        self.n = int(self.embedding_dim * self.embedding_dim_rel)

        self.m1, self.n1 = self.m, self.n // self.m
        self.A = nn.parameter.Parameter(torch.randn(self.m1, self.n1, requires_grad=True))

        self.m2, self.n2 = self.m // self.m1, self.n // self.n1
        self.B = nn.parameter.Parameter(torch.randn(self.m2, self.n2, requires_grad=True))
        print(f'Linear trans : {self.m1 * self.m2}  {self.n1 * self.n2}')

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
        # (1) Retrieve  head entity embeddings and apply BN + DP
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (2) Retrieve  relation embeddings and apply kronecker_product
        feature = batch_kronecker_product(emb_head_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        feature = self.bn_hidden_1(feature)
        # (3) Reshape
        feature = feature.reshape(len(feature), self.n2, self.n1)
        # (3.1)
        feature = torch.relu((torch.matmul(feature.transpose(1, 2), self.B.T).transpose(1, 2) @ self.A).flatten(1))
        # feature = (torch.matmul(feature.transpose(1, 2), self.B.T).transpose(1, 2) @ self.A).flatten(1)
        return torch.mm(self.hidden_dropout(self.bn_hidden_2(feature)), self.emb_ent_real.weight.transpose(1, 0))


class oldKronE(BaseKGE):
    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KronE'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings
        self.entity_embedding_dim = args.entity_embedding_dim
        self.rel_embedding_dim = args.rel_embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, self.entity_embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.rel_embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        self.normalizer = torch.nn.BatchNorm1d  # or nn.LayerNorm
        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = self.normalizer(self.entity_embedding_dim)
        self.bn_rel_real = self.normalizer(self.rel_embedding_dim)
        # (2) With additional parameters
        self.down_features = nn.Sequential(self.normalizer(self.entity_embedding_dim * self.rel_embedding_dim),
                                           nn.Linear(in_features=self.entity_embedding_dim * self.rel_embedding_dim,
                                                     out_features=self.entity_embedding_dim),
                                           nn.ReLU(),
                                           self.normalizer(self.entity_embedding_dim),
                                           torch.nn.Dropout(args.hidden_dropout_rate))

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    @staticmethod
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

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (2) KronE product of head and relation
        features = self.batch_kronecker_product(emb_head_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        features = self.down_features(features)
        return torch.mm(features, self.emb_ent_real.weight.transpose(1, 0))

    def forward_triples(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor, e2_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute score of given triple
        :param e1_idx:
        :param rel_idx:
        :param e2_idx:
        :return:
        """
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (2) KronE product of head and relation
        features = self.batch_kronecker_product(emb_head_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        features = self.down_features(features)

        # (1.3) Complex embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx)
        return (features * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)


class KronE_wo_f(BaseKGE):
    """
    non stands for without non linearity/ReLU
    """

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KronE_wo_f'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings
        self.entity_embedding_dim = args.entity_embedding_dim
        self.rel_embedding_dim = args.rel_embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, self.entity_embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.rel_embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        self.normalizer = torch.nn.BatchNorm1d  # or nn.LayerNorm
        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = self.normalizer(self.entity_embedding_dim)
        self.bn_rel_real = self.normalizer(self.rel_embedding_dim)
        # (2) With additional parameters
        self.down_features = nn.Sequential(self.normalizer(self.entity_embedding_dim * self.rel_embedding_dim),
                                           nn.Linear(in_features=self.entity_embedding_dim * self.rel_embedding_dim,
                                                     out_features=self.entity_embedding_dim),
                                           self.normalizer(self.entity_embedding_dim),
                                           torch.nn.Dropout(args.hidden_dropout_rate))

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    @staticmethod
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

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (2) KronE product of head and relation
        features = self.batch_kronecker_product(emb_head_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        features = self.down_features(features)
        return torch.mm(features, self.emb_ent_real.weight.transpose(1, 0))

    def forward_triples(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor, e2_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute score of given triple
        :param e1_idx:
        :param rel_idx:
        :param e2_idx:
        :return:
        """
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (2) KronE product of head and relation
        features = self.batch_kronecker_product(emb_head_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        features = self.down_features(features)

        # (1.3) Complex embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx)
        return (features * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)


class newKronE(BaseKGE):
    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'BaseKronE'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings
        self.entity_embedding_dim = args.entity_embedding_dim
        self.rel_embedding_dim = args.rel_embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, self.entity_embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.rel_embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        self.normalizer = torch.nn.BatchNorm1d  # or nn.LayerNorm
        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = self.normalizer(self.entity_embedding_dim)
        self.bn_ent_tail_real = self.normalizer(self.entity_embedding_dim)
        self.bn_rel_real = self.normalizer(self.rel_embedding_dim)

        self.bn_hidden_real = self.normalizer(self.entity_embedding_dim * self.entity_embedding_dim)
        # (2) With additional parameters

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    @staticmethod
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

    def forward_triples(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor, e2_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute score of given triple
        :param e1_idx:
        :param rel_idx:
        :param e2_idx:
        :return:
        """
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))

        features = emb_head_real * emb_rel_real

        # (2) KronE product of head and relation
        features = self.batch_kronecker_product(emb_head_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        features = self.bn_hidden_real(features)
        # MAybe Kronecker sum ?
        features = self.batch_kronecker_product(features.unsqueeze(1),
                                                self.bn_ent_tail_real(self.emb_ent_real(e2_idx)).unsqueeze(1)).flatten(
            1)
        return features.sum(dim=1)


class Shallom(BaseKGE):
    """
    A shallow neural model for relation prediction (https://arxiv.org/abs/2101.09090)
    """

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

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.detach().numpy()

    def forward_k_vs_all(self, e1_idx, e2_idx):
        emb_s, emb_o = self.entity_embeddings(e1_idx), self.entity_embeddings(e2_idx)
        return self.shallom(torch.cat((emb_s, emb_o), 1))

    def forward_triples(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor, e2_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute score of given triple
        :param e1_idx:
        :param rel_idx:
        :param e2_idx:
        :return:
        """

        emb_s, emb_o = self.entity_embeddings(e1_idx), self.entity_embeddings(e2_idx)
        scores = self.shallom(torch.cat((emb_s, emb_o), 1))
        print(e1_idx.shape)
        print(rel_idx.shape)
        print(e2_idx.shape)
        print(scores.shape)
        raise ValueError
        exit(1)
