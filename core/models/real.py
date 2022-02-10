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
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real

        self.emb_rel_real = nn.Embedding(args.num_relations, int(sqrt(args.embedding_dim)))  # real
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


class KPFullDistMult(BaseKGE):

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KPFullDistMult'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings
        self.embedding_dim = args.embedding_dim
        assert self.embedding_dim % 2 == 0
        self.m = self.embedding_dim // 2
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_tail_real = torch.nn.Dropout(args.input_dropout_rate)

        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_tail_real = torch.nn.BatchNorm1d(args.embedding_dim)

        # Batch Normalization
        self.bn_kp_exand_ent_real = torch.nn.BatchNorm1d(self.m ** 2)
        self.bn_kp_exand_rel_real = torch.nn.BatchNorm1d(self.m ** 2)
        self.bn_kp_exand_tail_real = torch.nn.BatchNorm1d(self.m ** 2)

        self.bn_final_a = torch.nn.BatchNorm1d(self.m ** 2)
        self.bn_final_b = torch.nn.BatchNorm1d(self.m ** 2)
        self.bn_final_c = torch.nn.BatchNorm1d(self.m ** 2)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward_triples(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor, e2_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute score of given triple
        :param e1_idx:
        :param rel_idx:
        :param e2_idx:
        :return:
        """
        # (1) Get head entity and apply BN + DP
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (2) Get relation entity and apply BN + DP
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (3) Get tail entity and apply BN + DP
        emb_tail_real = self.input_dp_tail_real(self.bn_tail_real(self.emb_ent_real(e2_idx)))
        # We used decompressed embeddings at testing time and got terrible results :)
        # if self.training is False: =>             return (emb_head_real*emb_rel_real*emb_tail_real).sum(dim=1)

        # (4) Kronecker Product Expansion on (1)
        emb_head_real = self.bn_kp_exand_ent_real(batch_kronecker_product(emb_head_real[:, :self.m].unsqueeze(1),
                                                                          emb_head_real[:, self.m:].unsqueeze(
                                                                              1)).flatten(
            1))
        # (2) Kronecker Product Expansion on (1)
        emb_rel_real = self.bn_kp_exand_rel_real(batch_kronecker_product(emb_rel_real[:, :self.m].unsqueeze(1),
                                                                         emb_rel_real[:, self.m:].unsqueeze(1)).flatten(
            1))

        # (2) Get relation embeddings and decompress it by taking the half
        emb_tail_real = self.bn_kp_exand_tail_real(batch_kronecker_product(emb_tail_real[:, :self.m].unsqueeze(1),
                                                                           emb_tail_real[:, self.m:].unsqueeze(
                                                                               1)).flatten(
            1))

        s = self.bn_final_a(emb_head_real) * self.bn_final_b(emb_rel_real) * self.bn_final_c(emb_tail_real)
        return s.sum(dim=1)


class KronE(BaseKGE):
    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KronE'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings
        self.entity_embedding_dim = args.entity_embedding_dim
        self.num_input_A = self.entity_embedding_dim // 2
        self.num_output_A = self.entity_embedding_dim // 4

        self.num_input_B = self.entity_embedding_dim // 2

        self.rel_embedding_dim = args.rel_embedding_dim
        self.rel_embedding_dim_mid = self.rel_embedding_dim // 2
        self.emb_ent_real = nn.Embedding(args.num_entities, self.entity_embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.rel_embedding_dim)

        # a by b kron c by d = ac by bd
        self.A = torch.rand(self.num_input_A, self.num_output_A, requires_grad=True)
        self.B = torch.rand(self.num_input_B, self.num_input_A, requires_grad=True)

        self.bn_weight_matrix = torch.nn.BatchNorm1d(
            (self.entity_embedding_dim // 4) * (self.entity_embedding_dim // 2))
        self.bn_hidden_features = torch.nn.BatchNorm1d(self.entity_embedding_dim)

        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        self.normalizer = torch.nn.BatchNorm1d  # or nn.LayerNorm
        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = self.normalizer(self.entity_embedding_dim)
        self.bn_rel_real = self.normalizer(self.rel_embedding_dim)
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

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
        # (1) Get head entity embeddings and decompress it by taking the half
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        emb_head_real = self.batch_kronecker_product(emb_head_real[:, :self.num_input_A].unsqueeze(1),
                                                     emb_head_real[:, self.num_input_A:].unsqueeze(1)).flatten(
            1)
        # (2) Get relation embeddings and decompress it by taking the half
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        emb_rel_real = self.batch_kronecker_product(emb_rel_real[:, :self.rel_embedding_dim_mid].unsqueeze(1),
                                                    emb_rel_real[:, self.rel_embedding_dim_mid:].unsqueeze(1)).flatten(
            1)
        # (3) Hadamard Product of (1) and (2)
        features = emb_head_real * emb_rel_real
        # (4) Obtain parameter matrix via kron to perform Linear transformation
        W = self.bn_weight_matrix(torch.kron(self.A, self.B))
        features = self.bn_hidden_features(features @ W)
        return torch.mm(features, self.emb_ent_real.weight.transpose(1, 0))


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
