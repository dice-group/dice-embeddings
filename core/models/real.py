from .base_model import *


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

        self.hidden_dropout = torch.nn.Dropout(args.hidden_dropout_rate)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward_k_vs_all(self, e1_idx: torch.Tensor, rel_idx: torch.Tensor):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        return torch.mm(self.hidden_dropout(emb_head_real * emb_rel_real), self.emb_ent_real.weight.transpose(1, 0))

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
        return (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)


class KronE(BaseKGE):
    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KronE'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings
        self.entity_embedding_dim = args.entity_embedding_dim
        self.rel_embedding_dim = args.rel_embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, self.entity_embedding_dim)  # real
        self.emb_rel_real = nn.Embedding(args.num_relations, self.rel_embedding_dim)  # real
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        self.normalizer = torch.nn.BatchNorm1d  # or nn.LayerNorm
        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = self.normalizer(self.entity_embedding_dim)
        self.bn_rel_real = self.normalizer(args.rel_embedding_dim)
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
