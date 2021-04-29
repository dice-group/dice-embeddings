from .base_model import *

class Shallom(BaseKGE):
    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'Shallom'
        shallom_width = int(args.shallom_width_ratio_of_emb * args.embedding_dim)
        self.loss = torch.nn.BCELoss()
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
        return torch.sigmoid(self.shallom(torch.cat((emb_s, emb_o), 1)))
class DistMult(BaseKGE):
    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'DistMult'
        self.loss = torch.nn.BCELoss()
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

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward_k_vs_all(self, e1_idx, rel_idx):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        real_score = torch.mm(emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        score = real_score
        return torch.sigmoid(score)
