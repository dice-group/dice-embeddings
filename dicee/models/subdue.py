from .clifford import DeCaL
import torch
import torch.nn as nn

class SubdueWithDeCal(DeCaL):
    def __init__(self, args):
        super().__init__(args)

        self.head = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.name = "SubdueWithDeCal"

    def forward(self, x, labels=None):
        h = self.entity_embeddings(x[:, 0])
        r = self.relation_embeddings(x[:, 1])
        t = self.entity_embeddings(x[:, 2])
        input_vec = torch.cat([h, r, t], dim=1)
        logits = self.head(input_vec).squeeze(-1)
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            return logits, loss
        return logits
