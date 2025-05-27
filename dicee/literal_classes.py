import os

import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from dicee.models.transformers import LayerNorm

class GatedLinearUnit(nn.Module):
    """
    Applies a gated linear unit (GLU) operation:
    Splits the input in half along the last dimension,
    applies a sigmoid gate to one half and multiplies it with the other.
    """

    def __init__(self, input_dim, gated_residual = True):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, input_dim)
        self.proj2 = nn.Linear(input_dim, input_dim)
        self.gate_residual =  gated_residual 

    def forward(self, x1, x2):
        if self.gate_residual:
            x1_proj = self.proj1(x1)
            x2_proj = self.proj2(x2)
            # Split into two parts
            return x1_proj * torch.sigmoid(x2_proj)  # Apply gating
        else:
            return x1 + x2


class LiteralEmbeddings(nn.Module):
    """
    A model for learning and predicting numerical literals using pre-trained KGE.
    """

    def __init__(
        self,
        num_of_data_properties: int,
        embedding_dims: int,
        entity_embeddings: torch.tensor,
        dropout: float = 0.3,
        gate_residual = True

    ):
        super().__init__()
        self.embedding_dim = embedding_dims
        self.num_of_data_properties = num_of_data_properties
        self.hidden_dim = embedding_dims * 2  # Combined entity + attribute embeddings

        # Use pre-trained entity embeddings
        self.entity_embeddings = nn.Embedding.from_pretrained(
            entity_embeddings.weight, freeze=True
        )

        #  data property (literal) embeddings
        self.data_property_embeddings = nn.Embedding(
            num_embeddings=num_of_data_properties,
            embedding_dim=self.embedding_dim,
        )

        # MLP components
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

        # Gated residual layer with layer norm
        self.residual = GatedLinearUnit(self.hidden_dim, gate_residual)
        self.layer_norm = LayerNorm(self.hidden_dim, bias=True)

    def forward(self, e_idx, relation_idx):
        """
        Args:
            e_idx (Tensor): Entity indices (batch).
            relation_idx (Tensor): Data property indices (batch).

        Returns:
            Tensor: scalar predictions.
        """
        # embeddings lookup
        e_emb = self.entity_embeddings(e_idx)  # [batch, emb_dim]
        a_emb = self.data_property_embeddings(relation_idx)  # [batch, emb_dim]

        # Concatenate entity and property embeddings
        tuple_emb = torch.cat((e_emb, a_emb), dim=1)  # [batch, 2 * emb_dim]

        # MLP with dropout and ReLU
        z = self.dropout(F.relu(self.fc(tuple_emb)))  # [batch, 2 * emb_dim]

        # Gated residual connection: combine original (tuple) + transformed embeddings
        residual = self.residual(z, tuple_emb)  # [batch, 2 * hidden]

        # Output scalar prediction and flatten to 1D
        out = self.fc_out(residual).flatten()  # [batch]
        return out


class LiteralDataset(Dataset):
    def __init__(
        self, file_path: str, ent_idx, normalization="z-norm", sampling_ratio=None
    ):
        self.train_file_path = file_path
        self.normalization = normalization
        self.normalization_params = {}
        self.sampling_ratio = sampling_ratio

        self.entity_to_idx = ent_idx
        self.num_entities = len(self.entity_to_idx)

        self._load_data()

    def _load_data(self):
        # Load mapping from train
        if not os.path.exists(self.train_file_path):
            raise FileNotFoundError(f"Data file not found at {self.train_file_path}")
        train_df = pd.read_csv(
            self.train_file_path,
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        self.data_property_to_idx = {
            rel: idx for idx, rel in enumerate(train_df["relation"].unique())
        }

        if self.sampling_ratio is not None:
            # reduce the train set for ablations using sampling ratio
            # keeps the sampling_ratio * 100 % of full training set in the train_df
            if 0 < self.sampling_ratio <= 1:

                train_df = (
                    train_df.groupby("relation", group_keys=False)
                    .apply(
                        lambda x: x.sample(frac=self.sampling_ratio, random_state=42)
                    )
                    .reset_index(drop=True)
                )
                print(
                    f"Training Literal Embedding model with {self.sampling_ratio*100:.1f}% of the train set."
                )
            else:
                raise ValueError("Split Fraction must be between 0 and 1.")

        self.num_data_properties = len(self.data_property_to_idx)

        train_df["head_idx"] = train_df["head"].map(self.entity_to_idx)
        train_df["rel_idx"] = train_df["relation"].map(self.data_property_to_idx)
        train_df = self._apply_normalization(train_df)

        self.triples = torch.tensor(
            train_df[["head_idx", "rel_idx"]].values, dtype=torch.long
        )
        self.tails = torch.tensor(train_df["tail"].values, dtype=torch.float32)
        self.tails_norm = torch.tensor(
            train_df["tail_norm"].values, dtype=torch.float32
        )

    def _apply_normalization(self, df):
        if self.normalization == "z-norm":
            stats = df.groupby("rel_idx")["tail"].agg(["mean", "std"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("rel_idx")["tail"].transform(
                lambda x: (x - x.mean()) / x.std()
            )

        elif self.normalization == "min-max":
            stats = df.groupby("rel_idx")["tail"].agg(["min", "max"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("rel_idx")["tail"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

        else:
            print(" No normalization applied.")
            df["tail_norm"] = df["tail"]
            self.normalization_params = None

        return df

    def __getitem__(self, index):
        return self.triples[index], self.tails_norm[index]

    def __len__(self):
        return len(self.triples)
