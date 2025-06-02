import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from dicee.models.transformers import LayerNorm


class GatedLinearUnit(nn.Module):
    """
    Applies a gated linear unit (GLU) operation:
    Splits the input in half along the last dimension,
    applies a sigmoid gate to one half and multiplies it with the other.
    """

    def __init__(self, input_dim, gated_residual=True):
        super().__init__()
        self.proj = nn.Linear(input_dim * 2, input_dim * 2)
        self.gate_residual = gated_residual

    def forward(self, x1, x2):
        if self.gate_residual:
            x_proj = self.proj(torch.cat((x1, x2), dim=1))
            value, gate = x_proj.chunk(2, dim=-1)
            # Split into two parts
            return value * torch.sigmoid(gate)  # Apply gating
        else:
            return x1 + x2


class LiteralEmbeddings(nn.Module):
    """
    A model for learning and predicting numerical literals using pre-trained KGE.

    Attributes:
        num_of_data_properties (int): Number of data properties (attributes).
        embedding_dims (int): Dimension of the embeddings.
        entity_embeddings (torch.tensor): Pre-trained entity embeddings.
        dropout (float): Dropout rate for regularization.
        gate_residual (bool): Whether to use gated residual connections.
        freeze_entity_embeddings (bool): Whether to freeze the entity embeddings during training.
    """

    def __init__(
        self,
        num_of_data_properties: int,
        embedding_dims: int,
        entity_embeddings: torch.tensor,
        dropout: float = 0.3,
        gate_residual=True,
        freeze_entity_embeddings=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dims
        self.num_of_data_properties = num_of_data_properties
        self.hidden_dim = embedding_dims * 2  # Combined entity + attribute embeddings

        # Use pre-trained entity embeddings
        self.entity_embeddings = nn.Embedding.from_pretrained(
            entity_embeddings.weight, freeze=freeze_entity_embeddings
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
        self.residual = GatedLinearUnit(self.hidden_dim, gated_residual=gate_residual)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, entity_idx, attr_idx):
        """
        Args:
            entity_idx (Tensor): Entity indices (batch).
            attr_idx (Tensor): Attribute (Data property)  indices (batch).

        Returns:
            Tensor: scalar predictions.
        """
        # embeddings lookup
        e_emb = self.entity_embeddings(entity_idx)  # [batch, emb_dim]
        a_emb = self.data_property_embeddings(attr_idx)  # [batch, emb_dim]

        # Concatenate entity and property embeddings
        tuple_emb = torch.cat((e_emb, a_emb), dim=1)  # [batch, 2 * emb_dim]

        # MLP with dropout and ReLU
        z = self.dropout(
            F.relu(self.layer_norm(self.fc(tuple_emb)))
        )  # [batch, 2 * emb_dim]

        # Gated residual connection: combine original (tuple) + transformed embeddings
        residual = self.residual(z, tuple_emb)  # [batch, 2 * emb_dim]

        # Output scalar prediction and flatten to 1D
        out = self.fc_out(residual).flatten()  # [batch]
        return out


class LiteralDataset(Dataset):
    """Dataset for loading and processing literal data for training Literal Embedding model.
    This dataset handles the loading, normalization, and preparation of triples
    for training a literal embedding model.

    Extends torch.utils.data.Dataset for supporting PyTorch dataloaders.

    Attributes:
        train_file_path (str): Path to the training data file.
        normalization (str): Type of normalization to apply ('z-norm', 'min-max', or None).
        normalization_params (dict): Parameters used for normalization.
        sampling_ratio (float): Fraction of the training set to use for ablations.
        entity_to_idx (dict): Mapping of entities to their indices.
        num_entities (int): Total number of entities.
        data_property_to_idx (dict): Mapping of data properties to their indices.
        num_data_properties (int): Total number of data properties.
    """

    def __init__(
        self,
        file_path: str,
        ent_idx: dict = None,
        normalization_type: str = "z-norm",
        sampling_ratio: float = None,
    ):
        self.train_file_path = file_path
        self.normalization_type = normalization_type
        self.normalization_params = {}
        self.sampling_ratio = sampling_ratio
        self.entity_to_idx = ent_idx
        self.num_entities = len(self.entity_to_idx)

        if self.entity_to_idx is None:
            raise ValueError(
                "entity_to_idx must be provided to initialize LiteralDataset."
            )

        self._load_data()

    def _load_data(self):
        train_df = self.load_and_validate_literal_data(
            self.train_file_path,
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        self.data_property_to_idx = {
            rel: idx for idx, rel in enumerate(train_df["attribute"].unique())
        }
        self.num_data_properties = len(self.data_property_to_idx)
        if self.sampling_ratio is not None:
            # reduce the train set for ablations using sampling ratio
            # keeps the sampling_ratio * 100 % of full training set in the train_df
            if 0 < self.sampling_ratio <= 1:
                train_df = (
                    train_df.groupby("attribute", group_keys=False)
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

        train_df["head_idx"] = train_df["head"].map(self.entity_to_idx)
        train_df["attr_idx"] = train_df["attribute"].map(self.data_property_to_idx)
        train_df = self._apply_normalization(train_df)

        self.triples = torch.tensor(
            train_df[["head_idx", "attr_idx"]].values, dtype=torch.long
        )
        self.values = torch.tensor(train_df["value"].values, dtype=torch.float32)
        self.values_norm = torch.tensor(
            train_df["value_norm"].values, dtype=torch.float32
        )

    def _apply_normalization(self, df):
        """Applies normalization to the tail values based on the specified type."""
        if self.normalization_type == "z-norm":
            stats = df.groupby("attribute")["value"].agg(["mean", "std"])
            self.normalization_params = stats.to_dict(orient="index")
            df["value_norm"] = df.groupby("attribute")["value"].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            self.normalization_params["type"] = "z-norm"

        elif self.normalization_type == "min-max":
            stats = df.groupby("attribute")["value"].agg(["min", "max"])
            self.normalization_params = stats.to_dict(orient="index")
            df["value_norm"] = df.groupby("attribute")["value"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            self.normalization_params["type"] = "min-max"

        else:
            print(" No normalization applied.")
            df["value_norm"] = df["value"]
            self.normalization_params = None
            self.normalization_params["type"] = None

        return df

    def __getitem__(self, index):
        return self.triples[index], self.values_norm[index]

    def __len__(self):
        return len(self.triples)

    @staticmethod
    def load_and_validate_literal_data(file_path: str = None) -> pd.DataFrame:
        """Loads and validates the literal data file.
        Args:
            file_path (str): Path to the literal data file.
        Returns:
            pd.DataFrame: DataFrame containing the loaded and validated data.
        """

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        # Try loading the file with either tab or comma separator
        last_exception = None
        df = None
        for sep in ["\t", ","]:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None, index_col=False)
                # Success—break out of the loop
                break
            except Exception as e:
                last_exception = e

        # After loop, check if df was successfully loaded
        if df is None or df.empty:
            raise ValueError(
                f"Could not read file '{file_path}' with tab or comma separator. Last error: {last_exception}"
            )

        assert (
            df.shape[1] == 3
        ), "Data file must contain exactly 3 columns: head, attribute, and value."
        # Name the columns
        df.columns = ["head", "attribute", "value"]

        # Validate column types
        if not pd.api.types.is_string_dtype(df["head"]):
            raise TypeError("Column 'head' must be of string type.")
        if not pd.api.types.is_string_dtype(df["attribute"]):
            raise TypeError("Column 'attribute' must be of string type.")
        if not pd.api.types.is_numeric_dtype(df["value"]):
            raise TypeError("Column 'value' must be numeric.")

        return df

    @staticmethod
    def denormalize(preds_norm, attributes, normalization_params) -> np.ndarray:
        """Denormalizes the predictions based on the normalization type.

        Args:
        preds_norm (np.ndarray): Normalized predictions to be denormalized.
        attributes (list): List of attributes corresponding to the predictions.
        normalization_params (dict): Dictionary containing normalization parameters for each attribute.

        Returns:
            np.ndarray: Denormalized predictions.

        """
        if normalization_params["type"] == "z-norm":
            # Extract means and stds only if z-norm is used
            means = np.array([normalization_params[i]["mean"] for i in attributes])
            stds = np.array([normalization_params[i]["std"] for i in attributes])
            return preds_norm * stds + means

        elif normalization_params["type"] == "min-max":
            # Extract mins and maxs only if min-max is used
            mins = np.array([normalization_params[i]["min"] for i in attributes])
            maxs = np.array([normalization_params[i]["max"] for i in attributes])
            return preds_norm * (maxs - mins) + mins

        elif normalization_params["type"] is None:
            return preds_norm

        else:
            raise ValueError(
                "Unsupported normalization type. Use 'z-norm', 'min-max', or None."
            )
