"""Literal embedding dataset.

Provides ``LiteralDataset`` for training models on numeric literal triples
(entity, data-property, value).
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LiteralDataset(Dataset):
    """Dataset for loading and processing literal data for Literal Embedding models.

    Handles loading, normalization, and preparation of ``(entity, attribute,
    value)`` triples.  Supports z-score and min-max normalization as well as
    optional sub-sampling for ablation studies.

    Parameters
    ----------
    file_path : str
        Path to the training data file (CSV / TSV / RDF).
    ent_idx : dict
        Entity-name → index mapping.
    normalization_type : str, optional
        ``'z-norm'``, ``'min-max'``, or ``None`` (default ``'z-norm'``).
    sampling_ratio : float or None, optional
        Fraction of the training set to keep (default ``None`` = use all).
    loader_backend : str, optional
        ``'pandas'`` or ``'rdflib'`` (default ``'pandas'``).
    """

    def __init__(
        self,
        file_path: str,
        ent_idx: dict = None,
        normalization_type: str = "z-norm",
        sampling_ratio: float = None,
        loader_backend: str = "pandas",
    ):
        self.train_file_path = file_path
        self.loader_backend = loader_backend
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
        """Load, filter, index, and normalize the literal data."""
        train_df = self.load_and_validate_literal_data(
            self.train_file_path, loader_backend=self.loader_backend
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        assert (
            not train_df.empty
        ), "Filtered train_df is empty — no entities match entity_to_idx."

        self.data_property_to_idx = {
            rel: idx
            for idx, rel in enumerate(sorted(train_df["attribute"].unique()))
        }
        self.num_data_properties = len(self.data_property_to_idx)

        if self.sampling_ratio is not None:
            if 0 < self.sampling_ratio <= 1:
                train_df = (
                    train_df.groupby("attribute", group_keys=False)
                    .apply(
                        lambda x: x.sample(
                            frac=self.sampling_ratio, random_state=42
                        )
                    )
                    .reset_index(drop=True)
                )
                print(
                    f"Training Literal Embedding model with "
                    f"{self.sampling_ratio * 100:.1f}% of the train set."
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

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _apply_normalization(self, df):
        """Apply the configured normalization to the ``value`` column."""
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
            if self.normalization_type is None:
                self.normalization_params = {}
                self.normalization_params["type"] = None

        return df

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        return self.triples[index], self.values_norm[index]

    def __len__(self):
        return len(self.triples)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_and_validate_literal_data(
        file_path: str = None, loader_backend: str = "pandas"
    ) -> pd.DataFrame:
        """Load and validate a literal data file.

        Parameters
        ----------
        file_path : str
            Path to the data file.
        loader_backend : str
            ``'pandas'`` or ``'rdflib'``.

        Returns
        -------
        pandas.DataFrame
            Three-column DataFrame with columns ``head``, ``attribute``,
            ``value``.
        """
        try:
            import rdflib
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "rdflib is required for loading RDF files. "
                "Please install it via 'pip install rdflib'."
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        df = None
        if loader_backend == "rdflib":
            try:
                g = rdflib.Graph().parse(file_path)
            except Exception as e:
                raise ValueError(f"Failed to parse RDF file: {e}")

            triples = []
            for s, p, o in g:
                if isinstance(o, rdflib.Literal):
                    value = o.toPython()
                    if isinstance(value, (int, float)):
                        triples.append((s.n3()[1:-1], p.n3()[1:-1], float(value)))

            df = pd.DataFrame(triples, columns=None)

        elif loader_backend == "pandas":
            last_exception = None
            for sep in ["\t", ","]:
                try:
                    df = pd.read_csv(
                        file_path, sep=sep, header=None, index_col=False
                    )
                    break
                except Exception as e:
                    last_exception = e
        else:
            raise ValueError(
                f"Unsupported loader backend: {loader_backend}. "
                "Use 'rdflib' or 'pandas'."
            )

        if df is None or df.empty:
            raise ValueError(
                f"Could not read file '{file_path}' with tab or comma "
                f"separator. Last error: {last_exception}"
            )

        assert (
            df.shape[1] == 3
        ), "Data file must contain exactly 3 columns: head, attribute, and value."
        df.columns = ["head", "attribute", "value"]

        if not pd.api.types.is_string_dtype(df["head"]):
            raise TypeError("Column 'head' must be of string type.")
        if not pd.api.types.is_string_dtype(df["attribute"]):
            raise TypeError("Column 'attribute' must be of string type.")
        if not pd.api.types.is_numeric_dtype(df["value"]):
            raise TypeError("Column 'value' must be numeric.")

        return df

    @staticmethod
    def denormalize(preds_norm, attributes, normalization_params) -> np.ndarray:
        """Reverse the normalization applied during training.

        Parameters
        ----------
        preds_norm : numpy.ndarray
            Normalized predictions.
        attributes : list
            Attribute names corresponding to each prediction.
        normalization_params : dict
            Parameters stored during training.

        Returns
        -------
        numpy.ndarray
            Denormalized predictions.
        """
        norm_type = normalization_params["type"]

        if norm_type == "z-norm":
            means = np.array(
                [normalization_params[i]["mean"] for i in attributes]
            )
            stds = np.array(
                [normalization_params[i]["std"] for i in attributes]
            )
            return preds_norm * stds + means

        elif norm_type == "min-max":
            mins = np.array(
                [normalization_params[i]["min"] for i in attributes]
            )
            maxs = np.array(
                [normalization_params[i]["max"] for i in attributes]
            )
            return preds_norm * (maxs - mins) + mins

        elif norm_type is None:
            return preds_norm

        else:
            raise ValueError(
                "Unsupported normalization type. Use 'z-norm', 'min-max', or None."
            )
