"""Dataset classes for knowledge graph embedding training.

This package groups the various ``torch.utils.data.Dataset`` implementations
used throughout DICE Embeddings into thematic sub-modules:

* :mod:`._bpe` – Byte-pair-encoding related datasets.
* :mod:`._negative_sampling` – Negative sampling based datasets.
* :mod:`._label_based` – Multi-label / multi-class scoring datasets.
* :mod:`._literal` – Literal (numeric) embedding dataset.
* :mod:`._factory` – ``construct_dataset`` / ``reload_dataset`` helpers.

All public names are re-exported here so that existing ``from
dicee.dataset_classes import …`` statements continue to work unchanged.
"""

# BPE datasets
from ._bpe import (  # noqa: F401
    BPE_NegativeSamplingDataset,
    MultiClassClassificationDataset,
    MultiLabelDataset,
)

# Label-based scoring datasets
from ._label_based import (  # noqa: F401
    AllvsAll,
    KvsAll,
    KvsSampleDataset,
    OnevsAllDataset,
)

# Negative-sampling datasets
from ._negative_sampling import (  # noqa: F401
    NegSampleDataset,
    OnevsSample,
    TriplePredictionDataset,
)

# Literal dataset
from ._literal import LiteralDataset  # noqa: F401

# Factory functions
from ._factory import construct_dataset, reload_dataset  # noqa: F401

__all__ = [
    # BPE
    "BPE_NegativeSamplingDataset",
    "MultiClassClassificationDataset",
    "MultiLabelDataset",
    # Label-based
    "AllvsAll",
    "KvsAll",
    "KvsSampleDataset",
    "OnevsAllDataset",
    # Negative-sampling
    "NegSampleDataset",
    "OnevsSample",
    "TriplePredictionDataset",
    # Literal
    "LiteralDataset",
    # Factory
    "construct_dataset",
    "reload_dataset",
]
