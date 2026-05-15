"""PFN convenience package.

This package groups the GraphPFN components while keeping backwards
compatibility with the existing top-level scripts.
"""

from pfn_dataset import PFNDataset, RandomSupportPrior, RichSubgraphPrior, build_dataset
from pfn_evaluate import evaluate_bce
from pfn_inference import evaluate, infer, score_triple
from pfn_model import TriplePFN
from pfn_train import train

__all__ = [
    "PFNDataset",
    "RandomSupportPrior",
    "RichSubgraphPrior",
    "TriplePFN",
    "build_dataset",
    "evaluate",
    "evaluate_bce",
    "infer",
    "score_triple",
    "train",
]
