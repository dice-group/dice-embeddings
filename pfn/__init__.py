"""PFN convenience package.

This package groups the GraphPFN components while keeping backwards
compatibility with the existing top-level scripts.
"""

from pfn.dataset import PFNDataset, RandomSupportPrior, RichSubgraphPrior, build_dataset
from pfn.evaluate import evaluate_bce
from pfn.inference import evaluate, infer, score_triple, visualize_triple_scoring
from pfn.model import TriplePFN
from pfn.train import train

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
    "visualize_triple_scoring",
]
