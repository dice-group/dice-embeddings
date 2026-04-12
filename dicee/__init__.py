"""DICE Embeddings - Knowledge Graph Embedding Library.

A library for training and using knowledge graph embedding models
with support for various scoring techniques and training strategies.

Submodules:
    evaluation: Model evaluation functions and Evaluator class
    models: KGE model implementations
    trainer: Training orchestration
    scripts: Utility scripts
"""
from .dataset_classes import *  # noqa
from .executer import Execute  # noqa
from .executer import TabularExecute  # noqa
from .knowledge_graph_embeddings import KGE  # noqa
from .query_generator import QueryGenerator  # noqa
from .static_funcs import *  # noqa
from .trainer import DICE_Trainer  # noqa
from .evaluation import Evaluator  # noqa

__version__ = '0.3.3'

__all__ = [
    'Execute',
    'TabularExecute',
    'KGE',
    'QueryGenerator',
    'DICE_Trainer',
    'Evaluator',
    '__version__',
]
