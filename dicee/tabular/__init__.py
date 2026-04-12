"""Tabular training utilities for knowledge-graph data."""

from .static_funcs import (
    convert_only,
    demo_entity_centric,
    load_kg_for_tabpfn,
    prepare_tabpfn_splits,
    summarize_converted_data,
    summarize_training_inputs,
)
from .tabular_dataset import EntityCentricConverter, KGToTabularConverter
from .trainer import TabularTrainer

__all__ = [
    "EntityCentricConverter",
    "KGToTabularConverter",
    "TabularTrainer",
    "prepare_tabpfn_splits",
    "convert_only",
    "demo_entity_centric",
    "load_kg_for_tabpfn",
    "summarize_converted_data",
    "summarize_training_inputs",
]
