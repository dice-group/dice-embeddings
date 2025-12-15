"""Knowledge Graph module for data loading and preprocessing.

Provides the KG class for handling knowledge graph data including
loading, preprocessing, and indexing operations.
"""
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import polars as pl
import tiktoken

from .read_preprocess_save_load_kg import LoadSaveToDisk, PreprocessKG, ReadFromDisk
class KG:
    """Knowledge Graph container and processor.

    Handles loading, preprocessing, and indexing of knowledge graph data
    from various sources including files, SPARQL endpoints, and serialized formats.

    Attributes:
        dataset_dir: Path to directory containing train/valid/test files.
        num_entities: Total number of unique entities.
        num_relations: Total number of unique relations.
        train_set: Indexed training triples as numpy array.
        valid_set: Indexed validation triples (optional).
        test_set: Indexed test triples (optional).
        entity_to_idx: Mapping from entity strings to indices.
        relation_to_idx: Mapping from relation strings to indices.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        byte_pair_encoding: bool = False,
        padding: bool = False,
        add_noise_rate: Optional[float] = None,
        sparql_endpoint: Optional[str] = None,
        path_single_kg: Optional[str] = None,
        path_for_deserialization: Optional[str] = None,
        add_reciprocal: Optional[bool] = None,
        eval_model: Optional[str] = None,
        read_only_few: Optional[int] = None,
        sample_triples_ratio: Optional[float] = None,
        path_for_serialization: Optional[str] = None,
        entity_to_idx: Optional[Dict] = None,
        relation_to_idx: Optional[Dict] = None,
        backend: Optional[str] = None,
        training_technique: Optional[str] = None,
        separator: Optional[str] = None
    ):
        """Initialize the Knowledge Graph.

        Args:
            dataset_dir: Path to folder with train.txt, valid.txt, test.txt.
            byte_pair_encoding: Whether to apply byte pair encoding.
            padding: Add padding to BPE encoded subword units.
            add_noise_rate: Ratio of noisy triples to add (e.g., 0.1 for 10%).
            sparql_endpoint: SPARQL endpoint URL for querying.
            path_single_kg: Path to a single file containing the KG.
            path_for_deserialization: Path to load pre-processed data.
            add_reciprocal: Whether to add reciprocal triples.
            eval_model: Evaluation mode ('train', 'val', 'test', or combinations).
            read_only_few: Limit number of triples to read.
            sample_triples_ratio: Ratio of triples to sample (0-1).
            path_for_serialization: Path to save processed data.
            entity_to_idx: Pre-existing entity to index mapping.
            relation_to_idx: Pre-existing relation to index mapping.
            backend: Data processing backend ('pandas', 'polars', 'rdflib').
            training_technique: Scoring technique for training.
            separator: Separator for parsing triple files.
        """
        # Store configuration
        self.dataset_dir = dataset_dir
        self.sparql_endpoint = sparql_endpoint
        self.path_single_kg = path_single_kg
        self.byte_pair_encoding = byte_pair_encoding
        self.ordered_shaped_bpe_tokens = None
        self.add_noise_rate = add_noise_rate
        self.num_entities: Optional[int] = None
        self.num_relations: Optional[int] = None
        self.path_for_deserialization = path_for_deserialization
        self.add_reciprocal = add_reciprocal
        self.eval_model = eval_model
        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.path_for_serialization = path_for_serialization
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.backend = backend or 'pandas'
        self.training_technique = training_technique
        self.separator = separator

        # Initialize dataset placeholders
        self.raw_train_set = None
        self.raw_valid_set = None
        self.raw_test_set = None
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.idx_entity_to_bpe_shaped: Dict = {}

        # Initialize BPE components
        self.enc = tiktoken.get_encoding("gpt2")
        self.num_tokens = self.enc.n_vocab
        self.num_bpe_entities: Optional[int] = None
        self.padding = padding
        self.dummy_id = self.enc.encode(" ")[0]
        self.max_length_subword_tokens: Optional[int] = None
        self.train_set_target = None
        self.target_dim: Optional[int] = None
        self.train_target_indices = None
        self.ordered_bpe_entities = None

        if self.path_for_deserialization is None:
            # Read a knowledge graph into memory
            ReadFromDisk(kg=self).start()
            # Map a knowledge graph into integer indexed.
            PreprocessKG(kg=self).start()
            # Saving.
            LoadSaveToDisk(kg=self).save()
        else:
            LoadSaveToDisk(kg=self).load()
        assert len(self.train_set) > 0, "Training set is empty"
        self.description_of_input=None
        self.describe()

        if self.entity_to_idx is not None:
            assert isinstance(self.entity_to_idx, dict) or isinstance(self.entity_to_idx, pd.DataFrame) or isinstance(self.entity_to_idx,
                                                                      pl.DataFrame), f"entity_to_idx must be a dict or a pandas/polars DataFrame: {type(self.entity_to_idx)}"
            # TODO:CD: Why do we need to create this inverse mapping at this point?
            if isinstance(self.entity_to_idx, dict):
                self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
                self.idx_to_relations = {v: k for k, v in self.relation_to_idx.items()}
            else:
                pass

    def describe(self) -> None:
        """Generate a description string of the dataset statistics."""
        source = (
            self.dataset_dir if isinstance(self.dataset_dir, str)
            else self.sparql_endpoint if isinstance(self.sparql_endpoint, str)
            else self.path_single_kg
        )
        lines = [f'\n{"="*20} Description of Dataset {source} {"="*20}']

        if self.byte_pair_encoding:
            lines.extend([
                f'Number of tokens: {self.num_tokens}',
                f'Max sequence length of sub-words: {self.max_length_subword_tokens}',
                f'Number of triples on train set: {len(self.train_set)}',
                f'Number of triples on valid set: {len(self.valid_set) if self.valid_set is not None else 0}',
                f'Number of triples on test set: {len(self.test_set) if self.test_set is not None else 0}',
            ])
        else:
            lines.extend([
                f'Number of entities: {self.num_entities}',
                f'Number of relations: {self.num_relations}',
                f'Number of triples on train set: {len(self.train_set)}',
                f'Number of triples on valid set: {len(self.valid_set) if self.valid_set is not None else 0}',
                f'Number of triples on test set: {len(self.test_set) if self.test_set is not None else 0}',
                f'Entity Index: {sys.getsizeof(self.entity_to_idx) / 1_000_000_000:.5f} GB',
                f'Relation Index: {sys.getsizeof(self.relation_to_idx) / 1_000_000_000:.5f} GB',
            ])

        self.description_of_input = '\n'.join(lines)

    @property
    def entities_str(self) -> List[str]:
        """Get list of all entity strings."""
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List[str]:
        """Get list of all relation strings."""
        return list(self.relation_to_idx.keys())

    def exists(self, h: str, r: str, t: str) -> bool:
        """Check if a triple exists in the training set.

        Args:
            h: Head entity string.
            r: Relation string.
            t: Tail entity string.

        Returns:
            True if the triple exists, False otherwise.
        """
        row_to_check = {
            'subject': self.entity_to_idx[h],
            'relation': self.relation_to_idx[r],
            'object': self.entity_to_idx[t]
        }
        return ((self.raw_train_set == pd.Series(row_to_check)).all(axis=1)).any()

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        """Iterate over training triples as string tuples."""
        for h, r, t in self.raw_train_set.to_numpy().tolist():
            yield self.idx_to_entity[h], self.idx_to_relations[r], self.idx_to_entity[t]

    def __len__(self) -> int:
        """Return number of triples in the raw training set."""
        return len(self.raw_train_set)

    def func_triple_to_bpe_representation(self, triple: List[str]):
        result = []

        for x in triple:
            unshaped_bpe_repr = self.enc.encode(x)
            if len(unshaped_bpe_repr) < self.max_length_subword_tokens:
                unshaped_bpe_repr.extend([self.dummy_id for _ in
                                          range(self.max_length_subword_tokens - len(unshaped_bpe_repr))])
            else:
                pass
            result.append(unshaped_bpe_repr)
        return result
