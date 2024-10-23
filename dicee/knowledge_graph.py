from typing import List
from .read_preprocess_save_load_kg import ReadFromDisk, PreprocessKG, LoadSaveToDisk
import sys
import pandas as pd
import polars as pl
import numpy as np
from .read_preprocess_save_load_kg.util import load_numpy_ndarray
class KG:
    """ Knowledge Graph """

    def __init__(self, dataset_dir: str = None,
                 byte_pair_encoding: bool = False,
                 padding: bool = False,
                 add_noise_rate: float = None,
                 sparql_endpoint: str = None,
                 path_single_kg: str = None,
                 path_for_deserialization: str = None,
                 add_reciprocal: bool = None, eval_model: str = None,
                 read_only_few: int = None, sample_triples_ratio: float = None,
                 path_for_serialization: str = None,
                 entity_to_idx=None, relation_to_idx=None, backend=None, training_technique: str = None, separator:str=None):
        """
        :param dataset_dir: A path of a folder containing train.txt, valid.txt, test.text
        :param byte_pair_encoding: Apply Byte pair encoding.
        :param padding: Add empty string into byte-pair encoded subword units representing triples
        :param add_noise_rate: Noisy triples added into the training adataset by x % of its size.
        :param sparql_endpoint: An endpoint of a triple store
        :param path_single_kg: The path of a single file containing the input knowledge graph
        :param path_for_deserialization: A path of a folder containing previously parsed data
        :param num_core: Number of subprocesses used for data loading
        :param add_reciprocal: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied.
        If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        :param training_technique
        """
        self.dataset_dir = dataset_dir
        self.sparql_endpoint = sparql_endpoint
        self.path_single_kg = path_single_kg

        self.byte_pair_encoding = byte_pair_encoding
        self.ordered_shaped_bpe_tokens = None
        self.add_noise_rate = add_noise_rate
        self.num_entities = None
        self.num_relations = None
        self.path_for_deserialization = path_for_deserialization
        self.add_reciprocal = add_reciprocal
        self.eval_model = eval_model

        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.path_for_serialization = path_for_serialization
        # dicts of str to int
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.backend = 'pandas' if backend is None else backend
        self.training_technique = training_technique
        self.raw_train_set, self.raw_valid_set, self.raw_test_set = None, None, None
        self.train_set, self.valid_set, self.test_set = None, None, None
        self.idx_entity_to_bpe_shaped = dict()

        # WIP:
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.num_tokens = self.enc.n_vocab  # ~ 50
        self.num_bpe_entities = None
        self.padding = padding
        # TODO: Find a unique token later
        self.dummy_id = self.enc.encode(" ")[0]
        self.max_length_subword_tokens = None
        self.train_set_target = None
        self.target_dim = None
        self.train_target_indices = None
        self.ordered_bpe_entities = None
        self.separator=separator

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
            assert isinstance(self.entity_to_idx, dict) or isinstance(self.entity_to_idx,
                                                                      pl.DataFrame), f"entity_to_idx must be a dict or a polars DataFrame: {type(self.entity_to_idx)}"

            if isinstance(self.entity_to_idx, dict):
                self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
                self.idx_to_relations = {v: k for k, v in self.relation_to_idx.items()}
            else:
                print(f"No inverse mapping created as self.entity_to_idx is not a type of dictionary but {type(self.entity_to_idx)}\n"
                      f"Backend might be selected as polars")

    def describe(self) -> None:
        self.description_of_input = f'\n------------------- Description of Dataset {self.dataset_dir if isinstance(self.dataset_dir, str) else self.sparql_endpoint if isinstance(self.sparql_endpoint, str) else self.path_single_kg} -------------------'
        if self.byte_pair_encoding:
            self.description_of_input += f'\nNumber of tokens:{self.num_tokens}' \
                                         f'\nNumber of max sequence of sub-words: {self.max_length_subword_tokens}' \
                                         f'\nNumber of triples on train set:' \
                                         f'{len(self.train_set)}' \
                                         f'\nNumber of triples on valid set:' \
                                         f'{len(self.valid_set) if self.valid_set is not None else 0}' \
                                         f'\nNumber of triples on test set:' \
                                         f'{len(self.test_set) if self.test_set is not None else 0}\n'
        else:
            self.description_of_input += f'\nNumber of entities:{self.num_entities}' \
                                         f'\nNumber of relations:{self.num_relations}' \
                                         f'\nNumber of triples on train set:' \
                                         f'{len(self.train_set)}' \
                                         f'\nNumber of triples on valid set:' \
                                         f'{len(self.valid_set) if self.valid_set is not None else 0}' \
                                         f'\nNumber of triples on test set:' \
                                         f'{len(self.test_set) if self.test_set is not None else 0}\n'
            self.description_of_input += f"Entity Index:{sys.getsizeof(self.entity_to_idx) / 1_000_000_000:.5f} in GB\n"
            self.description_of_input += f"Relation Index:{sys.getsizeof(self.relation_to_idx) / 1_000_000_000:.5f} in GB\n"

    @property
    def entities_str(self) -> List:
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List:
        return list(self.relation_to_idx.keys())

    def exists(self,h:str,r:str,t:str):
        # Row to check for existence
        row_to_check = {'subject': self.entity_to_idx[h], 'relation': self.relation_to_idx[r], 'object': self.entity_to_idx[t]}
        # Check if the row exists
        return ((self.raw_train_set == pd.Series(row_to_check)).all(axis=1)).any()

    def __iter__(self):
        for h, r, t in self.raw_train_set.to_numpy().tolist():
            yield self.idx_to_entity[h], self.idx_to_relations[r], self.idx_to_entity[t]
    def __len__(self):
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
