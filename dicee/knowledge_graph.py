from typing import List, Tuple
from .read_preprocess_save_load_kg import ReadFromDisk, PreprocessKG, LoadSaveToDisk
from .read_preprocess_save_load_kg.util import get_er_vocab, get_re_vocab, get_ee_vocab, create_constraints
import sys
import numpy as np
import concurrent
import tiktoken
import torch


class KG:
    """ Knowledge Graph """

    def __init__(self, dataset_dir: str = None,
                 byte_pair_encoding: bool = False,
                 add_noise_rate: float = None,
                 sparql_endpoint: str = None,
                 path_single_kg: str = None,
                 path_for_deserialization: str = None,
                 add_reciprical: bool = None, eval_model: str = None,
                 read_only_few: int = None, sample_triples_ratio: float = None,
                 path_for_serialization: str = None,
                 entity_to_idx=None, relation_to_idx=None, backend=None):
        """
        :param dataset_dir: A path of a folder containing train.txt, valid.txt, test.text
        :param byte_pair_encoding: Apply Byte pair encoding.
        :param add_noise_rate: Noisy triples added into the training adataset by x % of its size.
        : param sparql_endpoint: An endpoint of a triple store
        :param path_single_kg: The path of a single file containing the input knowledge graph
        :param path_for_deserialization: A path of a folder containing previously parsed data
        :param num_core: Number of subprocesses used for data loading
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied.
        If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        """
        self.dataset_dir = dataset_dir
        self.byte_pair_encoding = byte_pair_encoding
        if self.byte_pair_encoding:
            self.enc = tiktoken.get_encoding("gpt2")
            self.num_tokens = self.enc.n_vocab  # ~ 50
            self.dummy_id = self.enc.encode(" ")[0]
        else:
            self.enc = None
            self.num_tokens = None

        self.ordered_shaped_bpe_tokens = None
        self.sparql_endpoint = sparql_endpoint
        self.add_noise_rate = add_noise_rate
        self.num_entities = None
        self.num_relations = None
        self.path_single_kg = path_single_kg
        self.path_for_deserialization = path_for_deserialization
        self.add_reciprical = add_reciprical
        self.eval_model = eval_model

        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.path_for_serialization = path_for_serialization
        # dicts of str to int
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.backend = 'pandas' if backend is None else backend
        self.train_set, self.valid_set, self.test_set = None, None, None

        if self.path_for_deserialization is None:
            ReadFromDisk(kg=self).start()
            PreprocessKG(kg=self).start()
            # @TODO: move this into PreprocessKG
            if self.byte_pair_encoding:
                tokens = set()
                max_len = 0
                for i in self.train_set + self.valid_set + self.test_set:
                    max_token_length_per_triple = max(len(i[0]), len(i[1]), len(i[2]))
                    if max_token_length_per_triple > max_len:
                        max_len = max_token_length_per_triple

                for i in range(len(self.train_set)):
                    # Tuple of three tuples
                    s, p, o = self.train_set[i]
                    if len(s) < max_len:
                        s_encoded = s + tuple(self.dummy_id for _ in range(max_len - len(s)))
                    else:
                        s_encoded = s

                    if len(p) < max_len:
                        p_encoded = p + tuple(self.dummy_id for _ in range(max_len - len(p)))
                    else:
                        p_encoded = p

                    if len(o) < max_len:
                        o_encoded = o + tuple(self.dummy_id for _ in range(max_len - len(o)))
                    else:
                        o_encoded = o

                    tokens.add(s_encoded)
                    tokens.add(p_encoded)
                    tokens.add(o_encoded)
                    self.train_set[i] = (s_encoded, p_encoded, o_encoded)

                for i in range(len(self.valid_set)):
                    # Tuple of three tuples
                    s, p, o = self.valid_set[i]
                    if len(s) < max_len:
                        s_encoded = s + tuple(self.dummy_id for _ in range(max_len - len(s)))
                    else:
                        s_encoded = s
                    if len(p) < max_len:
                        p_encoded = p + tuple(self.dummy_id for _ in range(max_len - len(p)))
                    else:
                        p_encoded = p

                    if len(o) < max_len:
                        o_encoded = o + tuple(self.dummy_id for _ in range(max_len - len(o)))
                    else:
                        o_encoded = o
                    tokens.add(s_encoded)
                    tokens.add(p_encoded)
                    tokens.add(o_encoded)
                    self.valid_set[i] = (s_encoded, p_encoded, o_encoded)

                for i in range(len(self.test_set)):
                    # Tuple of three tuples
                    s, p, o = self.test_set[i]
                    if len(s) < max_len:
                        s_encoded = s + tuple(self.dummy_id for _ in range(max_len - len(s)))
                    else:
                        s_encoded = s
                    if len(p) < max_len:
                        p_encoded = p + tuple(self.dummy_id for _ in range(max_len - len(p)))
                    else:
                        p_encoded = p

                    if len(o) < max_len:
                        o_encoded = o + tuple(self.dummy_id for _ in range(max_len - len(o)))
                    else:
                        o_encoded = o
                    tokens.add(s_encoded)
                    tokens.add(p_encoded)
                    tokens.add(o_encoded)
                    self.test_set[i] = (s_encoded, p_encoded, o_encoded)

                # shaped_bpe_tokens
                self.ordered_shaped_bpe_tokens: List[Tuple[int, ..., int]]
                self.ordered_shaped_bpe_tokens = [shaped_bpe_token for shaped_bpe_token in tokens]

                # self.train_set = np.array(self.train_set)
                # self.test_set = np.array(self.test_set)
                # self.valid_set = np.array(self.valid_set)

            LoadSaveToDisk(kg=self).save()

            if self.eval_model:
                if self.valid_set is not None and self.test_set is not None:
                    if isinstance(self.valid_set, np.ndarray) and isinstance(self.test_set, np.ndarray):
                        data = np.concatenate([self.train_set, self.valid_set, self.test_set])
                    elif isinstance(self.valid_set, list) and isinstance(self.test_set, list):
                        data = self.train_set + self.valid_set + self.test_set
                    else:
                        raise KeyError(
                            f"Unrecognized type: valid_set {type(self.valid_set)} and test_set {type(self.test_set)}")
                else:
                    data = self.train_set
                """
                self.er_vocab = get_er_vocab(data, self.path_for_serialization + '/er_vocab.p')
                self.re_vocab = get_re_vocab(data, self.path_for_serialization + '/re_vocab.p')
                self.ee_vocab = get_ee_vocab(data, self.path_for_serialization + '/ee_vocab.p')
                if self.byte_pair_encoding is False:
                    self.constraints = create_constraints(self.train_set, self.path_for_serialization + '/constraints.p')
                """

                print('Submit er-vocab, re-vocab, and ee-vocab via  ProcessPoolExecutor...')
                # We need to benchmark the benefits of using futures  ?
                executor = concurrent.futures.ProcessPoolExecutor()
                self.er_vocab = executor.submit(get_er_vocab, data, self.path_for_serialization + '/er_vocab.p')
                self.re_vocab = executor.submit(get_re_vocab, data, self.path_for_serialization + '/re_vocab.p')
                self.ee_vocab = executor.submit(get_ee_vocab, data, self.path_for_serialization + '/ee_vocab.p')
                self.constraints = executor.submit(create_constraints, self.train_set,
                                                   self.path_for_serialization + '/constraints.p')
                self.domain_constraints_per_rel, self.range_constraints_per_rel = None, None

        else:
            LoadSaveToDisk(kg=self).load()

        # assert len(self.train_set) > 0
        # assert len(self.train_set[0]) > 0
        # assert isinstance(self.train_set, np.ndarray)
        # assert isinstance(self.train_set[0], np.ndarray)
        self._describe()

    def _describe(self) -> None:
        self.description_of_input = f'\n------------------- Description of Dataset {self.dataset_dir} -------------------'
        self.description_of_input += f'\nNumber of entities:{self.num_entities}' \
                                     f'\nNumber of relations:{self.num_relations}' \
                                     f'\nNumber of tokens:{self.num_tokens}' \
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
