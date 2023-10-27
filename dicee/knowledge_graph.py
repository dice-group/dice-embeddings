from typing import List
from .read_preprocess_save_load_kg import ReadFromDisk, PreprocessKG, LoadSaveToDisk
import sys
import tiktoken


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
                 entity_to_idx=None, relation_to_idx=None, backend=None, training_technique: str = None):
        """
        :param dataset_dir: A path of a folder containing train.txt, valid.txt, test.text
        :param byte_pair_encoding: Apply Byte pair encoding.
        :param add_noise_rate: Noisy triples added into the training adataset by x % of its size.
        :param sparql_endpoint: An endpoint of a triple store
        :param path_single_kg: The path of a single file containing the input knowledge graph
        :param path_for_deserialization: A path of a folder containing previously parsed data
        :param num_core: Number of subprocesses used for data loading
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied.
        If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        :param training_technique
        """
        self.dataset_dir = dataset_dir
        self.byte_pair_encoding = byte_pair_encoding
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
        self.training_technique = training_technique
        self.raw_train_set, self.raw_valid_set, self.raw_test_set = None, None, None
        self.train_set, self.valid_set, self.test_set = None, None, None
        self.idx_entity_to_bpe_shaped = dict()

        # WIP:
        self.enc = tiktoken.get_encoding("gpt2")
        self.num_tokens = self.enc.n_vocab  # ~ 50
        self.num_bpe_entities = None
        # TODO: Find a unique token later
        self.dummy_id = self.enc.encode(" ")[0]
        self.max_length_subword_tokens = None
        self.train_set_target = None
        self.target_dim = None
        self.train_target_indices = None
        self.ordered_bpe_entities = None

        if self.path_for_deserialization is None:
            ReadFromDisk(kg=self).start()
            PreprocessKG(kg=self).start()
            LoadSaveToDisk(kg=self).save()

        else:
            LoadSaveToDisk(kg=self).load()

        assert len(self.train_set) > 0

        """
        if self.byte_pair_encoding:
            assert isinstance(self.train_set[0], tuple) and len(self.train_set[0]) == 3
            assert isinstance(self.train_set[0][0], tuple) and isinstance(self.train_set[0][1], tuple)
            assert isinstance(self.train_set[0][2], tuple)
            assert isinstance(self.target_dim,int)

        else:
            assert isinstance(self.train_set, np.ndarray)
            assert isinstance(self.train_set[0], np.ndarray)
        """
        self._describe()

    def _describe(self) -> None:
        self.description_of_input = f'\n------------------- Description of Dataset {self.dataset_dir} -------------------'
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

    def func_triple_to_bpe_representation(self, triple:List[str]):
        result=[]

        for x in triple:
            unshaped_bpe_repr = self.enc.encode(x)
            if len(unshaped_bpe_repr) < self.max_length_subword_tokens:
                unshaped_bpe_repr.extend([self.dummy_id for _ in
                                            range(self.max_length_subword_tokens - len(unshaped_bpe_repr))])
            else:
                pass
            result.append(unshaped_bpe_repr)
        return result
