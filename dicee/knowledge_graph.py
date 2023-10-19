from typing import List
from .read_preprocess_save_load_kg import ReadFromDisk, PreprocessKG, LoadSaveToDisk
import sys


class CharEncoder:
    def __init__(self, chars):
        # import tiktoken
        # self.enc = tiktoken.get_encoding("gpt2")
        # self.num_tokens = self.enc.n_vocab
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_chat = {i: ch for i, ch in enumerate(chars)}
        self.n_vocab = len(self.char_to_idx)

    def encode(self, x: str):
        return [self.char_to_idx[c] for c in x]

    def decode(self, x):
        return ''.join([self.idx_to_char[i] for i in x])


class KG:
    """ Knowledge Graph """

    def __init__(self, dataset_dir: str = None,
                 bpe: bool = False,
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
        :param bpe: Apply Byte pair encoding.
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
        self.bpe = bpe
        self.sparql_endpoint = sparql_endpoint
        self.add_noise_rate = add_noise_rate
        self.num_entities = None
        self.num_relations = None
        self.num_tokens = None
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
            # WIP:
            if self.bpe:
                self.train_set["sentence"] = self.train_set["subject"] + " " + self.train_set[
                    "relation"] + " " + self.train_set["object"]
                self.train_set.drop(columns=["subject", "relation", "object"], inplace=True)
                text = "\n".join(self.train_set["sentence"].to_list())

                self.enc=CharEncoder(sorted(list(set(text))))
                self.num_tokens = self.enc.n_vocab

                self.train_set = self.enc.encode("\n".join(self.train_set["sentence"].to_list()))

                if self.valid_set is not None:
                    self.valid_set["sentence"] = self.valid_set["subject"] + " " + self.valid_set[
                        "relation"] + " " + self.valid_set["object"]
                    self.valid_set.drop(columns=["subject", "relation", "object"], inplace=True)
                    self.valid_set = self.enc.encode("\n".join(self.valid_set["sentence"].to_list()))

                if self.test_set is not None:
                    self.test_set["sentence"] = self.test_set["subject"] + " " + self.test_set[
                        "relation"] + " " + self.test_set["object"]
                    self.test_set.drop(columns=["subject", "relation", "object"], inplace=True)
                    self.test_set = self.enc.encode("\n".join(self.test_set["sentence"].to_list()))

            else:
                PreprocessKG(kg=self).start()
                LoadSaveToDisk(kg=self).save()
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
        # self.description_of_input += f"Train set :{self.train_set.nbytes / 1_000_000_000:.5f} in GB\n"

    @property
    def entities_str(self) -> List:
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List:
        return list(self.relation_to_idx.keys())
