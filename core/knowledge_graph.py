from typing import Dict, List, Tuple, Union
import numpy as np
from .read_preprocess_save_load_kg import ReadFromDisk, PreprocessKG, LoadSaveToDisk

class KG:
    """ Knowledge Graph """

    def __init__(self, data_dir: str = None, path_for_deserialization: str = None,
                 num_core: int = 0,
                 add_reciprical: bool = None, eval_model: str = None,
                 read_only_few: int = None, sample_triples_ratio: float = None,
                 path_for_serialization: str = None, add_noise_rate: float = None,
                 min_freq_for_vocab: int = None,
                 entity_to_idx=None, relation_to_idx=None, backend=None):
        """
        :param data_dir: A path of a folder containing the input knowledge graph
        :param path_for_deserialization: A path of a folder containing previously parsed data
        :param num_core: Number of subprocesses used for data loading
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied. If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        """
        self.num_entities = None
        self.num_relations = None
        self.data_dir = data_dir
        self.path_for_deserialization = path_for_deserialization
        self.num_core = num_core
        self.add_reciprical = add_reciprical
        self.eval_model = eval_model

        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.path_for_serialization = path_for_serialization
        self.add_noise_rate = add_noise_rate

        self.min_freq_for_vocab = min_freq_for_vocab
        # dicts of str to int
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.backend = 'pandas' if backend is None else backend
        self.train_set, self.valid_set, self.test_set = None, None, None

        if self.path_for_deserialization is None:
            ReadFromDisk(kg=self).start()
            PreprocessKG(kg=self).start()
            LoadSaveToDisk(kg=self).save()
        else:
            LoadSaveToDisk(kg=self).load()

        assert len(self.train_set) > 0
        assert len(self.train_set[0]) > 0
        assert isinstance(self.train_set, np.ndarray)
        assert isinstance(self.train_set[0], np.ndarray)

        self.__describe()

    def __describe(self) -> None:
        self.description_of_input = f'\n------------------- Description of Dataset {self.data_dir} -------------------'
        self.description_of_input += f'\nNumber of entities: {self.num_entities}' \
                                     f'\nNumber of relations: {self.num_relations}' \
                                     f'\nNumber of triples on train set: {len(self.train_set)}' \
                                     f'\nNumber of triples on valid set: {len(self.valid_set) if self.valid_set is not None else 0}' \
                                     f'\nNumber of triples on test set: {len(self.test_set) if self.test_set is not None else 0}\n'

    @property
    def entities_str(self) -> List:
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List:
        return list(self.relation_to_idx.keys())
