import numpy as np
from .util import load_pickle, load_numpy_ndarray
import os
from dicee.static_funcs import save_pickle, save_numpy_ndarray


class LoadSaveToDisk:
    def __init__(self, kg):
        self.kg = kg

    def save(self):
        assert self.kg.path_for_deserialization is None

        if self.kg.path_for_serialization is None:
            # No serialization
            return None

        if self.kg.byte_pair_encoding:
            print("What shall be saved ?!")
            # self.kg.ordered_bpe_entities
            # self.kg.ordered_bpe_relations
            if self.kg.training_technique in ["KvsAll", "AllvsAll"]:
                save_numpy_ndarray(data=self.kg.train_set, file_path=self.kg.path_for_serialization + '/train_set.npy')

            assert self.kg.ordered_bpe_entities is not None
            assert self.kg.ordered_bpe_relations is not None
            save_pickle(data=self.kg.ordered_bpe_entities, file_path=self.kg.path_for_serialization + '/ordered_bpe_entities.p')
            save_pickle(data=self.kg.ordered_bpe_relations, file_path=self.kg.path_for_serialization + '/ordered_bpe_relations.p')
        else:
            assert isinstance(self.kg.entity_to_idx, dict)
            assert isinstance(self.kg.relation_to_idx, dict)
            assert isinstance(self.kg.train_set, np.ndarray)

            # (1) Save dictionary mappings into disk
            save_pickle(data=self.kg.entity_to_idx, file_path=self.kg.path_for_serialization + '/entity_to_idx.p')
            save_pickle(data=self.kg.relation_to_idx, file_path=self.kg.path_for_serialization + '/relation_to_idx.p')

            save_numpy_ndarray(data=self.kg.train_set, file_path=self.kg.path_for_serialization + '/train_set.npy')
            if self.kg.valid_set is not None:
                save_numpy_ndarray(data=self.kg.valid_set, file_path=self.kg.path_for_serialization + '/valid_set.npy')
            if self.kg.test_set is not None:
                save_numpy_ndarray(data=self.kg.test_set, file_path=self.kg.path_for_serialization + '/test_set.npy')

    def load(self):
        assert self.kg.path_for_deserialization is not None
        assert self.kg.path_for_serialization == self.kg.path_for_deserialization

        self.kg.entity_to_idx = load_pickle(file_path=self.kg.path_for_deserialization + '/entity_to_idx.p')
        self.kg.relation_to_idx = load_pickle(file_path=self.kg.path_for_deserialization + '/relation_to_idx.p')
        assert isinstance(self.kg.entity_to_idx, dict)
        assert isinstance(self.kg.relation_to_idx, dict)
        self.kg.num_entities = len(self.kg.entity_to_idx)
        self.kg.num_relations = len(self.kg.relation_to_idx)

        self.kg.train_set = load_numpy_ndarray(file_path=self.kg.path_for_deserialization + '/train_set.npy')

        if os.path.isfile(self.kg.path_for_deserialization + '/valid_set.npy'):
            self.kg.valid_set = load_numpy_ndarray(file_path=self.kg.path_for_deserialization + '/valid_set.npy')
        if os.path.isfile(self.kg.path_for_deserialization + '/test_set.npy'):
            self.kg.test_set = load_numpy_ndarray(file_path=self.kg.path_for_deserialization + '/test_set.npy')

        if self.kg.eval_model:
            self.kg.er_vocab = load_pickle(file_path=self.kg.path_for_deserialization + '/er_vocab.p')
            self.kg.re_vocab = load_pickle(file_path=self.kg.path_for_deserialization + '/re_vocab.p')
            self.kg.ee_vocab = load_pickle(file_path=self.kg.path_for_deserialization + '/ee_vocab.p')
            self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = load_pickle(
                file_path=self.kg.path_for_deserialization + '/constraints.p')
