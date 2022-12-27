import time

import numpy as np
import concurrent
from .util import *
import os
import time


class LoadSaveToDisk:
    def __init__(self, kg):
        self.kg = kg

    def save(self):
        assert self.kg.path_for_deserialization is None

        if self.kg.path_for_serialization is None:
            # No serialization
            return None

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

        if self.kg.eval_model:
            if self.kg.valid_set is not None and self.kg.test_set is not None:
                assert isinstance(self.kg.valid_set, np.ndarray) and isinstance(self.kg.test_set, np.ndarray)
                data = np.concatenate([self.kg.train_set, self.kg.valid_set, self.kg.test_set])
            else:
                data = self.kg.train_set
            # We need to parallelise the next four steps.
            print('Submit er-vocab, re-vocab, and ee-vocab via  ProcessPoolExecutor...')
            executor = concurrent.futures.ProcessPoolExecutor()
            self.kg.er_vocab = executor.submit(get_er_vocab, data, self.kg.path_for_serialization + '/er_vocab.p')
            self.kg.re_vocab = executor.submit(get_re_vocab, data, self.kg.path_for_serialization + '/re_vocab.p')
            self.kg.ee_vocab = executor.submit(get_ee_vocab, data, self.kg.path_for_serialization + '/ee_vocab.p')
            self.kg.constraints = executor.submit(create_constraints, self.kg.train_set,
                                                  self.kg.path_for_serialization + '/constraints.p')
            self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = None, None

    def load(self):
        assert self.kg.path_for_deserialization is not None
        assert self.kg.path_for_serialization == self.kg.path_for_deserialization

        self.kg.entity_to_idx = load_pickle(file_path=self.kg.path_for_deserialization + '/entity_to_idx.p')
        self.kg.relation_to_idx = load_pickle(file_path=self.kg.path_for_deserialization + '/relation_to_idx.p')
        assert isinstance(self.kg.entity_to_idx, dict)
        assert isinstance(self.kg.relation_to_idx, dict)

        self.kg.train_set = load_numpy_ndarray(file_path=self.kg.path_for_deserialization + '/train_set.npy')

        if os.path.isfile(self.kg.path_for_deserialization + '/valid_set.npy'):
            self.kg.valid_set = load_numpy_ndarray(file_path=self.kg.path_for_deserialization + '/valid_set.npy')
        if os.path.isfile(self.kg.path_for_deserialization + '/test_set.npy'):
            self.kg.test_set = load_numpy_ndarray(file_path=self.kg.path_for_deserialization + '/test_set.npy')

        if self.kg.eval_model:
            self.kg.er_vocab = load_pickle(file_path=self.kg.path_for_deserialization + '/er_vocab.p')
            self.kg.re_vocab = load_pickle(file_path=self.kg.path_for_deserialization + '/re_vocab.p')
            self.kg.ee_vocab = load_pickle(file_path=self.kg.path_for_deserialization + '/ee_vocab.p')
            self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = load_pickle(file_path=self.kg.path_for_deserialization + '/constraints.p')