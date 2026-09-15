import logging
import os

import numpy as np
import pandas
import polars

from dicee.static_funcs import save_numpy_ndarray, save_pickle

from .util import load_numpy_ndarray, load_pickle

logger = logging.getLogger(__name__)


class LoadSaveToDisk:
    def __init__(self, kg):
        self.kg = kg

    def save(self):
        assert self.kg.path_for_deserialization is None

        if self.kg.path_for_serialization is None:
            # No serialization
            return None

        if self.kg.byte_pair_encoding:
            save_numpy_ndarray(data=self.kg.train_set, file_path=self.kg.path_for_serialization + '/train_set.npy')
            logger.warning("NO SAVING for BPE at save_load_disk.py")
            # ordered_bpe_entities/relations are only computed by the padded BPE
            # preprocessing path (see preprocess.py); models like BytE that use
            # the non-padded path (padding=False) never set them.
            if hasattr(self.kg, 'ordered_bpe_entities') and hasattr(self.kg, 'ordered_bpe_relations'):
                save_pickle(data=self.kg.ordered_bpe_entities, file_path=self.kg.path_for_serialization + '/ordered_bpe_entities.p')
                save_pickle(data=self.kg.ordered_bpe_relations, file_path=self.kg.path_for_serialization + '/ordered_bpe_relations.p')
        else:
            assert isinstance(self.kg.train_set, np.ndarray)
            # (1) Save dictionary mappings into disk
            if isinstance(self.kg.entity_to_idx, dict):
                save_pickle(data=self.kg.entity_to_idx, file_path=self.kg.path_for_serialization + '/entity_to_idx.p')
                save_pickle(data=self.kg.relation_to_idx, file_path=self.kg.path_for_serialization + '/relation_to_idx.p')
            elif isinstance(self.kg.entity_to_idx, polars.DataFrame):
                self.kg.entity_to_idx.write_csv(file=self.kg.path_for_serialization + "/entity_to_idx.csv", include_header=True)
                self.kg.relation_to_idx.write_csv(file=self.kg.path_for_serialization + "/relation_to_idx.csv", include_header=True)
            elif isinstance(self.kg.entity_to_idx, pandas.DataFrame):
                self.kg.entity_to_idx.to_csv(path_or_buf=self.kg.path_for_serialization + "/entity_to_idx.csv", header=True)
                self.kg.relation_to_idx.to_csv(path_or_buf=self.kg.path_for_serialization + "/relation_to_idx.csv", header=True)
            else:
                raise RuntimeError("Unexpected type for entity_to_idx or relation_to_idx")

            save_numpy_ndarray(data=self.kg.train_set, file_path=self.kg.path_for_serialization + '/train_set.npy')
            if self.kg.valid_set is not None:
                save_numpy_ndarray(data=self.kg.valid_set, file_path=self.kg.path_for_serialization + '/valid_set.npy')
            if self.kg.test_set is not None:
                save_numpy_ndarray(data=self.kg.test_set, file_path=self.kg.path_for_serialization + '/test_set.npy')

    def load(self):
        assert self.kg.path_for_deserialization is not None
        assert self.kg.path_for_serialization == self.kg.path_for_deserialization

        # Backward compatible loading: prefer CSV, fallback to legacy pickle format.
        if (os.path.isfile(self.kg.path_for_deserialization + '/entity_to_idx.csv')
            and os.path.isfile(self.kg.path_for_deserialization + '/relation_to_idx.csv')):
            self.kg.entity_to_idx = pandas.read_csv(self.kg.path_for_deserialization + '/entity_to_idx.csv', index_col=0)
            self.kg.relation_to_idx = pandas.read_csv(self.kg.path_for_deserialization + '/relation_to_idx.csv', index_col=0)
        elif (os.path.isfile(self.kg.path_for_deserialization + '/entity_to_idx.p')
            and os.path.isfile(self.kg.path_for_deserialization + '/relation_to_idx.p')):
            self.kg.entity_to_idx = load_pickle(file_path=self.kg.path_for_deserialization + '/entity_to_idx.p')
            self.kg.relation_to_idx = load_pickle(file_path=self.kg.path_for_deserialization + '/relation_to_idx.p')
        else:
            raise FileNotFoundError(f"Could not find mapping files in {self.kg.path_for_deserialization}. "
                "Expected entity_to_idx/relation_to_idx as either .csv or .p")

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
            constraints_path = self.kg.path_for_deserialization + '/constraints.p'
            if os.path.isfile(constraints_path):
                self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = load_pickle(
                    file_path=constraints_path
                )
