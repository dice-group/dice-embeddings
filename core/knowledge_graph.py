import time
from typing import Dict, List
from collections import defaultdict
import numpy as np
import pickle
import json
from dask import dataframe as ddf
import os
import pandas as pd
from .static_funcs import performance_debugger, get_er_vocab, get_ee_vocab, get_re_vocab
import glob

np.random.seed(1)
pd.set_option('display.max_columns', None)


class KG:
    """ Knowledge Graph Class
        1- Reading : Large input data is read via DASK
        2- Cleaning & Preprocessing :
                                    Remove triples with literals if exists
                                    Apply reciprocal data augmentation triples into train, valid and test datasets
                                    Add noisy triples (random facts sampled from all possible triples E x R x E)
        3- Serializing and Deserializing in parquet format
    """

    def __init__(self, data_dir: str = None, deserialize_flag: str = None, large_kg_parse=False, add_reciprical=False,
                 eval_model=True, read_only_few: int = None, sample_triples_ratio: float = None,
                 path_for_serialization: str = None, add_noise_rate: float = None, entity_to_idx=None,
                 relation_to_idx=None):
        """

        :param data_dir: A path of a folder containing the input knowledge graph
        :param deserialize_flag: A path of a folder containing previously parsed data
        :param large_kg_parse: A flag for using all cores to parse input knowledge graph
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied. If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        """

        # TODO: If input KG is really large:
        # Call a c++ code to partition this graph into many train.txt files
        # Add call dask read_csv(*) to read all of it in parallel.
        if deserialize_flag is None:
            # 1. LOAD Data. (First pass on data)
            print(
                f'[1 / 14] Loading training data: large_kg_parse: {large_kg_parse}, read_only_few: {read_only_few} , sample_triples_ratio: {sample_triples_ratio}...')
            self.train_set = self.load_data_parallel(data_dir + '/train', large_kg_parse, read_only_few,
                                                     sample_triples_ratio)
            print('Done !\n')
            print(
                f'[2 / 14] Loading valid data: large_kg_parse: {large_kg_parse}, read_only_few: {read_only_few}, sample_triples_ratio: {sample_triples_ratio}...')

            self.valid_set = self.load_data_parallel(data_dir + '/valid', large_kg_parse, read_only_few,
                                                     sample_triples_ratio)
            print('Done !\n')
            print(
                f'[3 / 14] Loading test data: large_kg_parse: {large_kg_parse}, read_only_few: {read_only_few}, sample_triples_ratio: {sample_triples_ratio}...')

            self.test_set = self.load_data_parallel(data_dir + '/test', large_kg_parse, read_only_few,
                                                    sample_triples_ratio)
            print('Done !\n')

            # 2. Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
            if add_reciprical and eval_model:
                print(
                    '[3.1 / 14] Add reciprocal triples to train, validation, and test sets, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}')

                self.train_set = pd.concat([self.train_set,
                                            pd.DataFrame({'subject': self.train_set['object'],
                                                          'relation': self.train_set['relation'].map(
                                                              lambda x: x + '_inverse'),
                                                          'object': self.train_set['subject']})], ignore_index=True)
                if len(self.valid_set) > 0:
                    self.valid_set = pd.concat([self.valid_set,
                                                pd.DataFrame({'subject': self.valid_set['object'],
                                                              'relation': self.valid_set['relation'].map(
                                                                  lambda x: x + '_inverse'),
                                                              'object': self.valid_set['subject']})], ignore_index=True)
                if len(self.test_set) > 0:
                    self.test_set = pd.concat([self.test_set,
                                               pd.DataFrame({'subject': self.test_set['object'],
                                                             'relation': self.test_set['relation'].map(
                                                                 lambda x: x + '_inverse'),
                                                             'object': self.test_set['subject']})], ignore_index=True)
                print('Done !\n')

            if add_noise_rate is not None:
                num_noisy_triples = int(len(self.train_set) * add_noise_rate)
                print(f'[4 / 14] Generating {num_noisy_triples} noisy triples for training data...')
                s = len(self.train_set)
                list_of_entities = pd.unique(self.train_set[['subject', 'object']].values.ravel('K'))

                self.train_set = pd.concat([self.train_set,
                                            # Noisy triples
                                            pd.DataFrame(
                                                {'subject': np.random.choice(list_of_entities, num_noisy_triples),
                                                 'relation': np.random.choice(
                                                     pd.unique(self.train_set[['relation']].values.ravel('K')),
                                                     num_noisy_triples),
                                                 'object': np.random.choice(list_of_entities, num_noisy_triples)}
                                            )
                                            ], ignore_index=True)

                del list_of_entities

                assert s + num_noisy_triples == len(self.train_set)

            if entity_to_idx is None and relation_to_idx is None:
                # 3. Concatenate dataframes.
                print(f'[4 / 14] Concatenating data to obtain index...')
                df_str_kg = pd.concat([self.train_set, self.valid_set, self.test_set], ignore_index=True)
                print('Done !\n')
                # 4. Create a bijection mapping  from entities to integer indexes.
                print('[5 / 14] Creating a mapping from entities to integer indexes...')
                ordered_list = pd.unique(df_str_kg[['subject', 'object']].values.ravel('K'))
                self.entity_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                                  columns=['entity'],
                                                  index=ordered_list)
                print('Done !\n')

                # 5. Create a bijection mapping  from relations to integer indexes.
                print('[6 / 14] Creating a mapping from relations to integer indexes...')
                ordered_list = pd.unique(df_str_kg['relation'].values.ravel('K'))
                self.relation_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                                    columns=['relation'],
                                                    index=ordered_list)
                print('Done !\n')
                # Free memory
                del ordered_list, df_str_kg

                ## 6. Serialize indexed entities and relations into disk for further usage.
                print('[7 / 14] Serializing compressed entity integer mapping...')
                self.entity_to_idx.to_parquet(path_for_serialization + '/entity_to_idx.gzip', compression='gzip')
                print('Done !\n')

                print('[8 / 14] Serializing compressed relation integer mapping...')
                self.relation_to_idx.to_parquet(path_for_serialization + '/relation_to_idx.gzip', compression='gzip')
                print('Done !\n')

                # 7. Convert from pandas dataframe to dictionaries for an easy access
                # We may want to benchmark using python dictionary and pandas frame
                print(
                    '[9 / 14] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...')
                self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
                self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
                self.num_entities = len(self.entity_to_idx)
                self.num_relations = len(self.relation_to_idx)
                print('Done !\n')
            else:
                print('[4 / 14] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...')
                # Time consuming
                self.entity_to_idx = entity_to_idx.to_dict()['entity']
                self.relation_to_idx = relation_to_idx.to_dict()['relation']
                self.num_entities = len(self.entity_to_idx)
                self.num_relations = len(self.relation_to_idx)

            if path_for_serialization is not None:
                # 8. Serialize already read training data in parquet format so that
                # the training data is stored in more memory efficient manner as well as
                # it can be reread later faster
                print('[10 / 14] Serializing training data for Continual Learning...')  # TODO: Do we really need it ?!
                self.train_set.to_parquet(path_for_serialization + '/train_df.gzip', compression='gzip')
                print('Done !\n')

            print('[11 / 14] Mapping training data into integers for training...')
            # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
            self.train_set['subject'] = self.train_set['subject'].map(
                lambda x: self.entity_to_idx[x] if self.entity_to_idx.get(x) else None)
            self.train_set['relation'] = self.train_set['relation'].map(
                lambda x: self.relation_to_idx[x] if self.relation_to_idx.get(x) else None)
            self.train_set['object'] = self.train_set['object'].map(
                lambda x: self.entity_to_idx[x] if self.entity_to_idx.get(x) else None)
            self.train_set.dropna(inplace=True)
            self.train_set = self.train_set.astype(int)

            print('Done !\n')
            if path_for_serialization is not None:
                # 10. Serialize (9).
                print('[12 / 14] Serializing integer mapped data...')  # TODO: Do we really need it ?!
                self.train_set.to_parquet(path_for_serialization + '/idx_train_df.gzip', compression='gzip')
                print('Done !\n')

            # 11. Convert data from pandas dataframe to numpy ndarray.
            print('[13 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...')
            self.train_set = self.train_set.values
            print('Done !\n')

            print('[14 / 14 ] Sanity checking...')
            # 12. Sanity checking: indexed training set can not have an indexed entity assigned with larger indexed than the number of entities.
            assert self.num_entities > max(self.train_set[:, 0]) and self.num_entities > max(self.train_set[:, 2])
            assert self.num_relations > max(self.train_set[:, 1])
            # 13. Sanity checking: data types
            assert isinstance(self.train_set[0], np.ndarray)
            assert isinstance(self.train_set[0][0], np.int64) and isinstance(self.train_set[0][1], np.int64)
            assert isinstance(self.train_set[0][2], np.int64)
            # 14. Repeat computations carried out from 8-13 on validation dataset.
            if len(self.valid_set) > 0:
                if path_for_serialization is not None:
                    print('[15 / 14 ] Serializing validation data for Continual Learning...')
                    self.valid_set.to_parquet(path_for_serialization + '/valid_df.gzip', compression='gzip')
                self.valid_set['subject'] = self.valid_set['subject'].map(lambda x: self.entity_to_idx[x])
                self.valid_set['relation'] = self.valid_set['relation'].map(lambda x: self.relation_to_idx[x])
                self.valid_set['object'] = self.valid_set['object'].map(lambda x: self.entity_to_idx[x])
                if path_for_serialization is not None:
                    self.valid_set.to_parquet(path_for_serialization + '/idx_valid_df.gzip', compression='gzip')
                self.valid_set = self.valid_set.values
                # Sanity checking
                assert self.num_entities > max(self.valid_set[0])
                assert self.num_entities > max(self.valid_set[0])
                assert self.num_entities > max(self.valid_set[2])
                assert self.num_entities > max(self.valid_set[2])

                assert isinstance(self.valid_set[0], np.ndarray)
                assert isinstance(self.valid_set[0][0], np.int64)
                assert isinstance(self.valid_set[0][1], np.int64)
                assert isinstance(self.valid_set[0][2], np.int64)
                print('Done !\n')
            else:
                self.valid_set = self.valid_set.values
            # 15. Repeat computations carried out from 8-13 on test dataset.
            if len(self.test_set) > 0:
                if path_for_serialization is not None:
                    print('[16 / 14 ] Serializing test data for Continual Learning...')
                    self.test_set.to_parquet(path_for_serialization + '/test_df.gzip', compression='gzip')
                self.test_set['subject'] = self.test_set['subject'].map(lambda x: self.entity_to_idx[x])
                self.test_set['relation'] = self.test_set['relation'].map(lambda x: self.relation_to_idx[x])
                self.test_set['object'] = self.test_set['object'].map(lambda x: self.entity_to_idx[x])
                if path_for_serialization is not None:
                    self.test_set.to_parquet(path_for_serialization + '/idx_test_df.gzip', compression='gzip')
                self.test_set = self.test_set.values
                # Sanity checking
                assert self.num_entities > max(self.test_set[0])
                assert self.num_entities > max(self.test_set[0])
                assert self.num_entities > max(self.test_set[2])
                assert self.num_entities > max(self.test_set[2])

                assert isinstance(self.test_set[0], np.ndarray)
                assert isinstance(self.test_set[0][0], np.int64)
                assert isinstance(self.test_set[0][1], np.int64)
                assert isinstance(self.test_set[0][2], np.int64)
                print('Done !\n')
            else:
                self.test_set = self.test_set.values

            if eval_model:  # and len(self.valid_set) > 0 and len(self.test_set) > 0:
                if len(self.valid_set) > 0 and len(self.test_set) > 0:
                    # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                    data = np.concatenate([self.train_set, self.valid_set, self.test_set])
                else:
                    data = self.train_set
                self.er_vocab = get_er_vocab(data)
                self.re_vocab = get_re_vocab(data)
                # 17. Create a bijection mapping from subject-object pairs to relations.
                self.ee_vocab = get_ee_vocab(data)

        else:
            self.deserialize(deserialize_flag)
            if eval_model:  # and len(self.valid_set) > 0 and len(self.test_set) > 0:
                if len(self.valid_set) > 0 and len(self.test_set) > 0:
                    # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                    data = np.concatenate([self.train_set, self.valid_set, self.test_set])
                else:
                    data = self.train_set
                self.er_vocab = get_er_vocab(data)
                self.re_vocab = get_re_vocab(data)
                # 17. Create a bijection mapping from subject-object pairs to relations.
                self.ee_vocab = get_ee_vocab(data)

        # 4. Display info
        self.description_of_input = f'\n------------------- Description of Dataset {data_dir} -------------------'
        self.description_of_input += f'\nNumber of entities: {self.num_entities}' \
                                     f'\nNumber of relations: {self.num_relations}' \
                                     f'\nNumber of triples on train set: {len(self.train_set)}' \
                                     f'\nNumber of triples on valid set: {len(self.valid_set)}' \
                                     f'\nNumber of triples on test set: {len(self.test_set)}\n'

    def deserialize(self, storage_path: str) -> None:
        """ Deserialize data """
        print(f'Deserialization Path Path: {storage_path}\n')
        print('Deserializing compressed entity integer mapping...')
        self.entity_to_idx = ddf.read_parquet(storage_path + '/entity_to_idx.gzip').compute()
        print('Done!\n')
        self.num_entities = len(self.entity_to_idx)
        print('Deserializing compressed relation integer mapping...')
        self.relation_to_idx = ddf.read_parquet(storage_path + '/relation_to_idx.gzip').compute()
        self.num_relations = len(self.relation_to_idx)
        print('Done!\n')
        print(
            'Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...')
        self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
        self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
        print('Done!\n')
        # 10. Serialize (9).
        print('Deserializing integer mapped data and mapping it to numpy ndarray...')
        self.train_set = ddf.read_parquet(storage_path + '/idx_train_df.gzip').values.compute()
        print('Done!\n')
        try:
            print('Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.valid_set = ddf.read_parquet(storage_path + '/idx_valid_df.gzip').values.compute()
            print('Done!\n')
        except FileNotFoundError:
            print('No valid data found!\n')
            self.valid_set = pd.DataFrame()

        try:
            print('Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.test_set = ddf.read_parquet(storage_path + '/idx_test_df.gzip').values.compute()
            print('Done!\n')
        except FileNotFoundError:
            print('No test data found\n')
            self.test_set = pd.DataFrame()

        with open(storage_path + '/configuration.json', 'r') as f:
            args = json.load(f)
        """
        if args['eval']:
            if len(self.valid_set) > 0 and len(self.test_set) > 0:
                # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                data = np.concatenate([self.train_set, self.valid_set, self.test_set])
            else:
                data = self.train_set
            self.er_vocab = get_er_vocab(data)
        """
    @staticmethod
    def load_data_parallel(data_path, large_kg_parse=True, read_only_few: int = None,
                           sample_triples_ratio: float = None) -> List:
        """
        Parse KG via DASK.
        :param read_only_few:
        :param data_path:
        :param large_kg_parse:
        :param sample_triples_ratio:
        :return:
        """
        # (1) Check file exists, .e.g, ../../train.* exists

        if glob.glob(data_path + '*'):
            # (1) Read knowledge graph  via
            # (1.1) Using the whitespace as a deliminator
            # (1.2) Taking first three columns detected in (1.1.)
            # Task would even allow us to read compressed KGs.
            df = ddf.read_csv(data_path + '*', delim_whitespace=True,
                              header=None, usecols=[0, 1, 2],
                              names=['subject', 'relation', 'object'], dtype=str)
            if isinstance(read_only_few, int):
                if read_only_few > 0:
                    df = df.loc[:read_only_few]
            if sample_triples_ratio:
                print(f'Subsampling {sample_triples_ratio} of input data...')
                df = df.sample(frac=sample_triples_ratio)

            # Drop rows having ^^
            df = df[df["object"].str.contains("<http://www.w3.org/2001/XMLSchema#double>") == False]
            df = df[df["object"].str.contains("<http://www.w3.org/2001/XMLSchema#boolean>") == False]
            df['subject'] = df['subject'].str.removeprefix("<").str.removesuffix(">")
            df['relation'] = df['relation'].str.removeprefix("<").str.removesuffix(">")
            df['object'] = df['object'].str.removeprefix("<").str.removesuffix(">")
            print('Dask Scheduler starts computation...')
            if large_kg_parse:
                df = df.compute(scheduler='processes')
            else:
                df = df.compute(scheduler='single-threaded')
            num_triples, y = df.shape
            assert y == 3
            return df
        else:
            print(f'{data_path} could not found!\n')
            return pd.DataFrame()

    def is_valid_test_available(self):
        if len(self.valid_set) > 0 and len(self.test_set) > 0:
            return True
        return False

    @property
    def entities_str(self) -> List:
        """
        entity_idx is a dictionary where keys are string representation of entities and
        values are integer indexes
        :return: list of ordered entities
        """
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List:
        """
        relation_idx is a dictionary where keys are string representation of relations and
        values are integer indexes
        :return: list of ordered relations
        """
        return list(self.relation_to_idx.keys())

    @staticmethod
    def map_str_triples_to_numpy_idx(triples, entity_idx, relation_idx) -> np.array:
        return np.array([(entity_idx[s], relation_idx[p], entity_idx[o]) for s, p, o in triples])
