import time
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import pickle
import json
import dask
from dask import dataframe as ddf
import os
import pandas as pd
from .static_funcs import performance_debugger, get_er_vocab, get_ee_vocab, get_re_vocab, \
    create_recipriocal_triples_from_dask, add_noisy_triples, index_triples, load_data_parallel, numpy_data_type_changer
from .sanity_checkers import dataset_sanity_checking
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
                 path_for_serialization: str = None, add_noise_rate: float = None,
                 min_freq_for_vocab: int = None, entity_to_idx=None,
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
        self.num_entities = None
        self.num_relations = None
        self.df_str_kg = None
        self.data_dir = data_dir
        self.deserialize_flag = deserialize_flag
        self.large_kg_parse = large_kg_parse
        self.add_reciprical = add_reciprical
        self.eval_model = eval_model

        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.large_kg_parse = large_kg_parse
        self.path_for_serialization = path_for_serialization
        self.add_noise_rate = add_noise_rate

        self.min_freq_for_vocab = min_freq_for_vocab
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        if large_kg_parse:
            self.scheduler_flag = 'processes'
        else:
            self.scheduler_flag = 'single-threaded'

        """
        # @ TODO Integrate LocalCluster facility to analyse the utilization of the hardware via the dashboard
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster()
        client = Client(cluster)
        print(client)
        print(client.dashboard_link)
        """
        # (1) Load + Preprocess input data
        if deserialize_flag is None:
            # (1.1) Load and Preprocess the data.
            self.train_set, self.valid_set, self.test_set = self.load_read_process()
            # (1.2) Update (1.1).
            self.apply_reciprical_or_noise()
            # (1.3) Construct integer indexing for entities and relations
            if entity_to_idx is None and relation_to_idx is None:
                # self.parallel_vocabulary_construction()  # via DASK
                self.sequential_vocabulary_construction()  # via Pandas
                print(
                    '[9 / 14] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...',
                    end='\t')
                self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
                self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
                self.num_entities = len(self.entity_to_idx)
                self.num_relations = len(self.relation_to_idx)
                print('Done !\n')
                print('[10 / 14] Mapping training data into integers for training...', end='\t')
                start_time = time.time()
                # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
                # @TODO: Benchmark pandasswifter vs Panddas vs DASK on large dataset.
                self.train_set = index_triples(self.train_set,
                                               self.entity_to_idx,
                                               self.relation_to_idx,
                                               multi_processing=False)
                print(f'Done ! {time.time() - start_time:.3f} seconds\n')
                if path_for_serialization is not None:
                    # 10. Serialize (9).
                    print('[11 / 14] Serializing integer mapped data...', end='\t')
                    self.train_set.to_parquet(path_for_serialization + '/idx_train_df.gzip', compression='gzip')
                    print('Done !\n')
                assert isinstance(self.train_set, pd.core.frame.DataFrame)
                # 11. Convert data from pandas dataframe to numpy ndarray.
                print('[12 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...', end='\t')
                self.train_set = self.train_set.values
                print('Done !\n')
            else:
                # self.parallel_vocabulary_construction()
                # self.sequential_vocabulary_construction()

                print(
                    '[4 / 14] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...',
                    end='\t')
                self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
                self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
                self.num_entities = len(self.entity_to_idx)
                self.num_relations = len(self.relation_to_idx)
                print('Done !\n')
                print('[10 / 14] Mapping training data into integers for training...', end='\t')
                # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
                self.train_set = index_triples(self.train_set, self.entity_to_idx, self.relation_to_idx)
                print('Done !\n')
                print('Train set compute...', end='\t')
                self.train_set = self.train_set.compute()
                if self.valid_set is not None:
                    print('Valid set compute...', end='\t')
                    self.valid_set = self.valid_set.compute()
                if self.test_set is not None:
                    print('Test set set compute...', end='\t')
                    self.test_set = self.test_set.compute()
                assert isinstance(self.train_set, pd.core.frame.DataFrame)
                # 11. Convert data from pandas dataframe to numpy ndarray.
                print('[12 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...', end='\t')
                self.train_set = self.train_set.values

            self.train_set = numpy_data_type_changer(self.train_set, num=max(self.num_entities, self.num_relations))

            print('[13 / 14 ] Sanity checking...', end='\t')
            # 12. Sanity checking: indexed training set can not have an indexed entity assigned with larger indexed than the number of entities.
            dataset_sanity_checking(self.train_set, self.num_entities, self.num_relations)
            print('Done !\n')
            if self.valid_set is not None:
                if path_for_serialization is not None:
                    print('[14 / 14 ] Serializing validation data for Continual Learning...', end='\t')
                    self.valid_set.to_parquet(
                        path_for_serialization + '/valid_df.gzip', compression='gzip')
                    print('Done !\n')
                print('[14 / 14 ] Indexing validation dataset...', end='\t')
                self.valid_set = index_triples(self.valid_set, self.entity_to_idx, self.relation_to_idx)
                print('Done !\n')
                if path_for_serialization is not None:
                    print('[15 / 14 ] Serializing indexed validation dataset...', end='\t')
                    self.valid_set.to_parquet(
                        path_for_serialization + '/idx_valid_df.gzip', compression='gzip')
                    print('Done !\n')
                # To numpy
                self.valid_set = self.valid_set.values  # .compute(scheduler=scheduler_flag)
                dataset_sanity_checking(self.valid_set, self.num_entities, self.num_relations)
                self.valid_set = numpy_data_type_changer(self.valid_set, num=max(self.num_entities, self.num_relations))
            if self.test_set is not None:
                if path_for_serialization is not None:
                    print('[16 / 14 ] Serializing test data for Continual Learning...', end='\t')
                    self.test_set.to_parquet(
                        path_for_serialization + '/test_df.gzip', compression='gzip')
                    print('Done !\n')
                print('[17 / 14 ] Indexing test dataset...', end='\t')
                self.test_set = index_triples(self.test_set, self.entity_to_idx, self.relation_to_idx)
                print('Done !\n')
                if path_for_serialization is not None:
                    print('[18 / 14 ] Serializing indexed test dataset...', end='\t')
                    self.test_set.to_parquet(
                        path_for_serialization + '/idx_test_df.gzip', compression='gzip')
                # To numpy
                self.test_set = self.test_set.values
                dataset_sanity_checking(self.test_set, self.num_entities, self.num_relations)
                self.test_set = numpy_data_type_changer(self.test_set, num=max(self.num_entities, self.num_relations))
                print('Done !\n')
            if eval_model:  # and len(self.valid_set) > 0 and len(self.test_set) > 0:
                if self.valid_set is not None and self.test_set is not None:
                    assert isinstance(self.valid_set, np.ndarray) and isinstance(self.test_set, np.ndarray)
                    # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                    data = np.concatenate([self.train_set, self.valid_set, self.test_set])
                else:
                    data = self.train_set
                # TODO do it via dask: No need to wait here.
                print('Creating Vocab...', end='\t')
                self.er_vocab = get_er_vocab(data)
                self.re_vocab = get_re_vocab(data)
                # 17. Create a bijection mapping from subject-object pairs to relations.
                self.ee_vocab = get_ee_vocab(data)
        else:
            self.deserialize(deserialize_flag)
            if eval_model:  # and len(self.valid_set) > 0 and len(self.test_set) > 0:
                if self.valid_set is not None and self.test_set is not None:
                    # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                    data = np.concatenate([self.train_set, self.valid_set, self.test_set])
                else:
                    data = self.train_set
                print('Creating Vocab...', end='\t')
                self.er_vocab = get_er_vocab(data)
                self.re_vocab = get_re_vocab(data)
                # 17. Create a bijection mapping from subject-object pairs to relations.
                self.ee_vocab = get_ee_vocab(data)

        # 4. Display info
        self.description_of_input = f'\n------------------- Description of Dataset {data_dir} -------------------'
        self.description_of_input += f'\nNumber of entities: {self.num_entities}' \
                                     f'\nNumber of relations: {self.num_relations}' \
                                     f'\nNumber of triples on train set: {len(self.train_set)}' \
                                     f'\nNumber of triples on valid set: {len(self.valid_set) if self.valid_set is not None else 0}' \
                                     f'\nNumber of triples on test set: {len(self.test_set) if self.test_set is not None else 0}\n'

    def parallel_vocabulary_construction(self):
        """
        (1) Concatenate dataframes/ train valid test sets

        (2) Remove triples with specified conditions if such conditions are given.

        (3) Create a bijection mapping from entities to integer indexes.
        """
        # 1. Concatenate dataframes.
        print('[4 / 14] Lazy Concatenating data to obtain index...', end='\t')
        x = [self.train_set]
        if self.valid_set is not None:
            x.append(self.valid_set)
        if self.test_set is not None:
            x.append(self.test_set)
        self.df_str_kg = ddf.concat(x, ignore_index=True)
        del x
        print('Done !\n')
        # (2) Remove triples from (1).
        self.remove_triples_with_condition()
        print('[5 / 14] Lazy Creating a mapping from entities to integer indexes...', end='\t')
        # (3) Create a bijection mapping from entities of (2) to integer indexes.
        self.entity_to_idx = dask.array.concatenate(
            [self.df_str_kg['subject'], self.df_str_kg['object']]).to_dask_dataframe(
            columns=['entity']).drop_duplicates()
        print('Computing entity indexes...', end='\t')
        start_time = time.time()
        # Takes time even in the lazy mode.
        self.entity_to_idx = self.entity_to_idx.set_index(self.entity_to_idx.entity)
        self.entity_to_idx['entity'] = dask.array.arange(0,
                                                         self.entity_to_idx.size.compute(scheduler=self.scheduler_flag))
        self.entity_to_idx = self.entity_to_idx.compute(scheduler=self.scheduler_flag)
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        start_time = time.time()
        print('[6 / 14] Serializing compressed entity integer mapping...', end='\t')
        self.entity_to_idx.to_parquet(self.path_for_serialization + '/entity_to_idx.gzip', compression='gzip')
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')

        # 5. Create a bijection mapping  from relations to integer indexes.
        print('[7 / 14] Lazy Creating a mapping from relations to integer indexes...', end='\t')
        self.relation_to_idx = self.df_str_kg['relation'].to_frame().drop_duplicates()
        print('Computing relation indexes...', end='\t')
        start_time = time.time()
        self.relation_to_idx = self.relation_to_idx.set_index(self.relation_to_idx.relation)
        self.relation_to_idx['relation'] = dask.array.arange(0, self.relation_to_idx.size.compute())
        self.relation_to_idx = self.relation_to_idx.compute(scheduler=self.scheduler_flag)
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        print('[8 / 14] Serializing compressed relation integer mapping...', end='\t')
        start_time = time.time()
        self.relation_to_idx.to_parquet(self.path_for_serialization + '/relation_to_idx.gzip', compression='gzip')
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        start_time = time.time()
        print('Computing train dataset...', end='\t')
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        self.train_set = self.train_set.compute(scheduler=self.scheduler_flag)
        if self.valid_set is not None:
            start_time = time.time()
            print('Computing validation dataset...', end='\t')
            self.valid_set = self.valid_set.compute(scheduler=self.scheduler_flag)
            print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        if self.test_set is not None:
            start_time = time.time()
            print('Computing test dataset...', end='\t')
            self.test_set = self.test_set.compute(scheduler=self.scheduler_flag)
            print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        del self.df_str_kg

    def sequential_vocabulary_construction(self):
        print('Train set compute...')
        self.train_set = self.train_set.compute(scheduler=self.scheduler_flag)
        if self.valid_set is not None:
            print('Valid set compute...')
            self.valid_set = self.valid_set.compute(scheduler=self.scheduler_flag)
        if self.test_set is not None:
            print('Test set set compute...')
            self.test_set = self.test_set.compute(scheduler=self.scheduler_flag)
        # 1. Concatenate dataframes.
        print('\n[4 / 14] Concatenating data to obtain index...')
        x = [self.train_set]
        if self.valid_set is not None:
            x.append(self.valid_set)
        if self.test_set is not None:
            x.append(self.test_set)
        # self.df_str_kg = ddf.concat(x, ignore_index=True)
        self.df_str_kg = pd.concat(x, ignore_index=True)
        del x
        print('Done !\n')
        # (2) Remove triples from (1).
        self.remove_triples_with_condition()
        print('[5 / 14] Creating a mapping from entities to integer indexes...')
        # (3) Create a bijection mapping from entities of (2) to integer indexes.
        ordered_list = pd.unique(self.df_str_kg[['subject', 'object']].values.ravel('K'))
        self.entity_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)), columns=['entity'], index=ordered_list)
        print('Done !\n')
        print('[6 / 14] Serializing compressed entity integer mapping...')
        self.entity_to_idx.to_parquet(self.path_for_serialization + '/entity_to_idx.gzip', compression='gzip')
        print('Done !\n')
        # 5. Create a bijection mapping  from relations to integer indexes.
        print('[7 / 14] Creating a mapping from relations to integer indexes...')
        ordered_list = pd.unique(self.df_str_kg['relation'].values.ravel('K'))
        self.relation_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                            columns=['relation'],
                                            index=ordered_list)
        print('Done !\n')
        print('[8 / 14] Serializing compressed relation integer mapping...')
        self.relation_to_idx.to_parquet(self.path_for_serialization + '/relation_to_idx.gzip', compression='gzip')
        print('Done !\n')
        del ordered_list

    def remove_triples_with_condition(self):
        if self.min_freq_for_vocab is not None:
            assert isinstance(self.min_freq_for_vocab, int)
            assert self.min_freq_for_vocab > 0
            print(
                f'[5 / 14] Dropping triples having infrequent entities or relations (>{self.min_freq_for_vocab})...',
                end=' ')
            # num_triples = df_str_kg.size.compute(scheduler=scheduler_flag)
            num_triples = self.df_str_kg.size  # .compute(scheduler=scheduler_flag)
            print('Total num triples:', num_triples, end=' ')
            # Compute entity frequency: index is URI, val is number of occurrences.
            # entity_frequency = dask.dataframe.concat([df_str_kg['subject'], df_str_kg['object']]).value_counts()
            entity_frequency = pd.concat([self.df_str_kg['subject'], self.df_str_kg['object']]).value_counts()

            relation_frequency = self.df_str_kg['relation'].value_counts()
            # low_frequency_entities index and values are the same URIs: dask.dataframe.core.DataFrame
            low_frequency_entities = entity_frequency[
                entity_frequency <= min_freq_for_vocab].index.values  # .compute(scheduler=scheduler_flag)
            low_frequency_relation = relation_frequency[
                relation_frequency <= min_freq_for_vocab].index.values  # .compute(scheduler=scheduler_flag)
            # If triple contains subject that is in low_freq, set False do not select
            self.df_str_kg = self.df_str_kg[~self.df_str_kg['subject'].isin(low_frequency_entities)]
            # If triple contains object that is in low_freq, set False do not select
            self.df_str_kg = self.df_str_kg[~self.df_str_kg['object'].isin(low_frequency_entities)]
            # If triple contains relation that is in low_freq, set False do not select
            self.df_str_kg = self.df_str_kg[~self.df_str_kg['relation'].isin(low_frequency_relation)]
            # print('\t after dropping:', df_str_kg.size.compute(scheduler=scheduler_flag))
            print('\t after dropping:', df_str_kg.size)  # .compute(scheduler=scheduler_flag))
            del low_frequency_entities
            print('Done !\n')

    def load_read_process(self) -> Tuple[dask.dataframe.DataFrame, dask.dataframe.DataFrame, dask.dataframe.DataFrame]:
        """ Load train valid (if exists), and test (if exists) into memory """

        # 1. LOAD Data. (First pass on data)
        print(
            f'[1 / 14] Lazy Loading and Preprocessing training data: read_only_few: {self.read_only_few} , sample_triples_ratio: {self.sample_triples_ratio}...',
            end='\t')
        self.train_set = load_data_parallel(self.data_dir + '/train', self.read_only_few, self.sample_triples_ratio)
        print('Done !\n')
        print(
            f'[2 / 14] Lazy Loading and Preprocessing valid data...',
            end='\t')
        self.valid_set = load_data_parallel(self.data_dir + '/valid')
        print('Done !\n')
        print(
            f'[3 / 14] Lazy Loading and Preprocessing test data...',
            end='\t')
        self.test_set = load_data_parallel(self.data_dir + '/test')
        print('Done !\n')
        return self.train_set, self.valid_set, self.test_set

    def apply_reciprical_or_noise(self) -> None:
        """ (1) Add reciprocal triples (2) Add noisy triples """
        # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
        if self.add_reciprical and self.eval_model:
            print(
                '[3.1 / 14] Add reciprocal triples to train, validation, and test sets, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}',
                end='\t')
            self.train_set = create_recipriocal_triples_from_dask(self.train_set)
            if self.valid_set is not None:
                self.valid_set = create_recipriocal_triples_from_dask(self.valid_set)
            if self.test_set is not None:
                self.test_set = create_recipriocal_triples_from_dask(self.test_set)
            print('Done !\n')
        # (2) Extend KG with triples where entities and relations are randomly sampled.
        if self.add_noise_rate is not None:
            print(f'[4 / 14] Adding noisy triples...', end='\t')
            self.train_set = add_noisy_triples(self.train_set, self.add_noise_rate)
            print('Done!\n')

    def deserialize(self, storage_path: str) -> None:
        """ Deserialize data """
        print(f'Deserialization Path Path: {storage_path}\n')
        print('Deserializing compressed entity integer mapping...')
        self.entity_to_idx = pd.read_parquet(storage_path + '/entity_to_idx.gzip')  # .compute()
        print('Done!\n')
        self.num_entities = len(self.entity_to_idx)
        print('Deserializing compressed relation integer mapping...')
        self.relation_to_idx = pd.read_parquet(storage_path + '/relation_to_idx.gzip')  # .compute()
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
            self.valid_set = None  # pd.DataFrame()

        try:
            print('Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.test_set = ddf.read_parquet(storage_path + '/idx_test_df.gzip').values.compute()
            print('Done!\n')
        except FileNotFoundError:
            print('No test data found\n')
            self.test_set = None

    @property
    def entities_str(self) -> List:
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List:
        return list(self.relation_to_idx.keys())

    """
    @staticmethod
    def map_str_triples_to_numpy_idx(triples, entity_idx, relation_idx) -> np.array:
        return np.array([(entity_idx[s], relation_idx[p], entity_idx[o]) for s, p, o in triples])
    """
