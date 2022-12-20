import time
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import numpy as np
import pickle
import json
import os
import pandas
import pandas as pd
from .static_funcs import performance_debugger, get_er_vocab, get_ee_vocab, get_re_vocab, \
    create_recipriocal_triples, add_noisy_triples, index_triples, load_data, create_constraints, \
    numpy_data_type_changer, vocab_to_parquet, timeit
from .sanity_checkers import dataset_sanity_checking
import glob
import pyarrow.parquet as pq
import concurrent.futures
import polars

show_all = True


class KG:
    """ Knowledge Graph """

    def __init__(self, data_dir: str = None, deserialize_flag: str = None,
                 num_core: int = 1,
                 add_reciprical: bool = None, eval_model: str = None,
                 read_only_few: int = None, sample_triples_ratio: float = None,
                 path_for_serialization: str = None, add_noise_rate: float = None,
                 min_freq_for_vocab: int = None,
                 entity_to_idx=None, relation_to_idx=None, backend=None):
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
        self.data_dir = data_dir
        self.deserialize_flag = deserialize_flag
        self.num_core = num_core
        self.add_reciprical = add_reciprical
        self.eval_model = eval_model

        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.path_for_serialization = path_for_serialization
        self.add_noise_rate = add_noise_rate

        self.min_freq_for_vocab = min_freq_for_vocab
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.backend = 'pandas' if backend is None else backend
        self.train_set, self.valid_set, self.test_set = None, None, None

        # (1) Read or Load data from disk into memory.
        Read_Load_Data_From_Disk(kg=self).start()
        # (2) Preprocess (1).
        Preprocess(kg=self).start()

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


class Read_Load_Data_From_Disk:
    """Read or Load the data from disk into memory"""

    def __init__(self, kg: KG):
        self.kg = kg

    @timeit
    def __load(self) -> None:
        """ Deserialize data """
        print(f'Deserialization Path: {self.kg.deserialize_flag}\n')
        start_time = time.time()
        print('[1 / 4] Deserializing compressed entity integer mapping...')
        self.kg.entity_to_idx = pd.read_parquet(self.kg.deserialize_flag + '/entity_to_idx.gzip')
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        self.kg.num_entities = len(self.kg.entity_to_idx)

        print('[2 / ] Deserializing compressed relation integer mapping...')
        start_time = time.time()
        self.kg.relation_to_idx = pd.read_parquet(self.kg.deserialize_flag + '/relation_to_idx.gzip')
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')

        self.kg.num_relations = len(self.kg.relation_to_idx)
        print(
            '[3 / 4] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...',
        )
        start_time = time.time()
        self.kg.entity_to_idx = self.kg.entity_to_idx.to_dict()['entity']
        self.kg.relation_to_idx = self.kg.relation_to_idx.to_dict()['relation']
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        # 10. Serialize (9).
        print('[4 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
        start_time = time.time()
        self.kg.train_set = pd.read_parquet(self.kg.deserialize_flag + '/idx_train_df.gzip').values
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        try:
            print('[5 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.kg.valid_set = pd.read_parquet(self.kg.deserialize_flag + '/idx_valid_df.gzip').values
            print('Done!\n')
        except FileNotFoundError:
            print('No valid data found!\n')
            self.kg.valid_set = None  # pd.DataFrame()

        try:
            print('[6 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.kg.test_set = pd.read_parquet(self.kg.deserialize_flag + '/idx_test_df.gzip').values  # .compute()
            print('Done!\n')
        except FileNotFoundError:
            print('No test data found\n')
            self.kg.test_set = None

        if self.kg.eval_model:
            if self.kg.valid_set is not None and self.kg.test_set is not None:
                # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                data = np.concatenate([self.kg.train_set, self.kg.valid_set, self.kg.test_set])
            else:
                data = self.kg.train_set
            print('[7 / 4] Creating er,re, and ee type vocabulary for evaluation...')
            start_time = time.time()
            self.kg.er_vocab = get_er_vocab(data)
            self.kg.re_vocab = get_re_vocab(data)
            # 17. Create a bijection mapping from subject-object pairs to relations.
            self.kg.ee_vocab = get_ee_vocab(data)
            self.kg.domain_constraints_per_rel, self.range_constraints_per_rel = create_constraints(self.kg.train_set)
            print(f'Done !\t{time.time() - start_time:.3f} seconds\n')

    @timeit
    def __read(self) -> None:
        """ Read the data into memory """
        for i in glob.glob(self.kg.data_dir + '/*'):
            if 'train' in i:
                # 1. LOAD Data. (First pass on data)
                print(f'Loading Training Data...')
                self.kg.train_set = load_data(i, self.kg.read_only_few, self.kg.sample_triples_ratio,
                                              backend=self.kg.backend)
            elif 'test' in i:
                print(f'Loading Test Data...')
                self.kg.test_set = load_data(i, backend=self.kg.backend)
            elif 'valid' in i:
                print(f'Loading Validation Data...')
                self.kg.valid_set = load_data(i, backend=self.kg.backend)
            else:
                print(f'Unrecognized data {i}')

    @timeit
    def start(self) -> None:
        """Read or Load the train, valid, and test datasets"""
        # (1) Read or load the data into memory, otherwise load it
        if self.kg.deserialize_flag:
            self.__load()
        else:
            self.__read()


class Preprocess:
    """ Preprocess the data in memory """

    def __init__(self, kg: KG):
        self.kg = kg

    @timeit
    def start(self) -> None:
        """Preprocess train, valid, and test datasets"""
        # (1) PrRead or load the data into memory, otherwise load it
        if self.kg.deserialize_flag:
            pass
        else:
            self.__preprocess()

    @timeit
    def __preprocess_pandas(self) -> None:
        """ Preprocess data stored in pandas/modin DataFrame"""
        assert self.kg.backend in ['pandas', 'modin']
        # (1.2) Update (1.1).
        self.apply_reciprical_or_noise()
        # (1.3) Construct integer indexing for entities and relations.
        if self.kg.entity_to_idx is None and self.kg.relation_to_idx is None:
            self.sequential_vocabulary_construction()
            print('[9 / 14] Obtaining entity to integer index mapping from pandas dataframe...')
            # CD: <> brackets are not removed anymore <http://embedding.cc/resource/Karim_Fegrouch>
            self.kg.entity_to_idx = self.kg.entity_to_idx.to_dict()['entity']
            print('Done !\n')
            print('[9 / 14] Obtaining relation to integer index mapping from pandas dataframe...')
            self.kg.relation_to_idx = self.kg.relation_to_idx.to_dict()['relation']
            print('Done !\n')
            self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(self.kg.relation_to_idx)
            print('[10 / 14] Mapping training data into integers for training...')
            start_time = time.time()
            # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
            self.kg.train_set = index_triples(self.kg.train_set,
                                              self.kg.entity_to_idx,
                                              self.kg.relation_to_idx)
            print(f'Done ! {time.time() - start_time:.3f} seconds\n')
            if self.kg.path_for_serialization is not None:
                # 10. Serialize (9).
                print('[11 / 14] Serializing integer mapped data...')
                self.kg.train_set.to_parquet(self.kg.path_for_serialization + '/idx_train_df.gzip', compression='gzip',
                                             engine='pyarrow')
                print('Done !\n')
            assert isinstance(self.kg.train_set, pd.core.frame.DataFrame)
            # 11. Convert data from pandas dataframe to numpy ndarray.
            print('[12 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...')
            # CD: Maybe to list?
            self.kg.train_set = self.kg.train_set.values
            print('Done !\n')
        else:
            print(
                '[4 / 14] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...',
            )
            self.kg.entity_to_idx = self.kg.entity_to_idx.to_dict()['entity']
            self.kg.relation_to_idx = self.kg.relation_to_idx.to_dict()['relation']
            self.kg.num_entities = len(self.kg.entity_to_idx)
            self.kg.num_relations = len(self.kg.relation_to_idx)
            print('Done !\n')
            print('[10 / 14] Mapping training data into integers for training...')
            # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
            self.kg.train_set = index_triples(self.kg.train_set, self.kg.entity_to_idx, self.kg.relation_to_idx)
            print('Done !\n')
            self.kg.train_set = self.kg.train_set
            assert isinstance(self.kg.train_set, pd.core.frame.DataFrame)
            # 11. Convert data from pandas dataframe to numpy ndarray.
            print('[12 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...')
            self.kg.train_set = self.kg.train_set.values
            print('Done !\n')

        self.kg.train_set = numpy_data_type_changer(self.kg.train_set,
                                                    num=max(self.kg.num_entities, self.kg.num_relations))
        print('[13 / 14 ] Sanity checking...')
        # 12. Sanity checking: indexed training set can not have an indexed entity assigned with larger indexed than the number of entities.
        dataset_sanity_checking(self.kg.train_set, self.kg.num_entities, self.kg.num_relations)
        print('Done !\n')
        if self.kg.valid_set is not None:
            if self.kg.path_for_serialization is not None:
                print('[14 / 14 ] Serializing validation data for Continual Learning...')
                self.kg.valid_set.to_parquet(
                    self.kg.path_for_serialization + '/valid_df.gzip', compression='gzip', engine='pyarrow')
                print('Done !\n')
            print('[14 / 14 ] Indexing validation dataset...')
            self.kg.valid_set = index_triples(self.kg.valid_set, self.kg.entity_to_idx, self.kg.relation_to_idx)
            print('Done !\n')
            if self.kg.path_for_serialization is not None:
                print('[15 / 14 ] Serializing indexed validation dataset...')
                self.kg.valid_set.to_parquet(
                    self.kg.path_for_serialization + '/idx_valid_df.gzip', compression='gzip', engine='pyarrow')
                print('Done !\n')
            # To numpy
            self.kg.valid_set = self.kg.valid_set.values
            dataset_sanity_checking(self.kg.valid_set, self.kg.num_entities, self.kg.num_relations)
            self.kg.valid_set = numpy_data_type_changer(self.kg.valid_set,
                                                        num=max(self.kg.num_entities, self.kg.num_relations))
        if self.kg.test_set is not None:
            if self.kg.path_for_serialization is not None:
                print('[16 / 14 ] Serializing test data for Continual Learning...')
                self.kg.test_set.to_parquet(
                    self.kg.path_for_serialization + '/test_df.gzip', compression='gzip', engine='pyarrow')
                print('Done !\n')
            print('[17 / 14 ] Indexing test dataset...')
            self.kg.test_set = index_triples(self.kg.test_set, self.kg.entity_to_idx, self.kg.relation_to_idx)
            print('Done !\n')
            if self.kg.path_for_serialization is not None:
                print('[18 / 14 ] Serializing indexed test dataset...')
                self.kg.test_set.to_parquet(
                    self.kg.path_for_serialization + '/idx_test_df.gzip', compression='gzip', engine='pyarrow')
            # To numpy
            self.kg.test_set = self.kg.test_set.values
            dataset_sanity_checking(self.kg.test_set, self.kg.num_entities, self.kg.num_relations)
            self.kg.test_set = numpy_data_type_changer(self.kg.test_set,
                                                       num=max(self.kg.num_entities, self.kg.num_relations))
            print('Done !\n')

    @timeit
    def __preprocess_polars(self):
        print('Preprocessing with Polars...')
        # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
        if self.kg.add_reciprical and self.kg.eval_model:
            @timeit
            def adding_reciprocal_triples():
                # (1.1) Add reciprocal triples into training set
                self.kg.train_set.extend(self.kg.train_set.select([
                    polars.col("object").alias('subject'),
                    polars.col("relation").apply(lambda x: x + '_inverse'),
                    polars.col("subject").alias('object')
                ]))
                if self.kg.valid_set is not None:
                    # (1.2) Add reciprocal triples into valid_set set.
                    self.kg.valid_set.extend(self.kg.valid_set.select([
                        polars.col("object").alias('subject'),
                        polars.col("relation").apply(lambda x: x + '_inverse'),
                        polars.col("subject").alias('object')
                    ]))
                if self.kg.test_set is not None:
                    # (1.2) Add reciprocal triples into test set.
                    self.kg.test_set.extend(self.kg.test_set.select([
                        polars.col("object").alias('subject'),
                        polars.col("relation").apply(lambda x: x + '_inverse'),
                        polars.col("subject").alias('object')
                    ]))

            adding_reciprocal_triples()

        # (2) Type checking
        try:
            assert isinstance(self.kg.train_set, polars.DataFrame)
        except TypeError:
            raise TypeError(f"{type(kg.train_set)}")
        assert isinstance(self.kg.valid_set, polars.DataFrame) or self.kg.valid_set is None
        assert isinstance(self.kg.test_set, polars.DataFrame) or self.kg.test_set is None
        if self.kg.min_freq_for_vocab is not None:
            raise NotImplementedError('With using Polars')

        @timeit
        def concat_splits(train, val, test):
            x = [train]
            if val is not None:
                x.append(val)
            if test is not None:
                x.append(test)
            return polars.concat(x)

        df_str_kg = concat_splits(self.kg.train_set, self.kg.valid_set, self.kg.test_set)

        @timeit
        def entity_index():
            # Entity Index: {'a':1, 'b':2} :
            self.kg.entity_to_idx = polars.concat((df_str_kg['subject'], df_str_kg['object'])).unique(
                maintain_order=True).rename('entity')
            self.kg.entity_to_idx.to_frame().to_pandas().to_parquet(
                self.kg.path_for_serialization + f'/entity_to_idx.gzip', compression='gzip', engine='pyarrow')
            self.kg.entity_to_idx = dict(zip(self.kg.entity_to_idx.to_list(), list(range(len(self.kg.entity_to_idx)))))

        entity_index()

        @timeit
        def relation_index():
            # Relation Index: {'r1':1, 'r2:'2}
            self.kg.relation_to_idx = df_str_kg['relation'].unique(maintain_order=True)
            self.kg.relation_to_idx.to_frame().to_pandas().to_parquet(
                self.kg.path_for_serialization + f'/relation_to_idx.gzip', compression='gzip', engine='pyarrow')
            self.kg.relation_to_idx = dict(
                zip(self.kg.relation_to_idx.to_list(), list(range(len(self.kg.relation_to_idx)))))

        relation_index()
        self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(self.kg.relation_to_idx)

        @timeit
        def indexer(data):
            return data.with_columns(
                [polars.col("subject").apply(lambda x: self.kg.entity_to_idx[x]).alias("subject"),
                 polars.col("relation").apply(lambda x: self.kg.relation_to_idx[x]).alias("relation"),
                 polars.col("object").apply(lambda x: self.kg.entity_to_idx[x]).alias("object")]
            )

        @timeit
        def index_train():
            # From str to int
            self.kg.train_set = indexer(self.kg.train_set).to_pandas()
            if self.kg.path_for_serialization is not None:
                self.kg.train_set.to_parquet(self.kg.path_for_serialization + '/idx_train_df.gzip', compression='gzip',
                                             engine='pyarrow')
            self.kg.train_set = self.kg.train_set.values

        index_train()

        @timeit
        def index_val():
            self.kg.valid_set = indexer(self.kg.valid_set).to_pandas()
            self.kg.valid_set.to_parquet(self.kg.path_for_serialization + '/idx_valid_df.gzip', compression='gzip',
                                         engine='pyarrow')
            self.kg.valid_set = self.kg.valid_set.values

        if self.kg.valid_set is not None:
            index_val()

        @timeit
        def index_test():
            self.kg.test_set = indexer(self.kg.test_set).to_pandas()
            self.kg.test_set.to_parquet(self.kg.path_for_serialization + '/idx_test_df.gzip', compression='gzip',
                                        engine='pyarrow')
            self.kg.test_set = self.kg.test_set.values

        if self.kg.test_set is not None:
            index_test()

    @timeit
    def __preprocess(self) -> None:
        """ Preprocess the read data """
        print('Preprocessing...')
        if self.kg.backend == 'polars':
            self.__preprocess_polars()
        elif self.kg.backend in ['pandas', 'modin']:
            self.__preprocess_pandas()
        else:
            raise KeyError(f'{self.kg.backend} not found')

        if self.kg.eval_model:  # and len(self.valid_set) > 0 and len(self.test_set) > 0:
            if self.kg.valid_set is not None and self.kg.test_set is not None:
                assert isinstance(self.kg.valid_set, np.ndarray) and isinstance(self.kg.test_set, np.ndarray)
                # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                data = np.concatenate([self.kg.train_set, self.kg.valid_set, self.kg.test_set])
            else:
                data = self.kg.train_set
            # We need to parallelise the next four steps.
            print('Creating Vocab...')
            executor = concurrent.futures.ProcessPoolExecutor()
            self.kg.er_vocab = executor.submit(get_er_vocab, data)  # get_er_vocab(data)
            self.kg.re_vocab = executor.submit(get_re_vocab, data)  # get_re_vocab(data)
            self.kg.ee_vocab = executor.submit(get_ee_vocab, data)  # get_ee_vocab(data)
            self.kg.constraints = executor.submit(create_constraints, self.kg.train_set)
            self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = None, None  # create_constraints(self.train_set)

    def sequential_vocabulary_construction(self) -> None:
        """
        (1) Read input data into memory
        (2) Remove triples with a condition
        (3) Serialize vocabularies in a pandas dataframe where
                    => the index is integer and
                    => a single column is string (e.g. URI)
        """
        try:
            assert isinstance(self.kg.train_set, pd.DataFrame)
        except AssertionError:
            print(type(self.kg.train_set))
            print('HEREE')
            exit(1)
        assert isinstance(self.kg.valid_set, pd.DataFrame) or self.kg.valid_set is None
        assert isinstance(self.kg.test_set, pd.DataFrame) or self.kg.test_set is None

        # (4) Remove triples from (1).
        self.remove_triples_from_train_with_condition()
        # Concatenate dataframes.
        print('\nConcatenating data to obtain index...')
        x = [self.kg.train_set]
        if self.kg.valid_set is not None:
            x.append(self.kg.valid_set)
        if self.kg.test_set is not None:
            x.append(self.kg.test_set)
        df_str_kg = pd.concat(x, ignore_index=True)
        del x
        print('Done !\n')

        print('Creating a mapping from entities to integer indexes...')
        # (5) Create a bijection mapping from entities of (2) to integer indexes.
        # ravel('K') => Return a contiguous flattened array.
        # ‘K’ means to read the elements in the order they occur in memory, except for reversing the data when strides are negative.
        ordered_list = pd.unique(df_str_kg[['subject', 'object']].values.ravel('K'))
        self.kg.entity_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)), columns=['entity'], index=ordered_list)
        print('Done !\n')
        vocab_to_parquet(self.kg.entity_to_idx, 'entity_to_idx.gzip', self.kg.path_for_serialization,
                         print_into='Serializing compressed entity integer mapping...')
        # 5. Create a bijection mapping  from relations to integer indexes.
        print('Creating a mapping from relations to integer indexes...')
        ordered_list = pd.unique(df_str_kg['relation'].values.ravel('K'))
        self.kg.relation_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                               columns=['relation'],
                                               index=ordered_list)
        print('Done !\n')

        vocab_to_parquet(self.kg.relation_to_idx, 'relation_to_idx.gzip', self.kg.path_for_serialization,
                         'Serializing compressed relation integer mapping...')
        del ordered_list

    def remove_triples_from_train_with_condition(self):
        if self.kg.min_freq_for_vocab is not None:
            assert isinstance(self.kg.min_freq_for_vocab, int)
            assert self.kg.min_freq_for_vocab > 0
            print(
                f'[5 / 14] Dropping triples having infrequent entities or relations (>{self.kg.min_freq_for_vocab})...',
                end=' ')
            num_triples = self.kg.train_set.size
            print('Total num triples:', num_triples, end=' ')
            # Compute entity frequency: index is URI, val is number of occurrences.
            entity_frequency = pd.concat([self.kg.train_set['subject'], self.kg.train_set['object']]).value_counts()
            relation_frequency = self.kg.train_set['relation'].value_counts()

            # low_frequency_entities index and values are the same URIs: dask.dataframe.core.DataFrame
            low_frequency_entities = entity_frequency[
                entity_frequency <= self.kg.min_freq_for_vocab].index.values
            low_frequency_relation = relation_frequency[
                relation_frequency <= self.kg.min_freq_for_vocab].index.values
            # If triple contains subject that is in low_freq, set False do not select
            self.kg.train_set = self.kg.train_set[~self.kg.train_set['subject'].isin(low_frequency_entities)]
            # If triple contains object that is in low_freq, set False do not select
            self.kg.train_set = self.kg.train_set[~self.kg.train_set['object'].isin(low_frequency_entities)]
            # If triple contains relation that is in low_freq, set False do not select
            self.kg.train_set = self.kg.train_set[~self.kg.train_set['relation'].isin(low_frequency_relation)]
            # print('\t after dropping:', df_str_kg.size.compute(scheduler=scheduler_flag))
            print('\t after dropping:', self.kg.train_set.size)  # .compute(scheduler=scheduler_flag))
            del low_frequency_entities
            print('Done !\n')

    def apply_reciprical_or_noise(self) -> None:
        """ (1) Add reciprocal triples (2) Add noisy triples """
        # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
        if self.kg.add_reciprical and self.kg.eval_model:
            print(
                '[3.1 / 14] Add reciprocal triples to train, validation, and test sets, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}',
            )
            self.kg.train_set = create_recipriocal_triples(self.kg.train_set)
            if self.kg.valid_set is not None:
                self.kg.valid_set = create_recipriocal_triples(self.kg.valid_set)
            if self.kg.test_set is not None:
                self.kg.test_set = create_recipriocal_triples(self.kg.test_set)
            print('Done !\n')

        # (2) Extend KG with triples where entities and relations are randomly sampled.
        if self.kg.add_noise_rate is not None:
            print(f'[4 / 14] Adding noisy triples...')
            self.kg.train_set = add_noisy_triples(self.kg.train_set, self.kg.add_noise_rate)
            print('Done!\n')
