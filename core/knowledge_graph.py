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
                 path_for_serialization: str = None, add_noise_rate: float = None):
        """

        :param data_dir: A path of a folder containing the input knowledge graph
        :param deserialize_flag: A path of a folder containing previously parsed data
        :param large_kg_parse: A flag for using all cores to parse input knowledge graph
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied. If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        """
        if deserialize_flag is None:
            # 1. LOAD Data. (First pass on data)
            print(
                f'[1 / 14] Loading training data: large_kg_parse: {large_kg_parse}, read_only_few: {read_only_few} , sample_triples_ratio: {sample_triples_ratio}...')
            self.train_set = self.load_data_parallel(data_dir + '/train.txt', large_kg_parse, read_only_few,
                                                     sample_triples_ratio)
            print('Done !\n')
            print(
                f'[2 / 14] Loading valid data: large_kg_parse: {large_kg_parse}, read_only_few: {read_only_few}, sample_triples_ratio: {sample_triples_ratio}...')

            self.valid_set = self.load_data_parallel(data_dir + '/valid.txt', large_kg_parse, read_only_few,
                                                     sample_triples_ratio)
            print('Done !\n')
            print(
                f'[3 / 14] Loading test data: large_kg_parse: {large_kg_parse}, read_only_few: {read_only_few}, sample_triples_ratio: {sample_triples_ratio}...')

            self.test_set = self.load_data_parallel(data_dir + '/test.txt', large_kg_parse, read_only_few,
                                                    sample_triples_ratio)
            print('Done !\n')

            # 2. Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
            if add_reciprical:
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
            print('Done!\n')

            # 5. Create a bijection mapping  from relations to integer indexes.
            print('[6 / 14] Creating a mapping from relations to integer indexes...')
            ordered_list = pd.unique(df_str_kg['relation'].values.ravel('K'))
            self.relation_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                                columns=['relation'],
                                                index=ordered_list)
            print('Done!\n')
            # Free memory
            del ordered_list, df_str_kg

            ## 6. Serialize indexed entities and relations into disk for further usage.
            print('[7 / 14]Serializing compressed entity integer mapping...')
            self.entity_to_idx.to_parquet(path_for_serialization + '/entity_to_idx.gzip', compression='gzip')
            print('Done!\n')

            print('[8 / 14]Serializing compressed relation integer mapping...')
            self.relation_to_idx.to_parquet(path_for_serialization + '/relation_to_idx.gzip', compression='gzip')
            print('Done!\n')

            # 7. Convert from pandas dataframe to dictionaries for an easy access
            # We may want to benchmark using python dictionary and pandas frame
            print(
                '[9 / 14]Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...')
            self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
            self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
            self.num_entities = len(self.entity_to_idx)
            self.num_relations = len(self.relation_to_idx)
            print('Done!\n')

            # 8. Serialize already read training data in parquet format so that
            # the training data is stored in more memory efficient manner as well as
            # it can be reread later faster
            print('[10 / 14] Serializing training data...')  # TODO: Do we really need it ?!
            self.train_set.to_parquet(path_for_serialization + '/train_df.gzip', compression='gzip')
            print('Done!\n')

            print('[11 / 14] Mapping training data into integers for training...')
            # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
            self.train_set['subject'] = self.train_set['subject'].map(lambda x: self.entity_to_idx[x])
            self.train_set['relation'] = self.train_set['relation'].map(lambda x: self.relation_to_idx[x])
            self.train_set['object'] = self.train_set['object'].map(lambda x: self.entity_to_idx[x])
            print('Done!\n')

            # 10. Serialize (9).
            print('[12 / 14] Serializing integer mapped data...')  # TODO: Do we really need it ?!
            self.train_set.to_parquet(path_for_serialization + '/idx_train_df.gzip', compression='gzip')
            print('Done!\n')

            # 11. Convert data from pandas dataframe to numpy ndarray.
            print('[13 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...')
            self.train_set = self.train_set.values
            print('Done!\n')

            print('[14 / 14 ] Sanity checking on training dataset...')
            # 12. Sanity checking: indexed training set can not have an indexed entity assigned with larger indexed than the number of entities.
            assert self.num_entities > max(self.train_set[:, 0]) and self.num_entities > max(self.train_set[:, 2])
            assert self.num_relations > max(self.train_set[:, 1])
            # 13. Sanity checking: data types
            assert isinstance(self.train_set[0], np.ndarray)
            assert isinstance(self.train_set[0][0], np.int64) and isinstance(self.train_set[0][1], np.int64)
            assert isinstance(self.train_set[0][2], np.int64)
            # 14. Repeat computations carried out from 8-13 on validation dataset.
            if len(self.valid_set) > 0:
                print('Serializing validation data...')  # TODO: Do we really need it ?!
                self.valid_set.to_parquet(path_for_serialization + '/valid_df.gzip', compression='gzip')
                self.valid_set['subject'] = self.valid_set['subject'].map(lambda x: self.entity_to_idx[x])
                self.valid_set['relation'] = self.valid_set['relation'].map(lambda x: self.relation_to_idx[x])
                self.valid_set['object'] = self.valid_set['object'].map(lambda x: self.entity_to_idx[x])
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
                print('Serializing test data...')  # TODO: Do we really need it ?!
                self.test_set.to_parquet(path_for_serialization + '/test_df.gzip', compression='gzip')
                self.test_set['subject'] = self.test_set['subject'].map(lambda x: self.entity_to_idx[x])
                self.test_set['relation'] = self.test_set['relation'].map(lambda x: self.relation_to_idx[x])
                self.test_set['object'] = self.test_set['object'].map(lambda x: self.entity_to_idx[x])
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

            # 4. Display info
            self.description_of_input = f'\n------------------- Description of Dataset {data_dir} -------------------'
            self.description_of_input += f'\nNumber of entities: {self.num_entities}' \
                                         f'\nNumber of relations: {self.num_relations}' \
                                         f'\nNumber of triples on train set: {len(self.train_set)}' \
                                         f'\nNumber of triples on valid set: {len(self.valid_set)}' \
                                         f'\nNumber of triples on test set: {len(self.test_set)}\n'
        else:
            self.deserialize(deserialize_flag)

    @staticmethod
    def index(data: List[List], add_reciprical=False) -> (Dict, Dict, Dict, Dict, Dict):
        """
        Index each triples into their integer representation. Performed in a single tread with single core.
        V

        :param data:
        :param add_reciprical:
        :return:
        """
        print(f'Indexing {len(data)} triples. Data augmentation flag => {add_reciprical}')
        # Entity to integer indexing
        entity_idxs = {}
        # Relation to integer indexing
        relation_idxs = {}

        # Mapping from (head entity & relation) to tail entity
        er_vocab = defaultdict(list)
        # Mapping from (relation & tail entity) to head entity
        pe_vocab = defaultdict(list)
        # Mapping from (head entity & tail entity) to relation
        ee_vocab = defaultdict(list)

        for triple in data:
            try:
                h, r, t = triple[0], triple[1], triple[2]
            except IndexError:
                print(f'{triple} is not parsed corrected.')
                continue

            # 1. Integer indexing entities and relations
            entity_idxs.setdefault(h, len(entity_idxs))
            entity_idxs.setdefault(t, len(entity_idxs))
            relation_idxs.setdefault(r, len(relation_idxs))

            # 2. Mappings for filtered evaluation
            # 2.1. (HEAD,RELATION) => TAIL
            er_vocab[(entity_idxs[h], relation_idxs[r])].append(entity_idxs[t])
            # 2.2. (RELATION,TAIL) => HEAD
            pe_vocab[(relation_idxs[r], entity_idxs[t])].append(entity_idxs[h])
            # 2.3. (HEAD,TAIL) => RELATION
            ee_vocab[(entity_idxs[h], entity_idxs[t])].append(relation_idxs[r])

            if add_reciprical:
                # 1. Create reciprocal triples (t r_reverse h)
                r_reverse = r + "_reverse"
                relation_idxs.setdefault(r_reverse, len(relation_idxs))

                er_vocab[(entity_idxs[t], relation_idxs[r_reverse])].append(entity_idxs[h])
                pe_vocab[(relation_idxs[r_reverse], entity_idxs[h])].append(entity_idxs[t])
                ee_vocab[(entity_idxs[t], entity_idxs[h])].append(relation_idxs[r_reverse])

        return entity_idxs, relation_idxs, er_vocab, pe_vocab, ee_vocab

    def deserialize(self, storage_path: str) -> None:
        """ Deserialize data """

        print('Deserializing compressed entity integer mapping...')
        self.entity_to_idx = ddf.read_parquet(storage_path + '/entity_to_idx.gzip').compute()
        print('Done!\n')
        self.num_entities = len(self.entity_to_idx)
        print('Deserializing compressed relation integer mapping...')
        self.relation_to_idx = ddf.read_parquet(storage_path + '/relation_to_idx.gzip').compute()
        self.num_relations = len(self.entity_to_idx)

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
            print('No valid data found')
            self.valid_set = pd.DataFrame()

        try:
            print('Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.test_set = ddf.read_parquet(storage_path + '/idx_test_df.gzip').values.compute()
            print('Done!\n')
        except FileNotFoundError:
            print('No test data found')
            self.test_set = pd.DataFrame()

        print(storage_path)
        with open(storage_path + '/configuration.json', 'r') as f:
            args = json.load(f)

        if args['eval']:
            if len(self.valid_set) > 0 and len(self.test_set) > 0:
                # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                data = np.concatenate([self.train_set, self.valid_set, self.test_set])
            else:
                data = self.train_set
            self.er_vocab = get_er_vocab(data)

    @staticmethod
    def index_parallel(data: List[List], add_reciprical=False) -> (Dict, Dict, Dict, Dict, Dict):
        """
        Index each triples into their integer representation. Performed in a single tread with single core.
        V

        :param data:
        :param add_reciprical:
        :return:
        """
        print(f'Indexing {len(data)} triples. Data augmentation flag => {add_reciprical}')
        # Entity to integer indexing
        entity_idxs = {}
        # Relation to integer indexing
        relation_idxs = {}

        # Mapping from (head entity & relation) to tail entity
        er_vocab = defaultdict(list)
        # Mapping from (relation & tail entity) to head entity
        pe_vocab = defaultdict(list)
        # Mapping from (head entity & tail entity) to relation
        ee_vocab = defaultdict(list)

        for triple in data:
            try:
                h, r, t = triple[0], triple[1], triple[2]
            except IndexError:
                print(f'{triple} is not parsed corrected.')
                continue

            # 1. Integer indexing entities and relations
            entity_idxs.setdefault(h, len(entity_idxs))
            entity_idxs.setdefault(t, len(entity_idxs))
            relation_idxs.setdefault(r, len(relation_idxs))

            # 2. Mappings for filtered evaluation
            # 2.1. (HEAD,RELATION) => TAIL
            er_vocab[(entity_idxs[h], relation_idxs[r])].append(entity_idxs[t])
            # 2.2. (RELATION,TAIL) => HEAD
            pe_vocab[(relation_idxs[r], entity_idxs[t])].append(entity_idxs[h])
            # 2.3. (HEAD,TAIL) => RELATION
            ee_vocab[(entity_idxs[h], entity_idxs[t])].append(relation_idxs[r])

            if add_reciprical:
                # 1. Create reciprocal triples (t r_reverse h)
                r_reverse = r + "_reverse"
                relation_idxs.setdefault(r_reverse, len(relation_idxs))

                er_vocab[(entity_idxs[t], relation_idxs[r_reverse])].append(entity_idxs[h])
                pe_vocab[(relation_idxs[r_reverse], entity_idxs[h])].append(entity_idxs[t])
                ee_vocab[(entity_idxs[t], entity_idxs[h])].append(relation_idxs[r_reverse])

        return entity_idxs, relation_idxs, er_vocab, pe_vocab, ee_vocab

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
        if os.path.exists(data_path):
            with open(data_path, 'r') as reader:
                s = next(reader)
                # Heuristic to infer the format of the input data
                # ntriples checking: Last two characters must be whitespace + . + \n
                if s[-3:] == ' .\n':
                    is_nt_format = True
                else:
                    is_nt_format = False

            # Whitespace is used as deliminator and first three items are considered.
            df = ddf.read_csv(data_path, delim_whitespace=True, header=None, usecols=[0, 1, 2],
                              names=['subject', 'relation', 'object'])

            if isinstance(read_only_few, int):
                if read_only_few > 0:
                    df = df.loc[:read_only_few]
            if sample_triples_ratio:
                df = df.sample(frac=sample_triples_ratio)

            if is_nt_format:
                # Drop rows having ^^
                df = df[df["object"].str.contains("<http://www.w3.org/2001/XMLSchema#double>") == False]
                df = df[df["object"].str.contains("<http://www.w3.org/2001/XMLSchema#boolean>") == False]
                df['subject'] = df['subject'].str.removeprefix("<").str.removesuffix(">")
                df['relation'] = df['relation'].str.removeprefix("<").str.removesuffix(">")
                df['object'] = df['object'].str.removeprefix("<").str.removesuffix(">")
            if large_kg_parse:
                df = df.compute(scheduler='processes')
            else:
                df = df.compute(scheduler='single-threaded')
            x, y = df.shape
            assert y == 3
            # print(f'Parsed via DASK: {df.shape}. Whitespace is used as delimiter.')
            return df
        else:
            print(f'{data_path} could not found!\n')
            return pd.DataFrame()

    @staticmethod
    def get_entities_and_relations(data):
        entities = set()
        relations = set()

        for triple in data:
            h, r, t = triple[0], triple[1], triple[2]
            entities.add(h)
            entities.add(t)
            relations.add(r)
        return sorted(list(entities)), sorted(list(relations))

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
    def ntriple_parser(l: List) -> List:
        """
        Given a list of strings (e.g. [<...>,<...>,<...>,''])
        :param l:
        :return:
        """

        """
        l=[<...>,<...>,<...>]
        :param l:
        :return:
        """
        raise NotImplementedError()
        assert l[3] == '.'
        try:
            s, p, o, _ = l[0], l[1], l[2], l[3]
            # ...=<...>
            assert p[0] == '<' and p[-1] == '>'
            p = p[1:-1]
            if s[0] == '<':
                assert s[-1] == '>'
                s = s[1:-1]
            if o[0] == '<':
                assert o[-1] == '>'
                o = o[1:-1]
        except AssertionError:
            print('Parsing error')
            print(l)
            exit(1)
        return [s, p, o]

    @performance_debugger('Pickle Dump of')
    def __pickle_dump_obj(self, obj, path, info) -> None:
        print(info, end='')
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @performance_debugger('JSON Dump of')
    def __json_dump_obj(self, obj, path, info) -> None:
        print(info, end='')
        with open(path, 'w') as handle:
            json.dump(obj, handle)

    @performance_debugger('Numpy Save')
    def __np_save_as_compressed(self, train, valid, test, path, info) -> None:
        print(info, end='')
        np.savez_compressed(path, train=train, valid=valid, test=test)

    @staticmethod
    def index(data: List[List], add_reciprical=False) -> (Dict, Dict, Dict, Dict, Dict):
        """
        Index each triples into their integer representation. Performed in a single tread with single core.
        V

        :param data:
        :param add_reciprical:
        :return:
        """
        print(f'Indexing {len(data)} triples. Data augmentation flag => {add_reciprical}')
        # Entity to integer indexing
        entity_idxs = {}
        # Relation to integer indexing
        relation_idxs = {}

        # Mapping from (head entity & relation) to tail entity
        er_vocab = defaultdict(list)
        # Mapping from (relation & tail entity) to head entity
        pe_vocab = defaultdict(list)
        # Mapping from (head entity & tail entity) to relation
        ee_vocab = defaultdict(list)

        for triple in data:
            try:
                h, r, t = triple[0], triple[1], triple[2]
            except IndexError:
                print(f'{triple} is not parsed corrected.')
                continue

            # 1. Integer indexing entities and relations
            entity_idxs.setdefault(h, len(entity_idxs))
            entity_idxs.setdefault(t, len(entity_idxs))
            relation_idxs.setdefault(r, len(relation_idxs))

            # 2. Mappings for filtered evaluation
            # 2.1. (HEAD,RELATION) => TAIL
            er_vocab[(entity_idxs[h], relation_idxs[r])].append(entity_idxs[t])
            # 2.2. (RELATION,TAIL) => HEAD
            pe_vocab[(relation_idxs[r], entity_idxs[t])].append(entity_idxs[h])
            # 2.3. (HEAD,TAIL) => RELATION
            ee_vocab[(entity_idxs[h], entity_idxs[t])].append(relation_idxs[r])

            if add_reciprical:
                # 1. Create reciprocal triples (t r_reverse h)
                r_reverse = r + "_reverse"
                relation_idxs.setdefault(r_reverse, len(relation_idxs))

                er_vocab[(entity_idxs[t], relation_idxs[r_reverse])].append(entity_idxs[h])
                pe_vocab[(relation_idxs[r_reverse], entity_idxs[h])].append(entity_idxs[t])
                ee_vocab[(entity_idxs[t], entity_idxs[h])].append(relation_idxs[r_reverse])

        return entity_idxs, relation_idxs, er_vocab, pe_vocab, ee_vocab

    @staticmethod
    def map_str_triples_to_numpy_idx(triples, entity_idx, relation_idx) -> np.array:
        return np.array([(entity_idx[s], relation_idx[p], entity_idx[o]) for s, p, o in triples])

    def triple_indexing(self, large_kg_parse) -> None:
        """
        :return:
        """
        # This part takes the most of the time.
        print('Triple indexing')
        if large_kg_parse:
            print('No Parallelism implemented yet')
            # If LARGE WE ASSUME THAT there is no val and test
            self.train = self.map_str_triples_to_numpy_idx(triples=self.train, entity_idx=self.entity_idx,
                                                           relation_idx=self.relation_idx)
            self.valid = np.array([])
            self.test = np.array([])
        else:
            self.train = self.map_str_triples_to_numpy_idx(triples=self.train, entity_idx=self.entity_idx,
                                                           relation_idx=self.relation_idx)
            if self.is_valid_test_available():
                self.valid = self.map_str_triples_to_numpy_idx(triples=self.valid, entity_idx=self.entity_idx,
                                                               relation_idx=self.relation_idx)

                self.test = self.map_str_triples_to_numpy_idx(triples=self.test, entity_idx=self.entity_idx,
                                                              relation_idx=self.relation_idx)
            else:
                self.valid = np.array([])
                self.test = np.array([])
