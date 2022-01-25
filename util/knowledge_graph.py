import time
from typing import Dict, List, Generator
from collections import defaultdict
import numpy as np
import multiprocessing
from itertools import zip_longest
import pickle
import json
from dask import dataframe as ddf
import os
import pandas as pd
from static_funcs import performance_debugger


class KG:
    def __init__(self, data_dir: str = None, deserialize_flag: str = None, large_kg_parse=False, add_reciprical=False,
                 eval=True, read_only_few: int = None):
        """

        :param data_dir: A path of a folder containing the input knowledge graph
        :param deserialize_flag: A path of a folder containing previously parsed data
        :param large_kg_parse: A flag for using all cores to parse input knowledge graph
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval: A flag indicating whether evaluation will be applied. If no eval, then entity relation mappings will be deleted to free memory.
        """
        if deserialize_flag is None:
            # 1. LOAD Data. (First pass on data)
            self.train_set = self.load_data_parallel(data_dir + '/train.txt', large_kg_parse, read_only_few)
            self.valid_set = self.load_data_parallel(data_dir + '/valid.txt', large_kg_parse, read_only_few)
            self.test_set = self.load_data_parallel(data_dir + '/test.txt', large_kg_parse, read_only_few)
            # 2. Concatenate list of triples. Could be done with DASK
            data = pd.concat([self.train_set, self.valid_set, self.test_set], ignore_index=True)

            # 3. ordered list of entities and relations
            ordered_list = pd.unique(data[['subject', 'object']].values.ravel('K'))
            self.entity_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                              columns=['entity'],
                                              index=ordered_list)
            ordered_list = pd.unique(data['relation'].values.ravel('K'))
            self.relation_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                                columns=['relation'],
                                                index=ordered_list)
            self.num_entities = len(self.entity_to_idx)
            self.num_relations = len(self.relation_to_idx)

            del ordered_list, data
            """
            if large_kg_parse == 1:
                # 3. ordered list of entities and relations
                ordered_list = pd.unique(data[['subject', 'object']].values.ravel('K'))
                self.entity_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                                  columns=['entity'],
                                                  index=ordered_list)
                self.num_entities = len(self.entity_to_idx)
                ordered_list = pd.unique(data['relation'].values.ravel('K'))
                self.relation_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                                    columns=['relation'],
                                                    index=ordered_list)
                self.num_relations = len(self.relation_to_idx)
                del ordered_list, data
            else:
                raise ValueError()
                # 2. INDEX. (SECOND pass over all triples)
                self.entity_idx, self.relation_idx, self.er_vocab, self.re_vocab, self.ee_vocab = self.index(data,
                                                                                                             add_reciprical=add_reciprical)
                # 3. INDEX Triples for training
                self.triple_indexing(large_kg_parse)
            """

            # 4. Display info
            s = '------------------- Description of Dataset' + data_dir + '----------------------------'
            print(f'\n{s}')
            print(f'Number of entities: {self.num_entities}')
            print(f'Number of relations: {self.num_relations}')

            print(f'Number of triples on train set: {len(self.train_set)}')
            print(f'Number of triples on valid set: {len(self.valid_set)}')
            print(f'Number of triples on test set: {len(self.test_set)}')
            s = len(s) * '-'
            print(f'{s}\n')
        else:
            print('DESERIALIZE')
            self.deserialize(deserialize_flag, eval)

    def serialize(self, p: str) -> None:
        # Serialize entities and relations sotred in pandas dataframe and predicates
        self.entity_to_idx: pd.DataFrame
        self.relation_to_idx: pd.DataFrame

        self.entity_to_idx.to_parquet(p + '/entity_to_idx.gzip', compression='gzip')
        self.relation_to_idx.to_parquet(p + '/relation_to_idx.gzip', compression='gzip')
        # Convert from pandas dataframe to dictionaries
        self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
        self.relation_to_idx = self.relation_to_idx.to_dict()['relation']

        assert len(self.entity_to_idx) == self.num_entities
        assert len(self.relation_to_idx) == self.num_relations

        # Store data in parquet format
        if len(self.train_set) > 0:
            self.train_set.to_parquet(p + '/train_df.gzip', compression='gzip')
            # Store as numpy
            self.train_set['subject'] = self.train_set['subject'].map(lambda x: self.entity_to_idx[x])
            self.train_set['relation'] = self.train_set['relation'].map(lambda x: self.relation_to_idx[x])
            self.train_set['object'] = self.train_set['object'].map(lambda x: self.entity_to_idx[x])
            self.train_set.to_parquet(p + '/idx_train_df.gzip', compression='gzip')
            self.train_set = self.train_set.values
            # Sanity checking
            assert self.num_entities > max(self.train_set[0])
            assert self.num_entities > max(self.train_set[0])
            assert self.num_entities > max(self.train_set[2])
            assert self.num_entities > max(self.train_set[2])

            assert isinstance(self.train_set[0], np.ndarray)
            assert isinstance(self.train_set[0][0], np.int64)
            assert isinstance(self.train_set[0][1], np.int64)
            assert isinstance(self.train_set[0][2], np.int64)

        if len(self.valid_set) > 0:
            self.valid_set.to_parquet(p + '/valid_df.gzip', compression='gzip')
            self.valid_set['subject'] = self.valid_set['subject'].map(lambda x: self.entity_to_idx[x])
            self.valid_set['relation'] = self.valid_set['relation'].map(lambda x: self.relation_to_idx[x])
            self.valid_set['object'] = self.valid_set['object'].map(lambda x: self.entity_to_idx[x])
            self.valid_set.to_parquet(p + '/idx_valid_df.gzip', compression='gzip')
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

        if len(self.test_set) > 0:
            self.test_set.to_parquet(p + '/test_df.gzip', compression='gzip')
            self.test_set['subject'] = self.test_set['subject'].map(lambda x: self.entity_to_idx[x])
            self.test_set['relation'] = self.test_set['relation'].map(lambda x: self.relation_to_idx[x])
            self.test_set['object'] = self.test_set['object'].map(lambda x: self.entity_to_idx[x])
            self.test_set.to_parquet(p + '/idx_test_df.gzip', compression='gzip')
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

    def deserialize(self, p: str, eval=True) -> None:
        """
        Deserialize data
        """
        # @ TODO Serialize data via parque
        if eval:
            print('Deserialize er_vocab')
            with open(p + '/er_vocab.pickle', 'rb') as reader:
                self.er_vocab = pickle.load(reader)
            print('Deserialize re_vocab')
            with open(p + '/re_vocab.pickle', 'rb') as reader:
                self.re_vocab = pickle.load(reader)
            print('Deserialize ee_vocab')
            with open(p + '/ee_vocab.pickle', 'rb') as reader:
                self.ee_vocab = pickle.load(reader)
        else:
            self.er_vocab, self.re_vocab, self.ee_vocab = None, None, None

        # Serialize JsonFiles
        print('Deserialize entity_idx')
        with open(p + "/entity_idx.json", "r") as reader:
            self.entity_idx = json.load(reader)
        print('Deserialize relation_idx')
        with open(p + "/relation_idx.json", "r") as reader:
            self.relation_idx = json.load(reader)
        print('Deserialize index datasets')
        loaded = np.load(p + '/indexed_splits.npz')

        self.train = loaded['train']
        self.valid = loaded['valid']
        self.test = loaded['test']

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
    def load_data_parallel(data_path, large_kg_parse=True, read_only_few: int = None) -> List:
        """
        Parse KG via DASK.
        :param read_only_few:
        :param data_path:
        :param large_kg_parse:
        :return:
        """
        print(f'LOADING {data_path} large kg:{large_kg_parse}')
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
            if large_kg_parse:
                df = df.compute(scheduler='processes')
            else:
                df = df.compute(scheduler='single-threaded')
            x, y = df.shape
            assert y == 3
            print(f'Parsed via DASK: {df.shape}. Whitespace is used as delimiter.')
            if is_nt_format:
                # Drop rows having ^^
                df.drop(df[df["object"].str.contains('<http://www.w3.org/2001/XMLSchema#double>')].index, inplace=True)
                print(f'Drop triples having numerical values are droped: Current size {len(df)}')
                df.drop(df[df["object"].str.contains('<http://www.w3.org/2001/XMLSchema#boolean>')].index, inplace=True)
                print(f'Drop triples having boolean values are droped: Current size {len(df)}')
                df['subject'] = df['subject'].str.removeprefix("<").str.removesuffix(">")
                df['relation'] = df['relation'].str.removeprefix("<").str.removesuffix(">")
                df['object'] = df['object'].str.removeprefix("<").str.removesuffix(">")
                return df  # .values.tolist()
            else:
                return df  # .values.tolist()
            """
            if is_nt_format:
                print('File is Ntriple => ')
                triples = []
                # TODO: do it by using all cores
                for i in df.values.tolist():
                    s, p, o = i[0], i[1], i[2]
                    if s[0] == '<' and s[-1] == '>':
                        s = s[1:-1]
                    if p[0] == '<' and p[-1] == '>':
                        p = p[1:-1]
                    if o[0] == '<' and o[-1] == '>':
                        o = o[1:-1]
                    triples.append([s, p, o])
                return triples
            else:
                return df.values.tolist()
            """
        else:
            print(f'{data_path} could not found ')
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
        if self.valid_set is not None and self.test_set is not None:
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

    # Not used anymore.
    def load_data(self, data_path, add_reciprical=True, load_only=None):
        raise NotImplemented()
        # line can be 1 or 2
        # a) <...> <...> <...> .
        # b) <...> <...> "..." .
        # c) ... ... ...
        # (a) and (b) correspond to the N-Triples format
        # (c) corresponds to the format of current link prediction benchmark datasets.
        print(f'{data_path} is being read.')
        try:
            data = []
            with open(data_path, "r") as f:
                for line in f:
                    # 1. Ignore lines with *** " *** or does only contain 2 or less characters.
                    if '"' in line or len(line) < 3:
                        continue

                    # 2. Tokenize(<...> <...> <...> .) => ['<...>', '<...>','<...>','.']
                    # Tokenize(... ... ...) => ['...', '...', '...',]
                    decomposed_list_of_strings = line.split()

                    # 3. Sanity checking.
                    try:
                        assert len(decomposed_list_of_strings) == 3 or len(decomposed_list_of_strings) == 4
                    except AssertionError:
                        print(f'Invalid input triple {line}. It can not be split into 3 or 4 items')
                        print('This triple will be ignored')
                        continue
                    # 4. Storing
                    if len(decomposed_list_of_strings) == 4:
                        assert decomposed_list_of_strings[-1] == '.'
                        data.append(self.ntriple_parser(decomposed_list_of_strings))
                    if len(decomposed_list_of_strings) == 3:
                        data.append(decomposed_list_of_strings)

                    if load_only is not None:
                        if len(data) == load_only:
                            break

                    if len(data) % 50_000_000 == 0:
                        print(f'Size of already parsed data {len(data)}')

        except FileNotFoundError:
            print(f'{data_path} is not found')
            return []
        if add_reciprical:
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        return data

    def process(self, x):
        raise NotImplemented
        # 2. Tokenize(<...> <...> <...> .) => ['<...>', '<...>','<...>','.']
        # Tokenize(... ... ...) => ['...', '...', '...',]
        decomposed_list_of_strings = x.split()

        # 3. Sanity checking.
        try:
            assert len(decomposed_list_of_strings) == 3 or len(decomposed_list_of_strings) == 4
        except AssertionError:
            print(f'Invalid input triple {x}. It can not be split into 3 or 4 items')
            print('This triple will be ignored')
        # 4. Storing
        if len(decomposed_list_of_strings) == 4:
            assert decomposed_list_of_strings[-1] == '.'
            decomposed_list_of_strings = self.ntriple_parser(decomposed_list_of_strings)
        if len(decomposed_list_of_strings) == 3:
            return decomposed_list_of_strings

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
