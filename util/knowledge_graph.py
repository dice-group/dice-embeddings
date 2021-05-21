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


def performance_debugger(func_name):
    def func_decorator(func):
        def debug(*args, **kwargs):
            starT = time.time()
            print('\n######', func_name, ' ', end='')
            r = func(*args, **kwargs)
            print(f' took  {time.time() - starT:.3f}  seconds')
            return r

        return debug

    return func_decorator


class KG:
    def __init__(self, data_dir=None, deserialize_flag=None, large_kg_parse=False, add_reciprical=False, eval=True):

        if deserialize_flag is None:
            # 1. LOAD Data. (First pass on data)
            self.train = self.load_data_parallel(data_dir + '/train.txt', large_kg_parse)
            self.valid = self.load_data_parallel(data_dir + '/valid.txt', large_kg_parse)
            self.test = self.load_data_parallel(data_dir + '/test.txt', large_kg_parse)
            # 2. Concatenate list of triples. Could be done with DASK
            data = self.train + self.valid + self.test

            self.entity_idx = None
            self.relation_idx = None
            self.train_set_idx = None
            self.val_set_idx = None
            self.test_set_idx = None
            # 2. INDEX. (SECOND pass over all triples)
            self.entity_idx, self.relation_idx, self.er_vocab, self.re_vocab, self.ee_vocab = self.index(data,
                                                                                                         add_reciprical=add_reciprical)

            # 3. INDEX Triples for training
            self.triple_indexing(large_kg_parse)

            # 4. Display info
            s = '------------------- Description of Dataset' + data_dir + '----------------------------'
            print(f'\n{s}')
            print(f'Number of triples: {len(data)}')
            print(f'Number of entities: {len(self.entity_idx)}')
            print(f'Number of relations: {len(self.relation_idx)}')

            print(f'Number of triples on train set: {len(self.train)}')
            print(f'Number of triples on valid set: {len(self.valid)}')
            print(f'Number of triples on test set: {len(self.test)}')
            s = len(s) * '-'
            print(f'{s}\n')
            # Free Memory
            del data
        else:
            print('DESERIALIZE')
            self.deserialize(deserialize_flag, eval)

    def deserialize(self, p:str, eval=True) -> None:
        """
        Deserialize data
        """
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

    def serialize(self, p: str) -> None:
        """
        Serialize
        1- the following mappings
            - head x relation -> tail
            - relation x tail -> head
            - head x tail -> relation
            - Entity to Integer Index
            - Relation to Integer Index
        2. Indexed triples
            - This can be done via DASK @TODO

        :param p:
        :return:
        """
        # Pickle tuple mappings.
        self.__pickle_dump_obj(self.er_vocab, path=p + '/er_vocab.pickle', info='(HEAD & RELATION) to TAIL')
        self.__pickle_dump_obj(self.re_vocab, path=p + '/re_vocab.pickle', info='(RELATION & TAIL) to HEAD')
        self.__pickle_dump_obj(self.ee_vocab, path=p + '/ee_vocab.pickle', info='(HEAD & TAIL) to RELATION')

        self.__json_dump_obj(self.entity_idx, path=p + "/entity_idx.json", info='Entity to Integer Index')

        self.__json_dump_obj(self.relation_idx, path=p + "/relation_idx.json", info='Relation to Integer Index')

        self.__np_save_as_compressed(train=self.train, valid=self.valid, test=self.test, path=p + '/indexed_splits',
                                     info='Indexed sets of triples')

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

    @property
    def num_entities(self):
        return len(self.entity_idx)

    @property
    def num_relations(self):
        return len(self.relation_idx)

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

    @staticmethod
    def load_data_parallel(data_path, large_kg_parse=True) -> List:
        """
        Parse KG via DASK.
        :param data_path:
        :param large_kg_parse:
        :return:
        """
        print(f'LOADING {data_path}')
        if os.path.exists(data_path):
            df = ddf.read_csv(data_path,
                              delim_whitespace=True, header=None,
                              usecols=[0, 1, 2])

            if large_kg_parse:
                df = df.compute(scheduler='processes')
            else:
                df = df.compute(scheduler='single-threaded')
            x, y = df.shape
            assert y == 3
            print(f'Parsed via DASK: {df.shape}. Whitespace is used as delimiter.')
            return df.values.tolist()  # Possibly time consuming
        else:
            return []

    def load_data(self, data_path, add_reciprical=True, load_only=None):
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
        if len(self.valid) > 0 and len(self.test) > 0:
            return True
        return False

    @property
    def train_set(self):
        return self.train

    @property
    def val_set(self):
        return self.valid

    @property
    def test_set(self):
        return self.test

    @property
    def entities_str(self) -> List:
        """
        entity_idx is a dictionary where keys are string representation of entities and
        values are integer indexes
        :return: list of ordered entities
        """
        return list(self.entity_idx.keys())

    @property
    def relations_str(self) -> List:
        """
        relation_idx is a dictionary where keys are string representation of relations and
        values are integer indexes
        :return: list of ordered relations
        """
        return list(self.relation_idx.keys())

    """    
    def get_er_idx_vocab(self):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in self.__data:
            er_vocab[(self.entity_to_idx[triple[0]], self.relation_to_idx[triple[1]])].append(
                self.entity_to_idx[triple[2]])
        return er_vocab

    def get_po_idx_vocab(self):
        # head entity and tail entity
        po_vocab = defaultdict(list)
        for triple in self.__data:
            # Predicate, Object : Subject
            s, p, o = triple[0], triple[1], triple[2]
            po_vocab[(self.relation_to_idx[p], self.entity_to_idx[o])].append(self.entity_to_idx[s])
        return po_vocab

    def get_ee_idx_vocab(self):
        # head entity and tail entity
        ee_vocab = defaultdict(list)
        for triple in self.__data:
            # Subject, Predicate Object
            s, p, o = triple[0], triple[1], triple[2]
            ee_vocab[(self.entity_to_idx[s], self.entity_to_idx[o])].append(self.relation_to_idx[p])
        return ee_vocab
    """

    """
    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
    """
