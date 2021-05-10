import time
from typing import Dict, List, Generator
from collections import defaultdict
import numpy as np
import multiprocessing
from itertools import zip_longest
import pickle
import json


class KG:
    def __init__(self, data_dir=None, deserialize_flag=None, add_reciprical=False, load_only=None):

        if deserialize_flag is None:
            # 1. LOAD Data. (First pass on data)
            self.train = self.load_data(data_dir + '/train.txt', add_reciprical=add_reciprical, load_only=load_only)
            self.valid = self.load_data(data_dir + '/valid.txt', add_reciprical=add_reciprical, load_only=load_only)
            self.test = self.load_data(data_dir + '/test.txt', add_reciprical=add_reciprical, load_only=load_only)
            data = self.train + self.valid + self.test

            self.entity_idx = None
            self.relation_idx = None
            self.train_set_idx = None
            self.val_set_idx = None
            self.test_set_idx = None
            # 2. INDEX. (SECOND pass over all triples)
            self.entity_idx, self.relation_idx, self.er_vocab, self.re_vocab, self.ee_vocab = self.index(data)

            # 3. INDEX Triples for training
            self.train, self.valid, self.test = self.triple_indexing()

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
            self.deserialize(deserialize_flag)

    def deserialize(self, p):
        """
        """
        print('Deserialize er_vocab')
        with open(p + '/er_vocab.pickle', 'rb') as reader:
            self.er_vocab = pickle.load(reader)
        print('Deserialize re_vocab')
        with open(p + '/re_vocab.pickle', 'rb') as reader:
            self.re_vocab = pickle.load(reader)
        print('Deserialize ee_vocab')
        with open(p + '/ee_vocab.pickle', 'rb') as reader:
            self.ee_vocab = pickle.load(reader)

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

    def serialize(self, p):
        # Pickle tuple mappings.
        with open(p + '/er_vocab.pickle', 'wb') as handle:
            pickle.dump(self.er_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(p + '/re_vocab.pickle', 'wb') as handle:
            pickle.dump(self.re_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(p + '/ee_vocab.pickle', 'wb') as handle:
            pickle.dump(self.ee_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Serialize JsonFiles
        with open(p + "/entity_idx.json", "w") as write_file:
            json.dump(self.entity_idx, write_file)
        with open(p + "/relation_idx.json", "w") as write_file:
            json.dump(self.relation_idx, write_file)
        np.savez_compressed(p + '/indexed_splits', train=self.train, valid=self.valid, test=self.test)

    @staticmethod
    def index(data) -> (Dict, Dict, Dict, Dict, Dict):
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
            h, r, t = triple[0], triple[1], triple[2]

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

        return entity_idxs, relation_idxs, er_vocab, pe_vocab, ee_vocab

    def triple_indexing(self) -> (np.array, np.array, np.array):
        train_set_idx = np.array(
            [(self.entity_idx[s], self.relation_idx[p], self.entity_idx[o]) for s, p, o in
             self.train_set])

        if self.is_valid_test_available():
            val_set_idx = np.array(
                [(self.entity_idx[s], self.relation_idx[p], self.entity_idx[o]) for s, p, o in
                 self.val_set])
            test_set_idx = np.array(
                [(self.entity_idx[s], self.relation_idx[p], self.entity_idx[o]) for s, p, o in
                 self.test_set])
        else:
            val_set_idx = np.array([])
            test_set_idx = np.array([])
        return train_set_idx, val_set_idx, test_set_idx

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

    def load_data_parallel(self, data_path, add_reciprical=True, load_only=None) -> Generator:
        # line can be 1 or 2
        # a) <...> <...> <...> .
        # b) <...> <...> "..." .
        # c) ... ... ...
        # (a) and (b) correspond to the N-Triples format
        # (c) corresponds to the format of current link prediction benchmark datasets.
        if add_reciprical:
            print('In data parallel loading, we do not apply recipriocal triples')
        """
        https://stackoverflow.com/questions/8717179/chunking-data-from-a-large-file-for-multiprocessing
        """
        from pathlib import Path
        size_in_bytes = Path(data_path).stat().st_size  # the size, in bytes,
        num_cores = 4
        import math

        chunk_size_per_core = math.ceil(size_in_bytes / num_cores)
        pool = multiprocessing.Pool(4)
        # data = []
        with open(data_path, "r") as reader:
            for _ in range(num_cores):
                reader.seek(chunk_size_per_core, 0)  # move the file pointer forward 6 bytes (i.e. to the 'w')
                # data.extend(pool.starmap_async(self.process, reader.readlines()[0]))
        return True
        exit(1)
        # wait for all jobs to finish
        for job in data:
            job.get()

        # clean up
        pool.close()

        print(p)
        p.join()
        exit(1)
        # init objects
        print(f'{data_path} is being read.')

        exit(1)
        # init objects
        pool = mp.Pool(4)
        jobs = []

        # create jobs
        for chunkStart, chunkSize in chunkify(data_path):
            jobs.append(pool.apply_async(process_wrapper, (data_path, chunkStart, chunkSize)))

        # wait for all jobs to finish
        for job in jobs:
            job.get()

        # clean up
        pool.close()

        exit(1)
        try:
            with open(data_path, "r") as f:

                print(f)
                exit(1)
                for line in f:
                    # 1. Ignore lines with *** " *** or does only contain 2 or less characters.
                    if '"' in line or len(line) < 3:
                        continue
                    results = pool.map(self.process, line, 4)

                    jobs.append(pool.apply_async(self.process, (line,)))
                    if load_only is not None:
                        if len(jobs) == load_only:
                            break
        except FileNotFoundError:
            print(f'{data_path} is not found')
            return []

        # wait for all jobs to finish
        for job in jobs:
            job.get()
        # clean up
        pool.close()
        return [i.get() for i in jobs]

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


def process_chunk(d):
    """Replace this with your own function
    that processes data one line at a
    time"""

    d = d.strip() + ' processed'
    return d


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
    ('a','b','c'), ('d','e','f'), ('g','x','x')"""

    return zip_longest(*[iter(iterable)] * n)
