from typing import Dict, List, Generator
from collections import defaultdict
import numpy as np
import multiprocessing as mp

"""
from multiprocessing import Pool

def process_line(line):
    return "FOO: %s" % line

if __name__ == "__main__":
    pool = Pool(4)
    with open('file.txt') as source_file:
        # chunk the work into batches of 4 lines at a time
        results = pool.map(process_line, source_file, 4)

    print results
    
or 

"""


class KG:
    def __init__(self, data_dir=None, add_reciprical=False, load_only=None, large_kg=True):
        # 1. LOAD Data. (First pass on data)
        if large_kg is False:
            self.train = self.load_data(data_dir + '/train.txt', add_reciprical=add_reciprical, load_only=load_only)
            self.valid = self.load_data(data_dir + '/valid.txt', add_reciprical=add_reciprical, load_only=load_only)
            self.test = self.load_data(data_dir + '/test.txt', add_reciprical=add_reciprical, load_only=load_only)
        else:
            self.train = self.load_data_parallel(data_dir + '/train.txt', add_reciprical=add_reciprical,
                                                 load_only=load_only)
            self.valid = self.load_data_parallel(data_dir + '/valid.txt', add_reciprical=add_reciprical,
                                                 load_only=load_only)
            self.test = self.load_data_parallel(data_dir + '/test.txt', add_reciprical=add_reciprical,
                                                load_only=load_only)

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
            val_set_idx = []
            test_set_idx = []
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

    def load_data_parallel(self, data_path, add_reciprical=True, load_only=None)->Generator:
        # line can be 1 or 2
        # a) <...> <...> <...> .
        # b) <...> <...> "..." .
        # c) ... ... ...
        # (a) and (b) correspond to the N-Triples format
        # (c) corresponds to the format of current link prediction benchmark datasets.
        if add_reciprical:
            print('In data parallel loading, we do not apply recipriocal triples')

        # init objects
        pool = mp.Pool(32)
        print(f'{data_path} is being read.')
        try:
            jobs = []
            with open(data_path, "r") as f:
                for line in f:
                    # 1. Ignore lines with *** " *** or does only contain 2 or less characters.
                    if '"' in line or len(line) < 3:
                        continue
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
