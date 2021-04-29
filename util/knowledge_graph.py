from typing import List
from collections import defaultdict


class KG:
    def __init__(self, data_dir=None, add_reciprical=False):
        # 1. First pass through data
        s = '------------------- Description of Dataset' + data_dir + '----------------------------'
        print(f'\n{s}')
        self.__train = self.load_data(data_dir + '/train.txt', add_reciprical=add_reciprical)
        self.__valid = self.load_data(data_dir + '/valid.txt', add_reciprical=add_reciprical)
        self.__test = self.load_data(data_dir + '/test.txt', add_reciprical=add_reciprical)

        self.__data = self.__train + self.__valid + self.__test
        self.__entities = self.get_entities(self.__data)
        self.__relations = self.get_relations(self.__data)
        self.__entity_idxs = {self.__entities[i]: i for i in range(len(self.__entities))}
        self.__relation_idxs = {self.__relations[i]: i for i in range(len(self.__relations))}
        print(f'Number of triples: {len(self.__data)}')
        print(f'Number of entities: {len(self.__entities)}')
        print(f'Number of relations: {len(self.__relations)}')

        print(f'Number of triples on train set: {len(self.__train)}')
        print(f'Number of triples on valid set: {len(self.__valid)}')
        print(f'Number of triples on test set: {len(self.__test)}')
        s = len(s) * '-'
        print(f'{s}\n')

        self.train_set_idx = [(self.entity_to_idx[s], self.relation_to_idx[p], self.entity_to_idx[o]) for s, p, o in
                              self.train_set]
        if self.is_valid_test_available():
            self.val_set_idx = [(self.entity_to_idx[s], self.relation_to_idx[p], self.entity_to_idx[o]) for s, p, o in
                                self.val_set]
            self.test_set_idx = [(self.entity_to_idx[s], self.relation_to_idx[p], self.entity_to_idx[o]) for s, p, o in
                                 self.test_set]
        else:
            self.val_set_idx = None
            self.test_set_idx = None

    @property
    def entities(self):
        return self.__entities

    @property
    def relations(self):
        return self.__relations

    @property
    def num_entities(self):
        return len(self.__entities)

    @property
    def entity_to_idx(self):
        return self.__entity_idxs

    @property
    def relation_to_idx(self):
        return self.__relation_idxs

    @property
    def num_relations(self):
        return len(self.__relations)

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

    def load_data(self, data_path, add_reciprical=True):
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

                for line in f.readlines():

                    # 1. Ignore lines with *** " *** or does only contain 2 or less characters.
                    if '"' in line or len(line) < 3:
                        continue

                    # 2. Tokenize(<...> <...> <...> .) => ['<...>', '<...>','<...>','.']
                    # Tokenize(... ... ...) => ['...', '...', '...',]
                    decomposed_list_of_strings = line.split()

                    # 3. Sanity checking.
                    assert len(decomposed_list_of_strings) == 3 or len(decomposed_list_of_strings) == 4
                    # 4. Storing
                    if len(decomposed_list_of_strings) == 4:
                        assert decomposed_list_of_strings[-1] == '.'
                        data.append(self.ntriple_parser(decomposed_list_of_strings))
                    if len(decomposed_list_of_strings) == 3:
                        data.append(decomposed_list_of_strings)
        except FileNotFoundError:
            print(f'{data_path} is not found')
            return []
        if add_reciprical:
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def is_valid_test_available(self):
        if len(self.__valid) > 0 and len(self.__test) > 0:
            return True
        return False

    @property
    def train_set(self):
        return self.__train

    @property
    def val_set(self):
        return self.__valid

    @property
    def test_set(self):
        return self.__test

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
