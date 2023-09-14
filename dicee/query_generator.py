from collections import defaultdict
from typing import Union, Dict, List, Tuple

import numpy as np
import random
import os
import pickle
import time
from copy import deepcopy


class QueryGenerator:
    def __init__(self, dataset: str, seed: int, gen_train: bool, gen_valid: bool, gen_test: bool):
        self.dataset = dataset

        self.seed = seed
        self.gen_train = gen_train or False
        self.gen_valid = gen_valid or False
        self.gen_test = gen_test or False
        self.max_ans_num = 1e6

        self.mode = str
        self.ent2id: Dict = {}
        self.rel2id: Dict = {}
        self.ent_in: Dict = {}
        self.ent_out: Dict = {}
        self.query_structures = [
            ['e', ['r']],
            ['e', ['r', 'r']],
            ['e', ['r', 'r', 'r']],
            [['e', ['r']], ['e', ['r']]],
            [['e', ['r']], ['e', ['r']], ['e', ['r']]],
            [['e', ['r', 'r']], ['e', ['r']]],
            [[['e', ['r']], ['e', ['r']]], ['r']],
            [['e', ['r']], ['e', ['r', 'n']]],
            [['e', ['r']], ['e', ['r']], ['e', ['r', 'n']]],
            [['e', ['r', 'r']], ['e', ['r', 'n']]],
            [['e', ['r', 'r', 'n']], ['e', ['r']]],
            [[['e', ['r']], ['e', ['r', 'n']]], ['r']],
            # union
            [['e', ['r']], ['e', ['r']], ['u']],
            [[['e', ['r']], ['e', ['r']], ['u']], ['r']]
        ]
        self.query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']

        self.set_global_seed(seed)

    def create_mappings(self):
        """ Create ent2id and rel2id dicts for query generation """

        datapath='./KGs/%s' % self.dataset
        ent2id_path = os.path.join(datapath, "ent2id.pkl")
        rel2id_path = os.path.join(datapath, "rel2id.pkl")

        # If ent2id and rel2id files already exist, read from them
        if os.path.exists(ent2id_path) and os.path.exists(rel2id_path):
            with open(ent2id_path, "rb") as f:
                self.ent2id = pickle.load(f)

            with open(rel2id_path, "rb") as f:
                self.rel2id = pickle.load(f)
            return
        #Otherise create these files
        ent_set = set()
        rel_set = set()
        with open(os.path.join(datapath, "train.txt"), "r") as f:
            for line in f.readlines():
                # Split the line and extract entities and relationships
                ent1, rel, ent2 = line.strip().split('\t')

                # Add entities and relationships to their respective sets
                ent_set.add(ent1)
                ent_set.add(ent2)
                rel_set.add(rel)

        # Create ent2id and rel2id dictionaries
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

        # Save ent2id and rel2id dictionaries to pickle files
        with open(os.path.join(datapath, "ent2id.pkl"), "wb") as f:
            pickle.dump(self.ent2id, f)

        with open(os.path.join(datapath, "rel2id.pkl"), "wb") as f:
            pickle.dump(self.rel2id, f)

    def mapid(self):
        """Convert text triples files into integer id triples for query generation via sampling"""
        datapath='./KGs/%s' % self.dataset
        train_id_path = os.path.join(datapath, "train_id.txt")
        valid_id_path = os.path.join(datapath, "valid_id.txt")
        test_id_path = os.path.join(datapath, "test_id.txt")

        # Check if all three *_id.txt files exist
        if os.path.exists(train_id_path) and os.path.exists(valid_id_path) and os.path.exists(test_id_path):
            return

        # If any of the files are missing, recreate all of them
        for file in ["train", "valid", "test"]:
            filepath = os.path.join(datapath, f"{file}.txt")
            filepath2 = os.path.join(datapath, f"{file}_id.txt")

            with open(filepath, "r") as in_file, open(filepath2, "w") as out_file:
                for line in in_file:
                    ent1, rel, ent2 = line.strip().split('\t')
                    ent1_id = self.ent2id[ent1]
                    rel_id = self.rel2id[rel]
                    ent2_id = self.ent2id[ent2]
                    out_file.write(f"{ent1_id}\t{rel_id}\t{ent2_id}\n")

    def list2tuple(self,l):
        return tuple(self.list2tuple(x) if type(x) == list else x for x in l)

    def tuple2list(self, x: Union[List, Tuple]) -> Union[List, Tuple]:
        """
        Convert a nested tuple to a nested list.
        """
        if isinstance(x, tuple):
            return [self.tuple2list(item) if isinstance(item, tuple) else item for item in x]
        else:
            return x
    def set_global_seed(self, seed: int):
        """Set seed"""
        np.random.seed(seed)
        random.seed(seed)



    def construct_graph(self, base_path: str, indexified_files: List[str]) -> Tuple[Dict, Dict]:
        """
        Construct graph from triples
        Returns dicts with incoming and outgoing edges
        """
        ent_in = defaultdict(lambda: defaultdict(set))
        ent_out = defaultdict(lambda: defaultdict(set))

        for filename in indexified_files:
            with open(f"{base_path}/{filename}", "r") as f:
                for line in f:
                    h, r, t = map(int, line.strip().split("\t"))
                    ent_out[h][r].add(t)
                    ent_in[t][r].add(h)

        self.ent_in = ent_in
        self.ent_out = ent_out

        return ent_in, ent_out

    def fill_query(self, query_structure: List[Union[str, List]],
                    ent_in: Dict, ent_out: Dict,
                    answer: int) -> bool:
        """
        Private method for fill_query logic.
        """
        assert type(query_structure[-1]) == list
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            r = -1
            for i in range(len(query_structure[-1]))[::-1]:
                if query_structure[-1][i] == 'n':
                    query_structure[-1][i] = -2
                    continue
                found = False
                for j in range(40):
                    if len(ent_in[answer].keys()) < 1:
                        return True  # not enough relations, return True to indicate broken flag
                    r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                    if r_tmp // 2 != r // 2 or r_tmp == r:
                        r = r_tmp
                        found = True
                        break
                if not found:
                    return True
                query_structure[-1][i] = r
                answer = random.sample(ent_in[answer][r], 1)[0]
            if query_structure[0] == 'e':
                query_structure[0] = answer
            else:
                return self.fill_query(query_structure[0], ent_in, ent_out, answer)
        else:
            same_structure = defaultdict(list)
            for i in range(len(query_structure)):
                same_structure[self.list2tuple(query_structure[i])].append(i)
            for i in range(len(query_structure)):
                if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                    assert i == len(query_structure) - 1
                    query_structure[i][0] = -1
                    continue
                broken_flag = self.fill_query(query_structure[i], ent_in, ent_out, answer)
                if broken_flag:
                    return True
            for structure in same_structure:
                if len(same_structure[structure]) != 1:
                    structure_set = set()
                    for i in same_structure[structure]:
                        structure_set.add(self.list2tuple(query_structure[i]))
                    if len(structure_set) < len(same_structure[structure]):
                        return True

    def achieve_answer(self, query: List[Union[str, List]],
                        ent_in: Dict, ent_out: Dict) -> set:
        """
        Private method for achieve_answer logic.
        """
        assert type(query[-1]) == list
        all_relation_flag = True
        for ele in query[-1]:
            if (type(ele) != int) or (ele == -1):
                all_relation_flag = False
                break
        if all_relation_flag:
            if type(query[0]) == int:
                ent_set = set([query[0]])
            else:
                ent_set = self.achieve_answer(query[0], ent_in, ent_out)
            for i in range(len(query[-1])):
                if query[-1][i] == -2:
                    ent_set = set(range(len(ent_in))) - ent_set
                else:
                    ent_set_traverse = set()
                    for ent in ent_set:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                    ent_set = ent_set_traverse
        else:
            ent_set = self.achieve_answer(query[0], ent_in, ent_out)
            union_flag = False
            if len(query[-1]) == 1 and query[-1][0] == -1:
                union_flag = True
            for i in range(1, len(query)):
                if not union_flag:
                    ent_set = ent_set.intersection(self.achieve_answer(query[i], ent_in, ent_out))
                else:
                    if i == len(query) - 1:
                        continue
                    ent_set = ent_set.union(self.achieve_answer(query[i], ent_in, ent_out))
        return ent_set

    def ground_queries(self, query_structure: List[Union[str, List]],
                       ent_in: Dict, ent_out: Dict, small_ent_in: Dict, small_ent_out: Dict,
                       gen_num: int, query_name: str):
        """Generating queries and achieving answers"""
        num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
        tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        s0 = time.time()

        old_num_sampled = -1
        while num_sampled < gen_num:

            num_try += 1
            empty_query_structure = deepcopy(query_structure)
            answer = random.sample(list(ent_in.keys()), 1)[0]

            broken_flag = self.fill_query(empty_query_structure, ent_in, ent_out, answer)

            if broken_flag:
                num_broken += 1
                continue

            query = empty_query_structure

            answer_set = self.achieve_answer(query, ent_in, ent_out)
            small_answer_set = self.achieve_answer(query, small_ent_in, small_ent_out)

            if len(answer_set) == 0:
                num_empty += 1
                continue

            if len(answer_set - small_answer_set) == 0:
                num_no_extra_answer += 1
                continue

            if 'n' in query_name:
                if len(small_answer_set - answer_set) == 0:
                    num_no_extra_negative += 1
                    continue

            if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > self.max_ans_num:
                num_more_answer += 1
                continue

            if self.list2tuple(query) in queries[self.list2tuple(query_structure)]:
                num_repeat += 1
                continue

            queries[self.list2tuple(query_structure)].add(self.list2tuple(query))
            tp_answers[self.list2tuple(query)] = small_answer_set
            fp_answers[self.list2tuple(query)] = small_answer_set - answer_set
            fn_answers[self.list2tuple(query)] = answer_set - small_answer_set

            num_sampled += 1
            tp_ans_num.append(len(tp_answers[self.list2tuple(query)]))
            fp_ans_num.append(len(fp_answers[self.list2tuple(query)]))
            fn_ans_num.append(len(fn_answers[self.list2tuple(query)]))


        return queries, tp_answers, fp_answers, fn_answers

    def unmap(self, query_type, queries, tp_answers, fp_answers, fn_answers):

        # Create id2ent dictionary
        id2ent = {v: k for k, v in self.ent2id.items()}
        id2rel = {v: k for k, v in self.rel2id.items()}

        # Unmap queries and create a mapping from ID-based queries to text-based queries
        unmapped_queries_dict = defaultdict(set)
        query_id_to_text = {}
        for query_structure_tuple, query_set in queries.items():
            for query in query_set:
                unmapped_query = self.unmap_query(query_structure_tuple, query, id2ent, id2rel)
                unmapped_queries_dict[query_structure_tuple].add(unmapped_query)
                query_id_to_text[query] = unmapped_query


        easy_answers = defaultdict(set)
        false_positives = defaultdict(set)
        hard_answers = defaultdict(set)
        for query, answer_set in tp_answers.items():
            unmapped_answer_set = {id2ent[answer] for answer in answer_set}
            easy_answers[query_id_to_text[query]] = unmapped_answer_set

            # Unmap fp_answers and update to false_positives
        for query, answer_set in fp_answers.items():
            unmapped_answer_set = {id2ent[answer] for answer in answer_set}
            false_positives[query_id_to_text[query]] = unmapped_answer_set

            # Unmap fn_answers and update to hard_answers
        for query, answer_set in fn_answers.items():
            unmapped_answer_set = {id2ent[answer] for answer in answer_set}
            hard_answers[query_id_to_text[query]] = unmapped_answer_set

            # Save the unmapped queries and answers


        return unmapped_queries_dict, easy_answers, false_positives, hard_answers




    def unmap_query(self,query_structure, query, id2ent, id2rel):
        # 2i
        if query_structure == (("e", ("r",)), ("e", ("r",))):
            ent1, (rel1_id,) = query[0]
            ent2, (rel2_id,) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            return ((ent1, (rel1,)), (ent2, (rel2,)))
        # 3i
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r",))):
            ent1, (rel1_id,) = query[0]
            ent2, (rel2_id,) = query[1]
            ent3, (rel3_id,) = query[2]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            ent3 = id2ent[ent3]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return ((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (rel3,)))
        # 2p
        elif query_structure == ("e", ("r", "r")):
            ent1, (rel1_id, rel2_id) = query
            ent1 = id2ent[ent1]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            return (ent1, (rel1, rel2))
        # 3p
        elif query_structure == ("e", ("r", "r", "r")):
            ent1, (rel1_id, rel2_id, rel3_id) = query
            ent1 = id2ent[ent1]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return (ent1, (rel1, rel2, rel3))
        # pi
        elif query_structure == (("e", ("r", "r")), ("e", ("r",))):
            ent1, (rel1_id, rel2_id) = query[0]
            ent2, (rel3_id,) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return ((ent1, (rel1, rel2)), (ent2, (rel3,)))
        # ip
        elif query_structure == ((("e", ("r",)), ("e", ("r",))), ("r",)):
            ent1, (rel1_id,) = query[0][0]
            ent2, (rel2_id,) = query[0][1]
            (rel3_id,) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return (((ent1, (rel1,)), (ent2, (rel2,))), (rel3,))
        # negation
        # 2in
        elif query_structure == (("e", ("r",)), ("e", ("r", "n"))):
            ent1, (rel1_id,) = query[0]
            ent2, (rel2_id, negation) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            return ((ent1, (rel1,)), (ent2, (rel2, "not")))
        # 3in
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))):
            ent1, (rel1_id,) = query[0]
            ent2, (rel2_id,) = query[1]
            ent3, (rel3_id, negation) = query[2]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            ent3 = id2ent[ent3]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return ((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (rel3, "not")))
        # pin
        elif query_structure == (("e", ("r", "r")), ("e", ("r", "n"))):
            ent1, (rel1_id, rel2_id) = query[0]
            ent2, (rel3_id, negation) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return ((ent1, (rel1, rel2)), (ent2, (rel3, "not")))
        # inp
        elif query_structure == ((("e", ("r",)), ("e", ("r", "n"))), ("r",)):
            ent1, (rel1_id,) = query[0][0]
            ent2, (rel2_id, negation) = query[0][1]
            (rel3_id,) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return (((ent1, (rel1,)), (ent2, (rel2, "not"))), (rel3,))
        # pni
        elif query_structure == (("e", ("r", "r", "n")), ("e", ("r",))):
            ent1, (rel1_id, rel2_id, negation) = query[0]
            ent2, (rel3_id,) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return ((ent1, (rel1, rel2, "not")), (ent2, (rel3,)))
        # union
        # 2u
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("u",)):
            ent1, (rel1_id,) = query[0]
            ent2, (rel2_id,) = query[1]

            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            return ((ent1, (rel1,)), (ent2, (rel2,)), ("union",))
        # up
        elif query_structure == ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)):
            ent1, (rel1_id,) = query[0][0]
            ent2, (rel2_id,) = query[0][1]
            (rel3_id,) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            rel3 = id2rel[rel3_id]
            return (((ent1, (rel1,)), (ent2, (rel2,)), ("union",)), (rel3,))

    def generate_queries(self, query_structure: list, gen_num: int, query_type: str):
        """
        Passing incoming and outgoing edges to ground queries depending on mode [train valid or text]
        and getting queries and answers in return
        """
        base_path = './KGs/%s' % self.dataset
        indexified_files = ['train_id.txt', 'valid_id.txt', 'test_id.txt']
        # Create ent2id and rel2id dicts
        self.create_mappings()
        self.mapid()
        #Contruct Graphs to record incoming and outgoing edges
        if self.gen_train or self.gen_valid:
            train_ent_in, train_ent_out = self.construct_graph(base_path, indexified_files[:1])
        if self.gen_valid or self.gen_test:
            valid_ent_in, valid_ent_out = self.construct_graph(base_path, indexified_files[:2])
            valid_only_ent_in, valid_only_ent_out = self.construct_graph(base_path, indexified_files[1:2])
        if self.gen_test:
            test_ent_in, test_ent_out = self.construct_graph(base_path, indexified_files[:3])
            test_only_ent_in, test_only_ent_out = self.construct_graph(base_path, indexified_files[2:3])

        assert len(query_structure) == 1
        idx = 0
        struct = query_structure[idx]
        print('General structure is', struct, "with name", query_type)

        """@Todos look into one hop queries """
        # if query_structure == ['e', ['r']]:
        #     if self.gen_train:
        #         self.write_links(train_ent_out, defaultdict(lambda: defaultdict(set)), self.max_ans_num,
        #                          'train-' + query_name)
        #     if self.gen_valid:
        #         self.write_links(valid_only_ent_out, train_ent_out, self.max_ans_num, 'valid-' + query_name)
        #     if self.gen_test:
        #         self.write_links(test_only_ent_out, valid_ent_out, self.max_ans_num, 'test-' + query_name)
        #     print("Link prediction queries created!")
        #     return

        if self.gen_train:
            self.mode = 'train'
            train_queries, train_tp_answers, train_fp_answers, train_fn_answers = self.ground_queries(
                struct, train_ent_in, train_ent_out, defaultdict(
                lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)), gen_num, query_type)


            return train_queries, train_tp_answers, train_fp_answers, train_fn_answers

        elif self.gen_valid:
            self.mode = 'valid'
            valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers = self.ground_queries(
                struct, valid_ent_in, valid_ent_out, train_ent_in, train_ent_out, gen_num, query_type)
            return valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers

        elif self.gen_test:
            self.mode = 'test'
            test_queries, test_tp_answers, test_fp_answers, test_fn_answers = self.ground_queries(
                struct, test_ent_in, test_ent_out, valid_ent_in, valid_ent_out, gen_num, query_type)
            return test_queries, test_tp_answers, test_fp_answers, test_fn_answers
        print('%s queries generated with structure %s' % (gen_num, struct))
    def save_queries(self,query_type: str, gen_num: int, save_path: str):

        # Find the index of query_type in query_names
        try:
            gen_id = self.query_names.index(query_type)
        except ValueError:
            print(f"Invalid query_type: {query_type}")
            return []
        queries, tp_answers, fp_answers, fn_answers = self.generate_queries(self.query_structures[gen_id:gen_id + 1],
                                                                            gen_num, query_type)
        unmapped_queries, easy_answers, false_positives, hard_answers = self.unmap(query_type, queries, tp_answers, fp_answers, fn_answers)

        # Save the unmapped queries and answers
        name_to_save = f'{self.mode}-{query_type}'
        with open(f'{save_path}/{name_to_save}-queries.pkl', 'wb') as f:
            pickle.dump(unmapped_queries, f)
        with open(f'{save_path}/{name_to_save}-easy-answers.pkl', 'wb') as f:
            pickle.dump(easy_answers, f)
        with open(f'{save_path}/{name_to_save}-false-positives.pkl', 'wb') as f:
            pickle.dump(false_positives, f)
        with open(f'{save_path}/{name_to_save}-hard-answers.pkl', 'wb') as f:
            pickle.dump(hard_answers, f)
    def get_queries(self, query_type: str, gen_num: int) :
        """
        Get queries of a specific type.
        @todo Not sure what to return dicts or lists and answers should be returned or not
        @todo add comments
        """

        # Find the index of query_type in query_names
        try:
            gen_id = self.query_names.index(query_type)
        except ValueError:
            print(f"Invalid query_type: {query_type}")
            return []
        

        queries, tp_answers, fp_answers, fn_answers = self.generate_queries(self.query_structures[gen_id:gen_id+1], gen_num,query_type)
        unmapped_queries, easy_answers, false_positives, hard_answers = self.unmap(query_type, queries, tp_answers,
                                                                                   fp_answers, fn_answers)

        return unmapped_queries, easy_answers, false_positives, hard_answers


# Example usage
# from dicee import QueryGenerator
# q = QueryGenerator(dataset="UMLS",save_path="./KGs/UMLS", seed=42, gen_train=False, gen_valid=False, gen_test=True)
# Either generate queries and save it at the given path
# q.save_queries(query_type="2in",gen_num=10,save_path= " ")
# or else
# query_dict, easy answers, false positives , hard answers=q.get_queries(query_type="2in",gen_num=10)
# use the dict to answer queries
