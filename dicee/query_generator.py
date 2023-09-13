from collections import defaultdict
from typing import Union, Dict, List, Tuple

import numpy as np
import random
import os
import pickle
import time
import logging
from copy import deepcopy


class QueryGenerator:
    def __init__(self, dataset: str, save_path: str, seed: int, gen_train: bool, gen_valid: bool, gen_test: bool):
        self.dataset = dataset
        self.save_path = save_path
        self.seed = seed
        self.gen_train = gen_train or False
        self.gen_valid = gen_valid or False
        self.gen_test = gen_test or False
        self.max_ans_num = 1e6

        self.save_name = True

        self.ent2id: Dict = {}
        self.rel2id: Dict = {}
        self.ent_in: Dict = {}
        self.ent_out: Dict = {}
        self.set_global_seed(seed)

    def create_mappings(self):
        """ Create ent2id and rel2id dicts for query generation """
        ent_set = set()
        rel_set = set()
        datapath='./KGs/%s' % self.dataset
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
        """ Construct graph from triples
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
                       gen_num: int, query_name: str, mode: str):
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
            # if num_sampled % (gen_num // 10) == 0 and num_sampled != old_num_sampled:
            #     logging.info(
            #         '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s, more_answer: %s, broken: %s, no extra: %s, no negative: %s, empty: %s' % (
            #             mode, query_structure, num_sampled, gen_num, (time.time() - s0) / num_sampled, num_try,
            #             num_repeat,
            #             num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty))

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

        # logging.info("{} tp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(tp_ans_num), np.min(tp_ans_num),
        #                                                                 np.mean(tp_ans_num), np.std(tp_ans_num)))
        # logging.info("{} fp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fp_ans_num), np.min(fp_ans_num),
        #                                                                 np.mean(fp_ans_num), np.std(fp_ans_num)))
        # logging.info("{} fn max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fn_ans_num), np.min(fn_ans_num),
        #                                                                 np.mean(fn_ans_num), np.std(fn_ans_num)))

        name_to_save = f'{mode}-{query_name}'
        with open(f'{self.save_path}/{name_to_save}-queries.pkl', 'wb') as f:
            pickle.dump(queries, f)
        with open(f'{self.save_path}/{name_to_save}-fp-answers.pkl', 'wb') as f:
            pickle.dump(fp_answers, f)
        with open(f'{self.save_path}/{name_to_save}-fn-answers.pkl', 'wb') as f:
            pickle.dump(fn_answers, f)
        with open(f'{self.save_path}/{name_to_save}-tp-answers.pkl', 'wb') as f:
            pickle.dump(tp_answers, f)

        return queries, tp_answers, fp_answers, fn_answers

    def generate_queries(self, query_structures: list, gen_num: int, query_type: str):
        """
        Passing incoming and outgoing edges to ground queries depending on mode [train valid or text]
        and getting queries and answers in return
        """
        base_path = './KGs/%s' % self.dataset
        indexified_files = ['train_id.txt', 'valid_id.txt', 'test_id.txt']

        if self.gen_train or self.gen_valid:
            train_ent_in, train_ent_out = self.construct_graph(base_path, indexified_files[:1])
        if self.gen_valid or self.gen_test:
            valid_ent_in, valid_ent_out = self.construct_graph(base_path, indexified_files[:2])
            valid_only_ent_in, valid_only_ent_out = self.construct_graph(base_path, indexified_files[1:2])
        if self.gen_test:
            test_ent_in, test_ent_out = self.construct_graph(base_path, indexified_files[:3])
            test_only_ent_in, test_only_ent_out = self.construct_graph(base_path, indexified_files[2:3])

        assert len(query_structures) == 1
        idx = 0
        query_structure = query_structures[idx]
        print('General structure is', query_structure, "with name", query_type)

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
            train_queries, train_tp_answers, train_fp_answers, train_fn_answers = self.ground_queries(
                query_structure, train_ent_in, train_ent_out, defaultdict(
                lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)), gen_num, query_type, 'train')
            print('%s queries generated with structure %s' % (gen_num, query_structure))
            return train_queries, train_tp_answers, train_fp_answers, train_fn_answers
        elif self.gen_valid:
            valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers = self.ground_queries(
                query_structure, valid_ent_in, valid_ent_out, train_ent_in, train_ent_out, gen_num, query_type, 'valid')
            return valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers
        elif self.gen_test:
            test_queries, test_tp_answers, test_fp_answers, test_fn_answers = self.ground_queries(
                query_structure, test_ent_in, test_ent_out, valid_ent_in, valid_ent_out, gen_num, query_type, 'test')
            return test_queries, test_tp_answers, test_fp_answers, test_fn_answers




    def get_queries(self, query_type: str, gen_num: int) :
        """
        Get queries of a specific type.
        @todo Not sure what to return dicts or lists and answers should be returned or not
        @todo unmap integer dicts of queries and answers to strings
        @todo add comments
        """

        e = 'e'
        r = 'r'
        n = 'n'
        u = 'u'
        query_structures = [
            [e, [r]],
            [e, [r, r]],
            [e, [r, r, r]],
            [[e, [r]], [e, [r]]],
            [[e, [r]], [e, [r]], [e, [r]]],
            [[e, [r, r]], [e, [r]]],
            [[[e, [r]], [e, [r]]], [r]],
            # negation
            [[e, [r]], [e, [r, n]]],
            [[e, [r]], [e, [r]], [e, [r, n]]],
            [[e, [r, r]], [e, [r, n]]],
            [[e, [r, r, n]], [e, [r]]],
            [[[e, [r]], [e, [r, n]]], [r]],
            # union
            [[e, [r]], [e, [r]], [u]],
            [[[e, [r]], [e, [r]], [u]], [r]]
        ]
        query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']
        # Find the index of query_type in query_names
        try:
            gen_id = query_names.index(query_type)
        except ValueError:
            print(f"Invalid query_type: {query_type}")
            return []
        
        # Create ent2id and rel2id dicts 
        self.create_mappings()
        self.mapid()
        queries, tp_answers, fp_answers, fn_answers=self.generate_queries(query_structures[gen_id:gen_id+1], gen_num,query_type)
        return queries


# Example usage
# from dicee import QueryGenerator
#
# q = QueryGenerator(dataset="UMLS",save_path="./KGs/UMLS", seed=42, gen_train=False, gen_valid=False, gen_test=True)
# query_dict=q.get_queries(query_type="2in",gen_num=10)
# print(query_dict)
