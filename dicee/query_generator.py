from collections import defaultdict
from typing import Union, Dict, List, Tuple

import numpy as np
import random
import os
import pickle
from copy import deepcopy
from .static_funcs import save_pickle, load_pickle


class QueryGenerator:
    def __init__(self, train_path, val_path: str, test_path: str, ent2id, rel2id, seed: int,
                 gen_valid: bool = False,
                 gen_test: bool = True):

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.gen_valid = gen_valid
        self.gen_test = gen_test

        self.seed = seed

        self.max_ans_num = 1e6

        self.mode = str
        self.ent2id: Dict = ent2id
        self.rel2id: Dict = rel2id
        self.ent_in: Dict = {}
        self.ent_out: Dict = {}
        self.query_name_to_struct = {"1p": ['e', ['r']],
                                     "2p": ['e', ['r', 'r']],
                                     "3p": ['e', ['r', 'r', 'r']],
                                     "2i": [['e', ['r']], ['e', ['r']]],
                                     "3i": [['e', ['r']], ['e', ['r']], ['e', ['r']]],
                                     "pi": [['e', ['r', 'r']], ['e', ['r']]],
                                     "ip": [[['e', ['r']], ['e', ['r']]], ['r']],
                                     "2in": [['e', ['r']], ['e', ['r', 'n']]],
                                     "3in": [['e', ['r']], ['e', ['r']], ['e', ['r', 'n']]],
                                     "pin": [['e', ['r', 'r']], ['e', ['r', 'n']]],
                                     "pni": [['e', ['r', 'r', 'n']], ['e', ['r']]],
                                     "inp": [[['e', ['r']], ['e', ['r', 'n']]], ['r']],
                                     # union
                                     "2u": [['e', ['r']], ['e', ['r']], ['u']],
                                     "up": [[['e', ['r']], ['e', ['r']], ['u']], ['r']]}
        self.set_global_seed(seed)

    def list2tuple(self, list_data):
        """
            Recursively convert a nested list into a nested tuple.

            Args:
                list_data (list): A possibly nested list.

            Returns:
                tuple: A nested tuple converted from the input list.
        """
        return tuple(self.list2tuple(x) if isinstance(x, list) else x for x in list_data)

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

    def construct_graph(self, paths: List[str]) -> Tuple[Dict, Dict]:
        """
        Construct graph from triples
        Returns dicts with incoming and outgoing edges
        """
        # Mapping from tail entity and a relation to heads.
        tail_relation_to_heads = defaultdict(lambda: defaultdict(set))
        # Mapping from head and relation to tails.
        head_relation_to_tails = defaultdict(lambda: defaultdict(set))

        for path in paths:
            with open(path, "r") as f:
                for line in f:
                    h, r, t = map(str, line.strip().split("\t"))
                    tail_relation_to_heads[self.ent2id[t]][self.rel2id[r]].add(self.ent2id[h])
                    head_relation_to_tails[self.ent2id[h]][self.rel2id[r]].add(self.ent2id[t])

        self.ent_in = tail_relation_to_heads
        self.ent_out = head_relation_to_tails

        return tail_relation_to_heads, head_relation_to_tails

    def fill_query(self, query_structure: List[Union[str, List]],
                   ent_in: Dict, ent_out: Dict,
                   answer: int) -> bool:
        """
        Private method for fill_query logic.
        """
        assert isinstance(query_structure[-1], list)
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
            Computes a set of entities that match the given query criteria based on the provided graph.

            Args:
                query (List[Union[str, List]]): The query structure with entities, relations, and special tokens.
                ent_in (Dict): Incoming edges for entities.
                ent_out (Dict): Outgoing edges for entities.

            Returns:
                set: Entities that satisfy the query criteria.
        """

        # Check if the last element of the query is list, indicating a set of relations or sub-queries
        assert isinstance(query[-1], list)

        all_relation_flag = True

        for ele in query[-1]:
            # Check if all elements in last part are relation IDs or negations, without any union indicator (-1).
            if (not isinstance(ele, int)) or (ele == -1):
                all_relation_flag = False
                break

        if all_relation_flag:
            # If the last part of the query contains only relations or negations
            if isinstance(query[0], int):
                # If the first element is an entity, consider it as the starting point
                ent_set = set([query[0]])
            else:
                # Otherwise, recursively compute the entities that match the sub-query
                ent_set = self.achieve_answer(query[0], ent_in, ent_out)

            # Traverse through the relations in the last part of the query
            for i in range(len(query[-1])):
                # If the relation is a negation, consider all entities except those in the current set
                if query[-1][i] == -2:
                    ent_set = set(range(len(ent_in))) - ent_set
                else:
                    # Otherwise, update the entity set based on the outgoing edges for the current relation
                    ent_set_traverse = set()
                    for ent in ent_set:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                    ent_set = ent_set_traverse

        else:
            # If the last part of the query contains other structures (like intersection tuple or unions)
            ent_set = self.achieve_answer(query[0], ent_in, ent_out)

            union_flag = False
            # If the last element is a union token, set the flag to true
            if len(query[-1]) == 1 and query[-1][0] == -1:
                union_flag = True

            # Traverse the rest of the query
            for i in range(1, len(query)):
                if not union_flag:
                    # If not a union, intersect the current entity set with the result of the sub-query
                    ent_set = ent_set.intersection(self.achieve_answer(query[i], ent_in, ent_out))
                else:
                    # If it's a union, combine the current entity set with the result of the sub-query
                    if i == len(query) - 1:
                        continue
                    ent_set = ent_set.union(self.achieve_answer(query[i], ent_in, ent_out))

        return ent_set

    def write_links(self, ent_out, small_ent_out):
        """
            Generates '1p' type queries from entities' outgoing edges.

            Given an entity and relation,
            we directly look up the links using the structure `ent_out[ent][rel]`. This is because
            1p queries inherently represent direct relationships, and thus, the target entities
            can be fetched directly from our knowledge graph without needing to perform complex reasoning.

            Args:
                ent_out: Outgoing edges for entities in the full dataset.
                small_ent_out: Outgoing edges for entities in a smaller dataset.

            Returns:
                queries, tp_answers, fn_answers, fp_answers: Generated queries and corresponding answers.
        """

        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        num_more_answer = 0
        for ent in ent_out:
            for rel in ent_out[ent]:
                if len(ent_out[ent][rel]) <= self.max_ans_num:
                    queries[('e', ('r',))].add((ent, (rel,)))
                    tp_answers[(ent, (rel,))] = small_ent_out[ent][rel]
                    fn_answers[(ent, (rel,))] = ent_out[ent][rel]
                    fp_answers[(ent, (rel,))] = set()
                else:
                    num_more_answer += 1
        return queries, tp_answers, fp_answers, fn_answers

    def ground_queries(self, query_structure: List[Union[str, List]],
                       ent_in: Dict, ent_out: Dict, small_ent_in: Dict, small_ent_out: Dict,
                       gen_num: int, query_name: str):
        """Generating queries and achieving answers"""
        (num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer,
         num_no_extra_negative, num_empty) = 0, 0, 0, 0, 0, 0, 0, 0
        tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        fn_answers = defaultdict(set)


        while num_sampled < gen_num:
            # Threshold for the number of tries
            if num_try == 1000000:
                print('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
                self.mode,
                query_structure,
                num_sampled, gen_num, num_try, num_repeat, num_more_answer,
                num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
                print("query generation with hard answers not equal to 0 not possible , try with a lesser number of queries or different dataset")
                break

            num_try += 1
            # use a deep copy to ensure the integrity of the original structure for subsequent iterations
            query = deepcopy(query_structure)
            answer = random.sample(list(ent_in.keys()), 1)[0]
            broken_flag = self.fill_query(query, ent_in, ent_out, answer)

            if broken_flag:
                num_broken += 1
                continue

            answer_set = self.achieve_answer(query, ent_in, ent_out)
            small_answer_set = self.achieve_answer(query, small_ent_in, small_ent_out)
            # check if the answer set is empty for the given query
            if len(answer_set) == 0:
                num_empty += 1
                continue
            # check if no hard answers are found
            if len(answer_set - small_answer_set) == 0:
                num_no_extra_answer += 1
                continue

            if 'n' in query_name:
                if len(small_answer_set - answer_set) == 0:
                    num_no_extra_negative += 1
                    continue
            # check if the answers generated are greater than maximum number of answers
            if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > self.max_ans_num:
                num_more_answer += 1
                print(num_more_answer)
                continue

            if self.list2tuple(query) in queries[self.list2tuple(query_structure)]:
                num_repeat += 1
                continue
            # Convert generated queries from list to tuple format:
            # - Tuples are hashable: Suitable for dictionary keys.
            # - Immutable nature ensures data consistency.
            # - Provides a standardized format for subsequent operations.
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

        return unmapped_queries_dict, easy_answers, false_positives, hard_answers

    def unmap_query(self, query_structure, query, id2ent, id2rel):
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
        # 1p
        elif query_structure == ("e", ("r",)):
            ent1, (rel1_id,) = query
            ent1 = id2ent[ent1]
            rel1 = id2rel[rel1_id]
            return (ent1, (rel1,))
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

    def generate_queries(self, query_struct, gen_num: int, query_type: str):
        """
        Passing incoming and outgoing edges to ground queries depending on mode [train valid or text]
        and getting queries and answers in return
        """
        train_tail_relation_to_heads, train_head_relation_to_tails = self.construct_graph(paths=[self.train_path])
        val_tail_relation_to_heads, val_head_relation_to_tails = self.construct_graph(
            paths=[self.train_path, self.val_path])
        # needed for 1p valid queries
        valid_only_ent_in, valid_only_ent_out = self.construct_graph(paths=[self.val_path])

        test_tail_relation_to_heads, test_head_relation_to_tails = self.construct_graph(
            paths=[self.train_path, self.val_path, self.test_path])
        # needed for 1p test queries
        test_only_ent_in, test_only_ent_out = self.construct_graph(paths=[self.test_path])
        if query_struct == ['e', ['r']]:
            if self.gen_train:
                queries, tp_answers, fp_answers, fn_answers = self.write_links(train_head_relation_to_tails,
                                                                               defaultdict(lambda: defaultdict(set)),
                                                                               'train')
            if self.gen_valid:
                queries, tp_answers, fp_answers, fn_answers = self.write_links(valid_only_ent_out, train_head_relation_to_tails,
                                                                               'valid')
            if self.gen_test:
                queries, tp_answers, fp_answers, fn_answers = self.write_links(test_only_ent_out, val_head_relation_to_tails, 'test')
            print("Link prediction queries created!")
            return queries, tp_answers, fp_answers, fn_answers

        elif self.gen_test:
            self.mode = 'test'
            test_queries, test_tp_answers, test_fp_answers, test_fn_answers = self.ground_queries(
                query_struct, test_tail_relation_to_heads, test_head_relation_to_tails, val_tail_relation_to_heads,
                val_head_relation_to_tails, gen_num, query_type)

            print(
                f"General structure is {query_struct} with name {query_type}. Number of queries generated: {len(test_tp_answers)}")
            return test_queries, test_tp_answers, test_fp_answers, test_fn_answers

    def save_queries(self, query_type: str, gen_num: int, save_path: str):
        """
        Generate queries with corresponding answers and  save them at the given path
        """

        # Find the index of query_type in query_names
        try:
            gen_id = self.query_names.index(query_type)
        except ValueError:
            print(f"Invalid query_type: {query_type}")
            return []
        queries, tp_answers, fp_answers, fn_answers = self.generate_queries(self.query_structures[gen_id:gen_id + 1],
                                                                            gen_num, query_type)
        unmapped_queries, easy_answers, false_positives, hard_answers = self.unmap(query_type, queries, tp_answers,
                                                                                   fp_answers, fn_answers)

        # Save the unmapped queries and answers
        name_to_save = f'{self.mode}-{query_type}'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}/{name_to_save}-queries.pkl', 'wb') as f:
            pickle.dump(unmapped_queries, f)
        with open(f'{save_path}/{name_to_save}-easy-answers.pkl', 'wb') as f:
            pickle.dump(easy_answers, f)
        with open(f'{save_path}/{name_to_save}-false-positives.pkl', 'wb') as f:
            pickle.dump(false_positives, f)
        with open(f'{save_path}/{name_to_save}-hard-answers.pkl', 'wb') as f:
            pickle.dump(hard_answers, f)

    def load_queries(self, path):
        raise NotImplementedError()

    def get_queries(self, query_type: str, gen_num: int):

        queries, tp_answers, fp_answers, fn_answers = self.generate_queries(self.query_name_to_struct[query_type],
                                                                            gen_num, query_type)
        unmapped_queries, easy_answers, false_positives, hard_answers = self.unmap(query_type, queries, tp_answers,
                                                                                   fp_answers, fn_answers)
        return unmapped_queries, easy_answers, false_positives, hard_answers

    @staticmethod
    def save_queries_and_answers(path: str, data: List[Tuple[str, Tuple[defaultdict]]]) -> None:
        """ Save Queries into Disk"""
        save_pickle(file_path=path, data=data)

    @staticmethod
    def load_queries_and_answers(path: str) -> List[Tuple[str, Tuple[defaultdict]]]:
        """ Load Queries from Disk to Memory"""
        print("Loading...")
        data = load_pickle(file_path=path)
        assert isinstance(data, list)
        assert isinstance(data[0], tuple)
        assert isinstance(data[0][0], str)
        assert isinstance(data[0][1], tuple)
        return data
