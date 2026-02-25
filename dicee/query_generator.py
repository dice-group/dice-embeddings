from collections import defaultdict
from typing import Union, Dict, List, Tuple, Optional
import numpy as np
import random
import os
import pickle
from copy import deepcopy
from .static_funcs import save_pickle, load_pickle


class QueryGenerator:
    def __init__(self, train_path, val_path: str, test_path: str, ent2id: Dict = None, rel2id: Dict = None,
                 seed: int = 1, gen_valid: bool = False, gen_test: bool = True, attr2id: Dict = None,
                 train_attr_path: Optional[str] = None, val_attr_path: Optional[str] = None,
                 test_attr_path: Optional[str] = None):

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.gen_valid = gen_valid
        self.gen_test = gen_test

        self.seed = seed

        self.max_ans_num = 1e6

        self.mode = str
        self.ent2id = ent2id
        self.rel2id: Dict = rel2id
        self.attr2id: Dict = attr2id
        self.train_attr_path = train_attr_path
        self.val_attr_path = val_attr_path
        self.test_attr_path = test_attr_path
        self.ent_in: Dict = {}
        self.ent_out: Dict = {}
        self.query_name_to_struct = {"1p": ['e', ['r']],
                                     "2p": ['e', ['r', 'r']],
                                     "3p": ['e', ['r', 'r', 'r']],
                                     "2i": [['e', ['r']], ['e', ['r']]], # @TODO: double check the evaluation
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
                                     "up": [[['e', ['r']], ['e', ['r']], ['u']], ['r']],
                                     # literals
                                     "ai": ['e', ['a', 'f']],
                                     "2ai": [['e', ['a', 'f']], ['e', ['a', 'f']]],
                                     "pai": [['e', ['r']], ['a', 'f']],
                                     "aip": [['e', ['a', 'f']], ['r']],
                                     "au": [['e', ['a', 'f']], ['e', ['a', 'f']], ['u']],
                                     "1ap": ['e', ['a']],
                                     "2ap": ['e', ['r', 'a']],
                                     "3ap": ['e', ['r', 'r', 'a']]}
        self.set_global_seed(seed)

        # Sanity checking
        assert isinstance(self.ent2id, dict) or self.ent2id is None
        assert isinstance(self.rel2id, dict) or self.rel2id is None
        assert isinstance(self.attr2id, dict) or self.attr2id is None

    def list2tuple(self, list_data):
        # @TODO: add description
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

    def construct_graph(self, paths: List[str], attr_paths: Optional[List[str]] = None) -> Tuple[Dict, Dict, Dict, Dict]:
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

        ent_attr_out = defaultdict(lambda: defaultdict(set))
        attr_to_ents = defaultdict(lambda: defaultdict(set))
        if attr_paths is not None and self.attr2id is not None:
            for path in attr_paths:
                if path is None:
                    continue
                with open(path, "r") as f:
                    for line in f:
                        e, a, v = map(str, line.strip().split("\t"))
                        e_id, a_id = self.ent2id[e], self.attr2id[a]
                        ent_attr_out[e_id][a_id].add(v)
                        attr_to_ents[a_id][v].add(e_id)

        self.ent_in = tail_relation_to_heads
        self.ent_out = head_relation_to_tails

        return tail_relation_to_heads, head_relation_to_tails, ent_attr_out, attr_to_ents

    def fill_query(self, query_structure: List[Union[str, List]],
                   ent_in: Dict, ent_out: Dict,
                   ent_attr_out: Dict, attr_to_ents: Dict,
                   answer: int) -> bool:
        """
        Private method for fill_query logic.
        """
        if not isinstance(query_structure[-1], list):
            # Support operator-only leaves such as ['a', 'f'].
            if isinstance(query_structure, list):
                wrapped_query = ['e', query_structure]
                return self.fill_query(wrapped_query, ent_in, ent_out, ent_attr_out, attr_to_ents, answer)
            return True
        all_relation_flag = True
        for ele in query_structure[-1]:
            if isinstance(ele, str) and ele not in ['r', 'n', 'a', 'f']:
                all_relation_flag = False
                break
        if all_relation_flag:
            r = -1
            for i in range(len(query_structure[-1]))[::-1]:
                if query_structure[-1][i] == 'n':
                    query_structure[-1][i] = -2
                    continue
                if query_structure[-1][i] == 'a':
                    if ent_attr_out is None or len(ent_attr_out[answer].keys()) < 1:
                        return True
                    a_tmp = random.sample(sorted(ent_attr_out[answer].keys()), 1)[0]
                    query_structure[-1][i] = a_tmp
                    continue
                if query_structure[-1][i] == 'f':
                    if ent_attr_out is None or len(ent_attr_out[answer].keys()) < 1:
                        return True
                    if i > 0 and isinstance(query_structure[-1][i - 1], int):
                        a_tmp = query_structure[-1][i - 1]
                    else:
                        a_tmp = random.sample(sorted(ent_attr_out[answer].keys()), 1)[0]
                        if i > 0 and query_structure[-1][i - 1] == 'a':
                            query_structure[-1][i - 1] = a_tmp

                    if len(ent_attr_out[answer][a_tmp]) < 1:
                        return True
                    ops = ['=', '<', '>']
                    op = random.sample(ops, 1)[0]
                    if op == '=':
                        f_val = random.sample(sorted(ent_attr_out[answer][a_tmp]), 1)[0]
                    else:
                        numeric_vals = []
                        candidate_vals = attr_to_ents[a_tmp].keys() if attr_to_ents is not None else ent_attr_out[answer][a_tmp]
                        for lit in candidate_vals:
                            try:
                                numeric_vals.append(float(lit))
                            except (TypeError, ValueError):
                                continue
                        if len(numeric_vals) < 1:
                            op = '='
                            f_val = random.sample(sorted(ent_attr_out[answer][a_tmp]), 1)[0]
                        else:
                            f_val = float(np.mean(numeric_vals))
                    query_structure[-1][i] = (op, f_val)
                    continue
                if query_structure[-1][i] != 'r':
                    continue
                found = False
                for j in range(40):
                    if len(ent_in[answer].keys()) < 1:
                        return True  # not enough relations, return True to indicate broken flag
                    r_tmp = random.sample(sorted(ent_in[answer].keys()), 1)[0]
                    if r_tmp // 2 != r // 2 or r_tmp == r:
                        r = r_tmp
                        found = True
                        break
                if not found:
                    return True
                query_structure[-1][i] = r
                answer = random.sample(sorted(ent_in[answer][r]), 1)[0]
            if query_structure[0] == 'e':
                query_structure[0] = answer
            else:
                return self.fill_query(query_structure[0], ent_in, ent_out, ent_attr_out, attr_to_ents, answer)
        else:
            same_structure = defaultdict(list)
            for i in range(len(query_structure)):
                same_structure[self.list2tuple(query_structure[i])].append(i)
            for i in range(len(query_structure)):
                if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                    assert i == len(query_structure) - 1
                    query_structure[i][0] = -1
                    continue
                broken_flag = self.fill_query(query_structure[i], ent_in, ent_out, ent_attr_out, attr_to_ents, answer)
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
                       ent_in: Dict, ent_out: Dict,
                       ent_attr_out: Dict = None, attr_to_ents: Dict = None,
                       query_name: str = None) -> set:
        """
        Private method for achieve_answer logic.
        @TODO: Document the code
        """
        assert isinstance(query[-1], list)
        path_ops = query[-1]

        # literal filter pattern: [..., [attr_id, (op, value)]]
        if len(path_ops) >= 2 and isinstance(path_ops[-1], tuple) and isinstance(path_ops[-2], int):
            attr_id = path_ops[-2]
            op, threshold = path_ops[-1]
            prefix_ops = path_ops[:-2]

            if isinstance(query[0], int):
                ent_set = set([query[0]])
            else:
                ent_set = self.achieve_answer(query[0], ent_in, ent_out, ent_attr_out, attr_to_ents, query_name)

            for rel_or_neg in prefix_ops:
                if rel_or_neg == -2:
                    ent_set = set(range(len(ent_in))) - ent_set
                else:
                    ent_set_traverse = set()
                    for ent in ent_set:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][rel_or_neg])
                    ent_set = ent_set_traverse

            filtered_entities = set()
            if attr_to_ents is not None:
                for lit, ents in attr_to_ents[attr_id].items():
                    if op == "=":
                        threshold_num = None
                        lit_num = None
                        try:
                            threshold_num = float(threshold)
                        except (TypeError, ValueError):
                            threshold_num = None
                        try:
                            lit_num = float(lit)
                        except (TypeError, ValueError):
                            lit_num = None

                        if threshold_num is not None and lit_num is not None:
                            if lit_num == threshold_num:
                                filtered_entities = filtered_entities.union(ents)
                        elif str(lit) == str(threshold):
                            filtered_entities = filtered_entities.union(ents)
                        continue

                    try:
                        lit_val = float(lit)
                        threshold_val = float(threshold)
                    except (TypeError, ValueError):
                        continue
                    if (op == "<" and lit_val < threshold_val) or (op == ">" and lit_val > threshold_val):
                        filtered_entities = filtered_entities.union(ents)

            if len(prefix_ops) == 0:
                return filtered_entities
            return ent_set.intersection(filtered_entities)

        all_relation_flag = True
        for ele in path_ops:
            # @TODO: unclear
            if not isinstance(ele, int) or (ele == -1):
                all_relation_flag = False
                break
        if all_relation_flag:
            if isinstance(query[0], int):
                # @TODO: unclear
                ent_set = set([query[0]])
            else:
                ent_set = self.achieve_answer(query[0], ent_in, ent_out, ent_attr_out, attr_to_ents, query_name)
            for i in range(len(path_ops)):
                if path_ops[i] == -2:
                    ent_set = set(range(len(ent_in))) - ent_set
                else:
                    if query_name in {"1ap", "2ap", "3ap"} and i == len(path_ops) - 1:
                        lit_set = set()
                        if ent_attr_out is not None:
                            for ent in ent_set:
                                lit_set = lit_set.union(ent_attr_out[ent][path_ops[i]])
                        ent_set = lit_set
                        continue
                    ent_set_traverse = set()
                    for ent in ent_set:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][path_ops[i]])
                    ent_set = ent_set_traverse
        else:
            ent_set = self.achieve_answer(query[0], ent_in, ent_out, ent_attr_out, attr_to_ents, query_name)
            union_flag = False
            if len(query[-1]) == 1 and query[-1][0] == -1:
                union_flag = True
            for i in range(1, len(query)):
                if not union_flag:
                    ent_set = ent_set.intersection(
                        self.achieve_answer(query[i], ent_in, ent_out, ent_attr_out, attr_to_ents, query_name))
                else:
                    if i == len(query) - 1:
                        continue
                    ent_set = ent_set.union(
                        self.achieve_answer(query[i], ent_in, ent_out, ent_attr_out, attr_to_ents, query_name))
        return ent_set

    def write_links(self, ent_out, small_ent_out):
        # @TODO: Explain why this is needed
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
                       gen_num: int, query_name: str,
                       ent_attr_out: Dict = None, attr_to_ents: Dict = None,
                       small_ent_attr_out: Dict = None, small_attr_to_ents: Dict = None):
        """Generating queries and achieving answers"""
        (num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer,
         num_no_extra_negative, num_empty) = 0, 0, 0, 0, 0, 0, 0, 0
        tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        if 'a' in query_name and ent_attr_out is not None:
            answer_population = sorted(set(ent_in.keys()).union(set(ent_attr_out.keys())))
        else:
            answer_population = sorted(ent_in.keys())
        if len(answer_population) == 0:
            return queries, tp_answers, fp_answers, fn_answers

        # @TODO: Incorrect reasoning: It can enter an infinite loop
        while num_sampled < gen_num:
            if num_try == 100_000:
                break
            num_try += 1
            # @TODO: Why do we need a deep copy here ?
            query = deepcopy(query_structure)
            answer = random.sample(answer_population, 1)[0]
            broken_flag = self.fill_query(query, ent_in, ent_out, ent_attr_out, attr_to_ents, answer)

            if broken_flag:
                num_broken += 1
                continue

            answer_set = self.achieve_answer(query, ent_in, ent_out, ent_attr_out, attr_to_ents, query_name)
            small_answer_set = self.achieve_answer(
                query, small_ent_in, small_ent_out, small_ent_attr_out, small_attr_to_ents, query_name)

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
                print(num_more_answer)
                continue

            if self.list2tuple(query) in queries[self.list2tuple(query_structure)]:
                num_repeat += 1
                continue

            easy = small_answer_set
            false_pos = small_answer_set - answer_set
            hard = answer_set - small_answer_set
            # easy and false_pos intentionally overlap by definition (false_pos ⊆ easy).
            # Keep only the meaningful consistency checks.
            if (easy & hard) or (false_pos & hard):
                continue

            queries[self.list2tuple(query_structure)].add(self.list2tuple(query))
            tp_answers[self.list2tuple(query)] = easy
            fp_answers[self.list2tuple(query)] = false_pos
            fn_answers[self.list2tuple(query)] = hard

            num_sampled += 1
            tp_ans_num.append(len(tp_answers[self.list2tuple(query)]))
            fp_ans_num.append(len(fp_answers[self.list2tuple(query)]))
            fn_ans_num.append(len(fn_answers[self.list2tuple(query)]))

        return queries, tp_answers, fp_answers, fn_answers

    def unmap(self, query_type, queries, tp_answers, fp_answers, fn_answers):

        # Create id2ent dictionary
        id2ent = {v: k for k, v in self.ent2id.items()}
        id2rel = {v: k for k, v in self.rel2id.items()}
        id2attr = {v: k for k, v in self.attr2id.items()} if self.attr2id else {}

        # Unmap queries and create a mapping from ID-based queries to text-based queries
        unmapped_queries_dict = defaultdict(set)
        query_id_to_text = {}
        for query_structure_tuple, query_set in queries.items():
            for query in query_set:
                unmapped_query = self.unmap_query(query_structure_tuple, query, id2ent, id2rel, id2attr)
                unmapped_queries_dict[query_structure_tuple].add(unmapped_query)
                query_id_to_text[query] = unmapped_query

        easy_answers = defaultdict(set)
        false_positives = defaultdict(set)
        hard_answers = defaultdict(set)

        def _unmap_answer(answer):
            if isinstance(answer, int):
                return id2ent.get(answer, answer)
            return answer

        for query, answer_set in tp_answers.items():
            unmapped_answer_set = {_unmap_answer(answer) for answer in answer_set}
            easy_answers[query_id_to_text[query]] = unmapped_answer_set

            # Unmap fp_answers and update to false_positives
        for query, answer_set in fp_answers.items():
            unmapped_answer_set = {_unmap_answer(answer) for answer in answer_set}
            false_positives[query_id_to_text[query]] = unmapped_answer_set

            # Unmap fn_answers and update to hard_answers
        for query, answer_set in fn_answers.items():
            unmapped_answer_set = {_unmap_answer(answer) for answer in answer_set}
            hard_answers[query_id_to_text[query]] = unmapped_answer_set

        return unmapped_queries_dict, easy_answers, false_positives, hard_answers

    def unmap_query(self, query_structure, query, id2ent, id2rel, id2attr):
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
        # literals
        elif query_structure == ("e", ("a", "f")):
            ent1, (attr_id, filter_spec) = query
            ent1 = id2ent[ent1]
            attr = id2attr.get(attr_id, attr_id)
            return (ent1, (attr, filter_spec))
        elif query_structure == (("e", ("a", "f")), ("e", ("a", "f"))):
            ent1, (attr1_id, filter_spec1) = query[0]
            ent2, (attr2_id, filter_spec2) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            attr1 = id2attr.get(attr1_id, attr1_id)
            attr2 = id2attr.get(attr2_id, attr2_id)
            return ((ent1, (attr1, filter_spec1)), (ent2, (attr2, filter_spec2)))
        elif query_structure == (("e", ("a", "f")), ("e", ("a", "f")), ("u",)):
            ent1, (attr1_id, filter_spec1) = query[0]
            ent2, (attr2_id, filter_spec2) = query[1]
            ent1 = id2ent[ent1]
            ent2 = id2ent[ent2]
            attr1 = id2attr.get(attr1_id, attr1_id)
            attr2 = id2attr.get(attr2_id, attr2_id)
            return ((ent1, (attr1, filter_spec1)), (ent2, (attr2, filter_spec2)), ("union",))
        elif query_structure == ("e", ("a",)):
            ent1, (attr_id,) = query
            ent1 = id2ent[ent1]
            attr = id2attr.get(attr_id, attr_id)
            return (ent1, (attr,))
        elif query_structure == ("e", ("r", "a")):
            ent1, (rel1_id, attr_id) = query
            ent1 = id2ent[ent1]
            rel1 = id2rel[rel1_id]
            attr = id2attr.get(attr_id, attr_id)
            return (ent1, (rel1, attr))
        elif query_structure == ("e", ("r", "r", "a")):
            ent1, (rel1_id, rel2_id, attr_id) = query
            ent1 = id2ent[ent1]
            rel1 = id2rel[rel1_id]
            rel2 = id2rel[rel2_id]
            attr = id2attr.get(attr_id, attr_id)
            return (ent1, (rel1, rel2, attr))
        elif query_structure == (("e", ("r",)), ("a", "f")):
            ent1, (rel1_id,) = query[0]
            attr_id, filter_spec = query[1]
            ent1 = id2ent[ent1]
            rel1 = id2rel[rel1_id]
            attr = id2attr.get(attr_id, attr_id)
            return ((ent1, (rel1,)), (attr, filter_spec))
        elif query_structure == (("e", ("a", "f")), ("r",)):
            ent1, (attr_id, filter_spec) = query[0]
            (rel1_id,) = query[1]
            ent1 = id2ent[ent1]
            attr = id2attr.get(attr_id, attr_id)
            rel1 = id2rel[rel1_id]
            return ((ent1, (attr, filter_spec)), (rel1,))
        return query

    def generate_queries(self, query_struct:List, gen_num: int, query_type: str):
        """
        Passing incoming and outgoing edges to ground queries depending on mode [train valid or text]
        and getting queries and answers in return
        @ TODO: create a class for each single query struct
        """


        train_tail_relation_to_heads, train_head_relation_to_tails, train_ent_attr_out, train_attr_to_ents = self.construct_graph(
            paths=[self.train_path],
            attr_paths=[self.train_attr_path] if self.train_attr_path else None,
        )
        val_tail_relation_to_heads, val_head_relation_to_tails, val_ent_attr_out, val_attr_to_ents = self.construct_graph(
            paths=[self.train_path, self.val_path],
            attr_paths=[self.train_attr_path, self.val_attr_path] if self.train_attr_path else None,
        )
        # ?!
        valid_only_ent_in, valid_only_ent_out, valid_only_ent_attr_out, valid_only_attr_to_ents = self.construct_graph(
            paths=[self.val_path, self.test_path],
            attr_paths=[self.val_attr_path, self.test_attr_path] if self.val_attr_path else None,
        )

        test_tail_relation_to_heads, test_head_relation_to_tails, test_ent_attr_out, test_attr_to_ents = self.construct_graph(
            paths=[self.train_path, self.val_path, self.test_path],
            attr_paths=[self.train_attr_path, self.val_attr_path, self.test_attr_path] if self.train_attr_path else None,
        )
        # ?!
        test_only_ent_in, test_only_ent_out, test_only_ent_attr_out, test_only_attr_to_ents = self.construct_graph(
            paths=[self.test_path],
            attr_paths=[self.test_attr_path] if self.test_attr_path else None,
        )
        self.mode = 'test'
        test_queries, test_tp_answers, test_fp_answers, test_fn_answers = self.ground_queries(
            query_struct, test_tail_relation_to_heads, test_head_relation_to_tails, val_tail_relation_to_heads,
            val_head_relation_to_tails, gen_num, query_type,
            ent_attr_out=test_ent_attr_out, attr_to_ents=test_attr_to_ents,
            small_ent_attr_out=val_ent_attr_out, small_attr_to_ents=val_attr_to_ents)
        # @TODO: test_queries has keys that are tuple ,e.g. ('e', ('r',))
        # Yet, query structure defined as a list ['e', ['r']].
        # Fix this inconsistency
        print(
            f"General structure is {query_struct} with name {query_type}. Number of queries generated: {len(test_tp_answers)}")
        return test_queries, test_tp_answers, test_fp_answers, test_fn_answers

    def save_queries(self, query_type: str, gen_num: int, save_path: str):
        """

        """
        if query_type not in self.query_name_to_struct:
            print(f"Invalid query_type: {query_type}")
            return []
        queries, tp_answers, fp_answers, fn_answers = self.generate_queries(self.query_name_to_struct[query_type],
                                                                            gen_num, query_type)
        unmapped_queries, easy_answers, false_positives, hard_answers = self.unmap(query_type, queries, tp_answers,
                                                                                   fp_answers, fn_answers)

        # Save the unmapped queries and answers
        name_to_save = f'{self.mode}-{query_type}'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # TODO: CD: Deprecate the pickle usage for data serialization.
        # TODO: CD: yet since this files are small we can get away with them
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
