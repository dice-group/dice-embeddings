from dicee import QueryGenerator
from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
import os
import numpy as np
import pickle
import torch

# (1) Train Clifford Embeddings model with AllvsAll on Family dataset
if False:
    args = Namespace()
    args.model = 'Keci'
    args.scoring_technique = "AllvsAll"
    args.path_single_kg = "KGs/UMLS/train.txt"
    args.num_epochs = 100
    args.batch_size = 1024
    args.lr = 0.1
    args.embedding_dim = 512
    reports = Execute(args).start()
    # (2) Load the pretrained model
    pre_trained_kge = KGE(path=reports['path_experiment_folder'])
else:
    pre_trained_kge = KGE(path="Experiments/2023-09-15 19-05-11.090264")
# (3) Generate queries
q = QueryGenerator(dataset="UMLS", seed=0, gen_train=False, gen_valid=False, gen_test=True)


# Either generate queries and save it at the given path
# q.save_queries(query_type="2in",gen_num=10,save_path="UMLS_Queries")
def evaluate(model, scores, easy_answers, hard_answers):
    # @TODO Please move this function into dicee.static
    # Calculate MRR considering the hard and easy answers
    total_mrr = 0
    total_h1 = 0
    total_h3 = 0
    total_h10 = 0
    num_queries = len(scores)

    for query, entity_score in scores.items():
        assert len(entity_score) == len(model.entity_to_idx)
        entity_scores = [(ei, s) for ei, s in zip(model.entity_to_idx.keys(), entity_score)]
        entity_scores = sorted(entity_scores, key=lambda x: x[1], reverse=True)

        # Extract corresponding easy and hard answers
        easy_ans = easy_answers[query]
        hard_ans = hard_answers[query]
        easy_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in easy_ans]
        hard_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in hard_ans]

        answer_indices = easy_answer_indices + hard_answer_indices

        cur_ranking = np.array(answer_indices)

        # Sort by position in the ranking; indices for (easy + hard) answers
        cur_ranking, indices = np.sort(cur_ranking), np.argsort(cur_ranking)
        num_easy = len(easy_ans)
        num_hard = len(hard_ans)

        # Indices with hard answers only
        masks = indices >= num_easy

        # Reduce ranking for each answer entity by the amount of (easy+hard) answers appearing before it
        answer_list = np.arange(num_hard + num_easy, dtype=float)
        cur_ranking = cur_ranking - answer_list + 1

        # Only take indices that belong to the hard answers
        cur_ranking = cur_ranking[masks]
        # print(cur_ranking)
        mrr = np.mean(1.0 / cur_ranking)
        h1 = np.mean((cur_ranking <= 1).astype(float))
        h3 = np.mean((cur_ranking <= 3).astype(float))
        h10 = np.mean((cur_ranking <= 10).astype(float))
        total_mrr += mrr
        total_h1 += h1
        total_h3 += h3
        total_h10 += h10
    # average for all queries of a type
    avg_mrr = total_mrr / num_queries
    avg_h1 = total_h1 / num_queries
    avg_h3 = total_h3 / num_queries
    avg_h10 = total_h10 / num_queries

    return avg_mrr, avg_h1, avg_h3, avg_h10


query_name_dict = {
    ("e", ("r",)): "1p",
    ("e", ("r", "r")): "2p",
    ("e", ("r", "r", "r",),): "3p",
    (("e", ("r",)), ("e", ("r",))): "2i",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
    ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
    (("e", ("r", "r")), ("e", ("r",))): "pi",
    # negation
    (("e", ("r",)), ("e", ("r", "n"))): "2in",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
    ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
    (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
    (("e", ("r", "r", "n")), ("e", ("r",))): "pni",

    # union
    (("e", ("r",)), ("e", ("r",)), ("u",)): "2u",
    ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up",

}
# @ TODO please add all other types here
# 1p doesn't work
# 2p works
# 3p doesn'twork
#
for query_type in ["2up"]:
    query_structs_and_queries, easy_answers, false_positives, hard_answers = q.get_queries(query_type=query_type,
                                                                                           gen_num=10)
    # Iterate over query types
    for query_structure, queries in query_structs_and_queries.items():
        entity_scores = dict()
        for q in queries:
            entity_scores[q] = pre_trained_kge.answer_multi_hop_query(query_type=query_type, query=q, only_scores=True)
        mrr, h1, h3, h10 = evaluate(pre_trained_kge, entity_scores, easy_answers, hard_answers)
        print(mrr)
