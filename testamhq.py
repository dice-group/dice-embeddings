from dicee import QueryGenerator
from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
import os
import numpy as np
import pickle
import torch
from dicee.static_funcs import evaluate
# (1) Train Clifford Embeddings model with AllvsAll on Family dataset or load the pre-trained KGE model
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
    pre_trained_kge = KGE(path="/Users/sourabh/dice-embeddings/Experiments/2023-08-06 01-53-29.233645")
# (3) Generate queries of a particular type depending on the flag
qg = QueryGenerator(datapath="KGs/UMLS", seed=0, gen_train=False, gen_valid=False, gen_test=True)
# Generate queries for the following type and asnwer the queries
query_names = ['1p','2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']
for query_type in query_names:
    # Generates n queries with answers
    query_structs_and_queries, easy_answers, false_positives, hard_answers = qg.get_queries(query_type=query_type,
                                                                                           gen_num=10)
    # Iterate over query types
    for query_structure, queries in query_structs_and_queries.items():
        entity_scores = dict()

        for q in queries:
            # Get scores for all entities
            entity_scores[q] = pre_trained_kge.answer_multi_hop_query(query_type=query_type, query=q, only_scores=True)
        # Evaluation on hard answers using the filtered setting
        mrr, h1, h3, h10 = evaluate(pre_trained_kge, entity_scores, easy_answers, hard_answers)
        print(query_type, ":", mrr)
