from dicee import QueryGenerator
from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
import pandas as pd

from dicee.static_funcs import evaluate
from dicee.static_funcs import load_pickle
from dicee.static_funcs import load_json

data_frames_results = []

queries_saved = False
for kge_name in ["DistMult", "ComplEx", "Keci", "Pykeen_QuatE", "Pykeen_MuRE"]:
    # (1) Train
    args = Namespace()
    args.model = kge_name
    args.scoring_technique = "KvsAll"
    args.path_dataset_folder = "KGs/UMLS"
    args.num_epochs = 20
    args.batch_size = 1024
    args.lr = 0.1
    args.embedding_dim = 128
    reports = Execute(args).start()
    # (2) Load the pretrained model
    pre_trained_kge = KGE(path=reports['path_experiment_folder'])
    configs = load_json(reports['path_experiment_folder'] + '/report.json')
    num_param = configs["NumParam"]
    runtime = configs["Runtime"]

    # (3) Generate queries of a particular type depending on the flag
    qg = QueryGenerator(
        train_path="KGs/UMLS/train.txt",
        val_path="KGs/UMLS/valid.txt",
        test_path="KGs/UMLS/test.txt",
        ent2id=pre_trained_kge.entity_to_idx,
        rel2id=pre_trained_kge.relation_to_idx, seed=1)
    # Generate queries for the following type and answer the queries

    if queries_saved is False:
        # To generate Queries Queries and Answers
        queries_and_answers = [(q, qg.get_queries(query_type=q, gen_num=100)) for q in
                               ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u',
                                'up']]
        qg.save_queries_and_answers(path="Queries", data=queries_and_answers)
        queries_saved = True
    # Load saved queries and answers
    queries_and_answers = qg.load_queries_and_answers(path="Queries")

    results_kge = []
    for query_type, (query_structs_and_queries, easy_answers, false_positives, hard_answers) in queries_and_answers:
        for _, queries in query_structs_and_queries.items():
            # Compute query scores for all entities given queries
            entity_scores_for_each_query = pre_trained_kge.answer_multi_hop_query(query_type=query_type,
                                                                                  queries=queries,
                                                                                  only_scores=True)
            # Map a query to query scores of all entities.
            entity_scores = {q: scores for scores, q in zip(entity_scores_for_each_query, queries)}
            # Compute scores
            mrr, h1, h3, h10 = evaluate(pre_trained_kge.entity_to_idx, entity_scores, easy_answers, hard_answers)
            # Store evaluation
            results_kge.append([query_type, len(queries), mrr, h1, h3, h10])
    df = pd.DataFrame(results_kge, columns=["Query", "Size", "MRR", "H1", "H3", "H10"])
    data_frames_results.append((kge_name, num_param, runtime, df))

for kge_name, num_param, runtime, df in data_frames_results:
    print(kge_name, num_param, runtime)
    print(df[["MRR", "H1", "H3", "H10"]].mean())
