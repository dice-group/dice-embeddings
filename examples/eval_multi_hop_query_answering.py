""" Multi-hop Query answering via neural link predictors

Here, we show how to
(1) Train a neural link predictor is trained on a single hop queries and
(2) Answer multi-hop queries with (1).

Structure of this example as follows
pip install dicee

(1) Train a neural link predictor and save it into disk
(2) Load (1) from disk into memory
(3) Generate multi-hop queries
(4) Answer multi-hop queries
(5) Report results
"""
from dicee import QueryGenerator
from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
import pandas as pd

from dicee.static_funcs import evaluate
from dicee.static_funcs import load_pickle
from dicee.static_funcs import load_json
import argparse
import os
from typing import Generator, List, Tuple


def initialize_query_generator(args):
    return QueryGenerator(train_path=f"{args.dataset_dir}/train.txt",
                          val_path=f"{args.dataset_dir}/valid.txt",
                          test_path=f"{args.dataset_dir}/test.txt",
                          seed=args.random_seed)


def initialize_kge_models(args) -> Generator:
    # Detect all folders under args.experiment_dir
    if args.experiment_dir is not None:
        sub_folder_str_paths = os.listdir(args.experiment_dir)
        for path in sub_folder_str_paths:
            if path == "summary.csv":
                continue
            # Return also the config file

            full_path = args.experiment_dir + "/" + path

            yield KGE(path=args.experiment_dir + "/" + path), load_json(f'{full_path}/configuration.json')
    elif args.pretrained_model is not None:
        yield KGE(args.pretrained_model), load_json(f'{args.pretrained_model}/configuration.json')


def answer_multi_hop_queries(pre_trained_kge, queries_and_answers) -> pd.DataFrame:
    # (4) Answer multi-hop queries with a neural link predictor
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
    return pd.DataFrame(results_kge, columns=["Query", "Number", "MRR", "H1", "H3", "H10"])


def eval_multi_hop_query_answering(args) -> Generator[Tuple[str, pd.DataFrame], None, None]:
    # Load the data
    qg = initialize_query_generator(args)
    # Load KGE models
    queries_and_answers = None
    for kge_model, config in initialize_kge_models(args):
        qg.ent2id = kge_model.entity_to_idx
        qg.rel2id = kge_model.relation_to_idx
        if args.query_and_answers is None and queries_and_answers is None:
            # To generate Queries Queries and Answers

            queries_and_answers = [(q, qg.get_queries(query_type=q, gen_num=args.num_queries)) for q in
                                   [  # "1p",  E? . r(e, E?)
                                       "2p",
                                       # E? . ∃E1 : r1(e, E1) ∧ r2(E1, E?) $E_?\:.\:\exists E_1:r_1(e,E_1)\land r_2(E_1, E_?)$
                                       "3p",  # $E_?\:.\:\exists E_1E_2.r_1(e,E_1)\land r_2(E_1, E_2)\land r_3(E_2,E_?)$
                                       "2i",
                                       "3i",  # $E_?\:.\:r_1(e_1,E_?)\land r_2(e_2,E_?)\land r_3(e_3,E_?)$
                                       "ip",
                                       "pi",
                                       "2u",
                                       "up"
                                   ]]
            # qg.save_queries_and_answers(path="Queries", data=queries_and_answers)
            # queries_saved = True
        else:
            """"""
            assert queries_and_answers is not None
            # queries_and_answers = qg.load_queries_and_answers(path=args.query_and_answers)

        df = answer_multi_hop_queries(kge_model, queries_and_answers)
        yield kge_model.name, config, df


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, default="KGs/UMLS")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="A path of a family directory containing pre-trained model directories")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="A path of a single pre-trained model directory")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--query_and_answers", type=str, default=None)
    return parser.parse_args()


def display(results):
    for (kg_name, config, df) in results:
        print(f"{kg_name}\tAdaptive SWA:{config['adaptive_swa']}\t SWA:{config['swa']}")
        print(df)
        print("#")

        print(df["MRR"].to_latex(index=False))


if __name__ == '__main__':
    display(eval_multi_hop_query_answering(get_default_arguments()))
