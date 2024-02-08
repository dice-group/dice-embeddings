""" Link Prediction Evaluation

Compute MRR and Hit@N scores on datasplits involving reciprical triples.
"""
from dicee.static_funcs import get_er_vocab, get_re_vocab, create_recipriocal_triples
from dicee.eval_static_funcs import evaluate_link_prediction_performance_with_reciprocals
import pandas as pd
from dicee import KGE
import argparse


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    return parser.parse_args()

def eval_lp(args):
    train_triples = create_recipriocal_triples(pd.read_csv(f"{args.dataset_dir}/train.txt",
                                                               sep="\s+",
                                                               header=None, usecols=[0, 1, 2],
                                                               names=['subject', 'relation', 'object'],
                                                               dtype=str)).values.tolist()
    valid_triples = create_recipriocal_triples(pd.read_csv(f"{args.dataset_dir}/valid.txt",
                                                               sep="\s+",
                                                               header=None, usecols=[0, 1, 2],
                                                               names=['subject', 'relation', 'object'],
                                                               dtype=str)).values.tolist()
    test_triples = create_recipriocal_triples(pd.read_csv(f"{args.dataset_dir}/test.txt",
                                                              sep="\s+",
                                                              header=None, usecols=[0, 1, 2],  
                                                              names=['subject', 'relation', 'object'],
                                                              dtype=str)).values.tolist()
    all_triples = train_triples + valid_triples + test_triples
    model = KGE(args.pretrained)
    print(model)
    print("Evaluating the link prediction performance on the test split...")
    print(evaluate_link_prediction_performance_with_reciprocals(model, triples=test_triples,er_vocab=get_er_vocab(all_triples)))


if __name__ == '__main__':
    eval_lp(get_default_arguments())
