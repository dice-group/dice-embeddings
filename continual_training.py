from core.executer import ContinuousExecute
import argparse
import dask.dataframe as dd
import os
import json
from types import SimpleNamespace


def argparse_default(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    # Dataset and storage related
    parser.add_argument("--path_experiment_folder", type=str, default="Experiments/2022-07-08 14:30:55.265088",
                        help="The path of a folder containing pretrained model")
    parser.add_argument("--num_epochs", type=int, default=None, help='Number of epochs for training.')
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_core", type=int, default=None, help='Number of cores to be used.')
    parser.add_argument('--scoring_technique', default=None, help="KvsSample, 1vsAll, KvsAll, NegSample")
    parser.add_argument('--neg_ratio', type=int, default=None,
                        help='The number of negative triples generated per positive triple.')
    parser.add_argument('--optim', type=str, default=None, help='[NAdam, Adam, SGD]')
    parser.add_argument('--batch_size', type=int, default=None, help='Mini batch size')
    parser.add_argument("--seed_for_computation", type=int, default=1, help='Seed for all, see pl seed_everything().')
    parser.add_argument("--torch_trainer", type=bool, default=True)
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    args = argparse_default()
    ContinuousExecute(args).start()
