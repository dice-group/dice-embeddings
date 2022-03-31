from core.executer import ContinuousExecute
from core import load_json
import argparse
import dask.dataframe as dd
import os
import json
from types import SimpleNamespace


def argparse_default(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    # Dataset and storage related
    parser.add_argument("--path_experiment_folder", type=str, default="Experiments/2022-03-24 07:51:51.865808",
                        help="The path of a folder containing pretrained model")
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=1,
                        help='Number of epochs for training. Overwrite previous ep'
                             'This disables max_epochs and min_epochs of pl.Trainer')
    parser.add_argument("--lr", type=float, default=0.01)
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    args = argparse_default()
    # @TODO: Not yet implemented
    args.eval = 0
    args.eval_on_train = 0
    ContinuousExecute(args).start()
