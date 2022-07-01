from core.executer import ContinuousExecute
import argparse
import dask.dataframe as dd
import os
import json
from types import SimpleNamespace


def argparse_default(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    # Dataset and storage related
    parser.add_argument("--path_experiment_folder", type=str, default="Experiments/2022-07-01 19:22:13.691723",
                        help="The path of a folder containing pretrained model")
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=10,
                        help='Number of epochs for training. Overwrite previous ep'
                             'This disables max_epochs and min_epochs of pl.Trainer')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed_for_computation", type=int, default=1, help='Seed for all, see pl seed_everything().')
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    args = argparse_default()
    ContinuousExecute(args).start()
