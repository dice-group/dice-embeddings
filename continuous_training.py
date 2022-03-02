from core.executer import Execute
from core import load_configuration
from collections import namedtuple
import argparse
import dask.dataframe as dd
import os
import json


class ContinuousExecute(Execute):
    def __init__(self, args):
        assert os.path.exists(args.path_experiment_folder)
        assert os.path.isfile(args.path_experiment_folder + '/idx_train_df.gzip')
        assert os.path.isfile(args.path_experiment_folder + '/configuration.json')
        previous_args = load_configuration(args.path_experiment_folder + '/configuration.json')
        previous_args.update(vars(args))
        super().__init__(previous_args, continuous_training=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    # Dataset and storage related
    parser.add_argument("--path_experiment_folder", type=str, default="DAIKIRI_Storage/2022-03-02 15:21:32.026941",
                        help="The path of a folder containing pretrained model")
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=10,
                        help='Number of epochs for training. Overwrite previous ep'
                             'This disables max_epochs and min_epochs of pl.Trainer')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.1)

    ContinuousExecute(parser.parse_args()).start()
