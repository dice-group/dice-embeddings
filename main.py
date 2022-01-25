from executer import Execute
import pytorch_lightning as pl
import argparse


def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default parameters of Trainer
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods
    # Number of workers for data loader
    # https: // pytorch - lightning.readthedocs.io / en / latest / guides / speed.html  # num-workers
    # Dataset and storage related
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/Family')
    parser.add_argument("--storage_path", type=str, default='DAIKIRI_Storage')
    parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')
    parser.add_argument("--read_only_few", type=int, default=0, help='READ only first N triples. If 0, read all.')
    # Models.
    parser.add_argument("--model", type=str, default='QMult',
                        help="Available models: ConEx, ConvQ, ConvO,  QMult, OMult, Shallom, ConEx, ComplEx, DistMult")
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=500, help='Number of epochs for training. '
                                                                    'This disables max_epochs and min_epochs of pl.Trainer')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.1)
    # Model Parameters
    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=8, help="# of output channels in convolution")
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.5,
                        help='The ratio of the size of the affine transformation w.r.t. the size of the embeddings')
    # Flags for computation
    parser.add_argument("--large_kg_parse", type=int, default=0, help='A flag for using all cores at parsing.')
    parser.add_argument("--eval", type=int, default=0,
                        help='A flag for using evaluation. If 0, memory consumption is decreased')
    # Do we use still use it ?
    parser.add_argument("--continue_training", type=int, default=1, help='A flag for continues training')
    # Hyperparameters pertaining to regularization.
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.1)
    parser.add_argument("--feature_map_dropout_rate", type=int, default=.3)
    parser.add_argument('--apply_unit_norm', type=bool, default=False)
    # Hyperparameters for training.
    parser.add_argument('--scoring_technique', default='KvsAll', help="KvsAll technique or NegSample.")
    parser.add_argument('--negative_sample_ratio', type=int, default=1)
    # Data Augmentation.
    parser.add_argument("--add_reciprical", type=bool, default=False)
    parser.add_argument('--num_folds_for_cv', type=int, default=0, help='Number of folds in k-fold cross validation.'
                                                                        'If >2,no evaluation scenario is applied implies no evaluation.')

    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)


if __name__ == '__main__':
    exc = Execute(argparse_default())
    exc.start()
