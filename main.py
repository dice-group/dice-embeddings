from executer import Execute
import pytorch_lightning as pl
import argparse


def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS')
    parser.add_argument("--storage_path", type=str, default='DAIKIRI_Storage')
    parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')

    # Flags for computation
    parser.add_argument("--large_kg_parse", type=int, default=1, help='A flag for using all cores at parsing.')
    parser.add_argument("--eval", type=int, default=0, help='A flag for using evaluation. If 0, memory consumption is decreased')
    parser.add_argument("--continue_training", type=int, default=1, help='A flag for continues training')
    parser.add_argument("--read_only_few", type=int, default=0, help='READ only first N triples')

    # Models.
    parser.add_argument("--model", type=str, default='QMult',
                        help="Available models: ConEx, ConvQ, ConvO,  QMult, OMult, Shallom, ConEx, ComplEx, DistMult")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=10)
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=8, help="# of output channels in convolution")
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.5,
                        help='The ratio of the size of the affine transformation w.r.t. the size of the embeddings')

    # Hyperparameters pertaining to regularization.
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.1)
    parser.add_argument("--feature_map_dropout_rate", type=int, default=.3)
    parser.add_argument('--apply_unit_norm', type=bool, default=False)

    # Hyperparameters for training.
    parser.add_argument("--max_num_epochs", type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--scoring_technique', default='NegSample', help="KvsAll technique or NegSample.")
    parser.add_argument('--negative_sample_ratio', type=int, default=1)

    parser.add_argument('--learning_rate', default=.01)

    parser.add_argument("--check_val_every_n_epochs", type=int, default=10)

    # Data Augmentation.
    parser.add_argument("--add_reciprical", type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')
    parser.add_argument('--kvsall', default=False)
    parser.add_argument('--num_folds_for_cv', type=int, default=0, help='Number of folds in k-fold cross validation.'
                                                                        'If >2,no evaluation scenario is applied implies no evaluation.')
    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)


if __name__ == '__main__':
    exc = Execute(argparse_default())
    exc.start()
