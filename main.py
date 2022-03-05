from core.executer import Execute
import argparse
import pytorch_lightning as pl


def argparse_default(description=None):
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default Trainer param https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods

    # Dataset and storage related
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS',
                        help="The path of a folder containing input data")
    parser.add_argument("--large_kg_parse", type=int, default=0, help='A flag for using all cores at parsing.')
    parser.add_argument("--storage_path", type=str, default='DAIKIRI_Storage',
                        help="Embeddings, model, and any other related data will be stored therein.")
    parser.add_argument("--read_only_few", type=int, default=None, help='READ only first N triples. If 0, read all.')
    parser.add_argument("--sample_triples_ratio", type=float, default=None, help='Sample input data.')
    parser.add_argument("--seed_for_computation", type=int, default=1, help='Seed for all, see pl seed_everything().')

    # Models.
    parser.add_argument("--model", type=str,
                        default='QMult',
                        help="Available models: ConEx, ConvQ, ConvO,  QMult, OMult, "
                             "Shallom, ConEx, ComplEx, DistMult, KronE, KPDistMult")
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=10, help='Number of epochs for training. '
                                                                   'max and min_epochs of pl.Trainer disabled')
    parser.add_argument("--save_model_at_every_epoch", type=int, default=None,
                        help='At every X number of epochs model will be saved.')

    parser.add_argument('--batch_size', type=int, default=1024, help='Mini batch size')
    parser.add_argument("--lr", type=float, default=0.01, help='Learning rate')
    parser.add_argument("--label_smoothing_rate", type=float, default=None, help='None for not using it.')
    parser.add_argument("--label_relaxation_rate", type=float, default=None, help='None for not using it.')
    parser.add_argument("--add_noise_rate", type=float, default=None, help='None for not using it. '
                                                                           '.1 means extend train data by adding 10% random data')

    # Model Parameters
    # Hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=25,
                        help='Number of dimensions for an embedding vector. This parameter is used for those models requiring same number of embedding vector for entities and relations.')
    parser.add_argument('--entity_embedding_dim', type=int, default=8,
                        help='Number of dimensions for an entity embedding vector. '
                             'This parameter is used for those model having flexibility of using different sized entity and relation embeddings.')
    parser.add_argument('--rel_embedding_dim', type=int, default=8,
                        help='Number of dimensions for an entity embedding vector. '
                             'This parameter is used for those model having flexibility of using different sized entity and relation embeddings.')
    # Model Related
    parser.add_argument('--input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.0)
    parser.add_argument("--feature_map_dropout_rate", type=int, default=0.0)
    parser.add_argument('--apply_unit_norm', type=bool, default=False)
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=3, help="# of output channels in convolution")
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.5,
                        help='The ratio of the size of the affine transformation w.r.t. the size of the embeddings')

    # Flags for computation
    parser.add_argument("--eval", type=int, default=1,
                        help='A flag for using evaluation')
    parser.add_argument("--eval_on_train", type=int, default=1,
                        help='A flag for using train data to evaluation ')

    # Hyperparameters for training.
    parser.add_argument('--scoring_technique', default='KvsAll', help="1vsAll, KvsAll, NegSample.")
    parser.add_argument('--neg_ratio', type=int, default=0)
    # Data Augmentation.
    parser.add_argument('--num_folds_for_cv', type=int, default=0, help='Number of folds in k-fold cross validation.'
                                                                        'If >2 ,no evaluation scenario is applied implies no evaluation.')
    # This is a workaround for read
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    Execute(argparse_default()).start()
