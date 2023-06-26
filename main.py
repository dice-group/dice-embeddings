from dicee.executer import Execute
import pytorch_lightning as pl
from dicee.config import ParseDict
import argparse

def get_default_arguments(description=None):
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default Trainer param https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS',
                        help="The path of a folder containing input data")
    parser.add_argument("--save_embeddings_as_csv", type=bool, default=False,
                        help='A flag for saving embeddings in csv file.')
    parser.add_argument("--storage_path", type=str, default='Experiments',
                        help="Embeddings, model, and any other related data will be stored therein.")
    parser.add_argument("--model", type=str,
                        default="AConvQ",
                        help="Available models: ConEx, AConEx, ConvQ, AConQ, ConvO, AConvO,"
                             "QMult, OMult, Shallom, DistMult, TransE, ComplEx, Keci")
    parser.add_argument('--optim', type=str, default='Adam',
                        help='[Adam, SGD]')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Number of dimensions for an embedding vector. ')
    parser.add_argument("--num_epochs", type=int, default=50, help='Number of epochs for training. ')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini batch size')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--callbacks', '--list', nargs='+', default=[],
                        help='List of tuples representing a callback and values, e.g. [FPPE or PPE or PPE10 ,PPE20 or PPE, FPPE]')
    parser.add_argument("--backend", type=str, default='pandas',
                        help='Select [polars(seperator: \t), modin(seperator: \s+), pandas(seperator: \s+)]')
    parser.add_argument("--trainer", type=str, default='torchCPUTrainer',
                        help='PL (pytorch lightning trainer), torchDDP (custom ddp), torchCPUTrainer (custom cpu only)')
    parser.add_argument('--scoring_technique', default='KvsAll', help="KvsSample, 1vsAll, KvsAll, NegSample")
    parser.add_argument('--neg_ratio', type=int, default=0,
                        help='The number of negative triples generated per positive triple.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 penalty e.g.(0.00001)')
    parser.add_argument('--input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.0)
    parser.add_argument("--feature_map_dropout_rate", type=float, default=0.0)
    parser.add_argument("--normalization", type=str, default="None", help="[LayerNorm, BatchNorm1d, None]")
    parser.add_argument("--init_param", type=str, default='xavier_normal', help="[xavier_normal, None]")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0,
                        help="e.g. gradient_accumulation_steps=2 implies that gradients are accumulated at every second mini-batch")
    parser.add_argument('--num_folds_for_cv', type=int, default=0,
                        help='Number of folds in k-fold cross validation.'
                             'If >2 ,no evaluation scenario is applied implies no evaluation.')
    parser.add_argument("--eval_model", type=str, default="train_val_test",
                        help='test the link prediction results on the splits, e.g. '
                             'train_val,train_val_test, val_test, val_test_constraint ')
    parser.add_argument("--save_model_at_every_epoch", type=int, default=None,
                        help='At every X number of epochs model will be saved. If None, we save 4 times.')
    parser.add_argument("--label_smoothing_rate", type=float, default=0.0, help='None for not using it.')
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for convolution based models.")
    parser.add_argument("--num_of_output_channels", type=int, default=2,
                        help="# of output channels in convolution")
    parser.add_argument("--num_core", type=int, default=0,
                        help='Number of cores to be used. 0 implies using single CPU')
    parser.add_argument("--seed_for_computation", type=int, default=0,
                        help='Seed for all, see pl seed_everything().')
    parser.add_argument("--sample_triples_ratio", type=float, default=None, help='Sample input data.')
    parser.add_argument("--read_only_few", type=int, default=None,
                        help='READ only first N triples. If 0, read all.')
    parser.add_argument('--p', type=int, default=0,
                        help='P for Clifford Algebra')
    parser.add_argument('--q', type=int, default=0,
                        help='Q for Clifford Algebra')
    parser.add_argument('--auto_batch_finder', type=bool, default=False,
                        help='Find a batch size w.r.t. computational budgets')
    # @TODO: Temporary
    parser.add_argument("--pykeen_model_kwargs", nargs='*', action=ParseDict,
                        help='addtional paramters pass to pykeen_model')
    parser.add_argument("--use_SLCWALitModule", action="store_true",
                        help='whether to use SLCWALitModule in pykeen or not')
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    Execute(get_default_arguments()).start()
