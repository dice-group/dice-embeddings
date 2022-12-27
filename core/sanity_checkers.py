import os
import numpy as np
import glob


def sanity_checking_with_arguments(args):
    try:
        assert args.embedding_dim > 0
    except AssertionError:
        print(f'embedding_dim must be strictly positive. Currently:{args.embedding_dim}')
        raise

    if not (args.scoring_technique in ['KvsSample', 'CCvsAll', 'PvsAll', 'KvsAll', 'NegSample', '1vsAll',
                                       'BatchRelaxedKvsAll',
                                       'BatchRelaxed1vsAll']):
        raise KeyError(f'Invalid training strategy => {args.scoring_technique}.')

    assert args.learning_rate > 0
    if args.num_folds_for_cv is None:
        args.num_folds_for_cv = 0
    try:
        assert args.num_folds_for_cv >= 0
    except AssertionError:
        print(f'num_folds_for_cv can not be negative. Currently:{args.num_folds_for_cv}')
        raise
    # Check whether is a directory or a file?
    try:
        assert os.path.isdir(args.path_dataset_folder) or os.path.isfile(args.path_dataset_folder)
    except AssertionError:
        raise AssertionError(f'The path_dataset_folder does not lead to a directory ***{args.path_dataset_folder}***')
    # Check whether the input parameter leads a standard data format (e.g. FOLDER/train.txt) or a data in the parquet format
    if '.parquet' == args.path_dataset_folder[-8:]:
        """ all is good we have xxx.parquet data"""
    elif glob.glob(args.path_dataset_folder + '/train*'):
        """ all is good we have xxx/train.txt"""
    else:
        raise ValueError(
            f'Data format is not recognized.\nThe path_dataset_folder parameter **{args.path_dataset_folder}** must lead to (a) **folder/train.txt** or *** triples stored in the parquet format')
    #assert isinstance(args.eval, bool)


def config_kge_sanity_checking(args, dataset):
    """
    Sanity checking for input hyperparams.
    :return:
    """
    if args.batch_size > len(dataset.train_set):
        args.batch_size = len(dataset.train_set)
    if args.model == 'Shallom' and args.scoring_technique == 'NegSample':
        print(
            'Shallom can not be trained with Negative Sampling. Scoring technique is changed to KvsALL')
        args.scoring_technique = 'KvsAll'

    if args.scoring_technique == 'KvsAll':
        args.neg_ratio = None
    return args, dataset
