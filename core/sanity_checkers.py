import os
import numpy as np
import glob


def sanity_checking_with_arguments(args):
    try:
        assert args.embedding_dim > 0
    except AssertionError:
        print(f'embedding_dim must be strictly positive. Currently:{args.embedding_dim}')
        raise

    if not (args.scoring_technique in ['KvsAll', 'NegSample', '1vsAll']):
        # print(f'Invalid training strategy => {args.scoring_technique}.')
        raise KeyError(f'Invalid training strategy => {args.scoring_technique}.')

    assert args.learning_rate > 0
    if args.num_folds_for_cv is None:
        args.num_folds_for_cv = 0
    try:
        assert args.num_folds_for_cv >= 0
    except AssertionError:
        print(f'num_folds_for_cv can not be negative. Currently:{args.num_folds_for_cv}')
        raise

    try:
        assert os.path.isdir(args.path_dataset_folder)
    except AssertionError:
        raise AssertionError(f'The path does not direct to a file {args.path_dataset_folder}')

    try:
        assert glob.glob(args.path_dataset_folder + '/train*')
    except AssertionError:
        print(f'The directory {args.path_dataset_folder} must contain a train.*  .')
        raise

    args.eval = bool(args.eval)
    args.large_kg_parse = bool(args.large_kg_parse)


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


def dataset_sanity_checking(train_set: np.ndarray, num_entities: int, num_relations: int) -> None:
    n, d = train_set.shape
    assert d == 3
    assert num_entities > max(train_set[:, 0]) and num_entities > max(train_set[:, 2])
    assert num_relations > max(train_set[:, 1])
    # 13. Sanity checking: data types
    assert isinstance(train_set[0], np.ndarray)
    assert isinstance(train_set[0][0], np.int64) and isinstance(train_set[0][1], np.int64)
    assert isinstance(train_set[0][2], np.int64)
