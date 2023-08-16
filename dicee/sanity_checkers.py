import os
import glob
import requests


def is_sparql_endpoint_alive(sparql_endpoint: str = None):
    if sparql_endpoint:
        query = """SELECT (COUNT(*) as ?num_triples) WHERE {  ?s ?p ?o .} """
        response = requests.post(sparql_endpoint, data={'query': query})
        assert response.ok
        print('SPARQL connection is successful')
        return response.ok
    else:
        return False


def sanity_checking_with_arguments(args):
    try:
        assert args.embedding_dim > 0
    except AssertionError:
        print(f'embedding_dim must be strictly positive. Currently:{args.embedding_dim}')
        raise

    if args.scoring_technique not in ['KvsSample', 'KvsAll', 'NegSample', '1vsAll', 'Pyke']:
        raise KeyError(f'Invalid training strategy => {args.scoring_technique}.')

    assert args.learning_rate > 0
    if args.num_folds_for_cv is None:
        args.num_folds_for_cv = 0
    try:
        assert args.num_folds_for_cv >= 0
    except AssertionError:
        print(f'num_folds_for_cv can not be negative. Currently:{args.num_folds_for_cv}')
        raise
    # (1) Check
    if is_sparql_endpoint_alive(args.sparql_endpoint):
        try:
            assert args.path_dataset_folder is None and args.path_single_kg is None
        except AssertionError:
            raise RuntimeWarning(f'The path_dataset_folder and path_single_kg arguments '
                                 f'must be None if sparql_endpoint is given.'
                                 f'***{args.path_dataset_folder}***\n'
                                 f'***{args.path_single_kg}***\n'
                                 f'These two parameters are set to None.')
        args.path_dataset_folder = None
        args.path_single_kg = None
    elif args.path_dataset_folder is not None:
        try:
            assert isinstance(args.path_dataset_folder, str)
        except AssertionError:
            raise AssertionError(f'The path_dataset_folder must be string sparql_endpoint is not given.'
                                 f'***{args.path_dataset_folder}***')
        try:
            assert os.path.isdir(args.path_dataset_folder) or os.path.isfile(args.path_dataset_folder)
        except AssertionError:
            raise AssertionError(f'The path_dataset_folder does not lead to a directory '
                                 f'***{args.path_dataset_folder}***')
        # Check whether the input parameter leads a standard data format (e.g. FOLDER/train.txt)
        # or a data in the parquet format
        # @TODO: Rethink about this computation.
        if '.parquet' == args.path_dataset_folder[-8:]:
            """ all is good we have xxx.parquet data"""
        elif glob.glob(args.path_dataset_folder + '/train*'):
            """ all is good we have xxx/train.txt"""
        else:
            raise ValueError(
                f'Data format is not recognized.'
                f'\nThe path_dataset_folder parameter **{args.path_dataset_folder}** must lead to'
                f'(a) **folder/train.txt** or *** triples stored in the parquet format')
    elif args.path_single_kg is not None:
        assert args.path_dataset_folder is None
    elif args.path_dataset_folder is None and args.path_single_kg is None and args.sparql_endpoint is None:
        raise RuntimeError(f" Following arguments cannot be all None:"
                           f"path_dataset_folder:{args.path_dataset_folder},\t"
                           f"path_single_kg:{args.path_single_kg},\t"
                           f"sparql_endpoint:{args.sparql_endpoint}.")
    else:
        raise RuntimeError('Invalid computation flow!')


def config_kge_sanity_checking(args, dataset):
    """
    Sanity checking for input hyperparams.
    :return:
    """
    assert isinstance(args.batch_size, int) or args.batch_size is None
    if args.model == 'Shallom' and args.scoring_technique == 'NegSample':
        print(
            'Shallom can not be trained with Negative Sampling. Scoring technique is changed to KvsALL')
        args.scoring_technique = 'KvsAll'

    if args.scoring_technique == 'KvsAll':
        args.neg_ratio = None
    return args, dataset
