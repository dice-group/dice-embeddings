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


def validate_knowledge_graph(args):
    """ Validating the source of knowledge graph """
    # (1) Validate SPARQL endpoint
    if is_sparql_endpoint_alive(args.sparql_endpoint):
        try:
            assert args.dataset_dir is None and args.path_single_kg is None
        except AssertionError:
            raise RuntimeWarning(f'The dataset_dir and path_single_kg arguments '
                                 f'must be None if sparql_endpoint is given.'
                                 f'***{args.dataset_dir}***\n'
                                 f'***{args.path_single_kg}***\n'
                                 f'These two parameters are set to None.')
        # Set None.
        args.dataset_dir = None
        args.path_single_kg = None

    elif args.path_single_kg is not None:
        if args.sparql_endpoint is not None or args.path_single_kg is not None:
            print(f'The dataset_dir and sparql_endpoint arguments '
                  f'must be None if path_single_kg is given.'
                  f'***{args.dataset_dir}***\n'
                  f'***{args.sparql_endpoint}***\n'
                  f'These two parameters are set to None.')
            args.dataset_dir = None
            args.sparql_endpoint = None

    elif args.dataset_dir:
        try:
            assert isinstance(args.dataset_dir, str)
        except AssertionError:
            raise AssertionError(f'The dataset_dir must be string sparql_endpoint is not given.'
                                 f'***{args.dataset_dir}***')
        try:
            assert os.path.isdir(args.dataset_dir) or os.path.isfile(args.dataset_dir)
        except AssertionError:
            raise AssertionError(f'The dataset_dir does not lead to a directory '
                                 f'***{args.dataset_dir}***')
        # Check whether the input parameter leads a standard data format (e.g. FOLDER/train.txt)
        if glob.glob(args.dataset_dir + '/train*'):
            """ all is good we have xxx/train.txt"""
        else:
            raise ValueError(
                f"---dataset_dir **{args.dataset_dir}** must lead to "
                f"**folder** containing at least train.txt**. "
                f"Use --path_single_kg **folder/dataset.format**, if you have a single file.")

        if args.sparql_endpoint is not None or args.path_single_kg is not None:
            print(f'The sparql_endpoint and path_single_kg arguments '
                  f'must be None if dataset_dir is given.'
                  f'***{args.sparql_endpoint}***\n'
                  f'***{args.path_single_kg}***\n'
                  f'These two parameters are set to None.')
            args.sparql_endpoint = None
            args.path_single_kg = None


    elif args.dataset_dir is None and args.path_single_kg is None and args.sparql_endpoint is None:
        raise RuntimeError(f"One of the following arguments must be given: "
                           f"--dataset_dir:{args.dataset_dir},\t"
                           f"--path_single_kg:{args.path_single_kg},\t"
                           f"--sparql_endpoint:{args.sparql_endpoint}.")
    else:
        raise RuntimeError('Invalid computation flow!')


def sanity_checking_with_arguments(args):
    try:
        assert args.embedding_dim > 0
    except AssertionError:
        raise AssertionError(f'embedding_dim must be strictly positive. Currently:{args.embedding_dim}')

    if args.scoring_technique not in ["AllvsAll", "KvsSample", "KvsAll", "NegSample", "1vsAll", "Pyke", "Sentence"]:
        raise KeyError(f'Invalid training strategy => {args.scoring_technique}.')

    assert args.learning_rate > 0
    if args.num_folds_for_cv is None:
        args.num_folds_for_cv = 0
    try:
        assert args.num_folds_for_cv >= 0
    except AssertionError:
        raise AssertionError(f'num_folds_for_cv can not be negative. Currently:{args.num_folds_for_cv}')
    validate_knowledge_graph(args)

