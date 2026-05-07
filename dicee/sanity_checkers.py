import glob
import os

import requests
import torch


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
            #print(f'The dataset_dir and sparql_endpoint arguments '
            #      f'must be None if path_single_kg is given.'
            #      f'***{args.dataset_dir}***\n'
            #      f'***{args.sparql_endpoint}***\n'
            #      f'These two parameters are set to None.')
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
            raise FileNotFoundError(
                f"Dataset directory not found: {args.dataset_dir}\n"
                f"\nSuggestions:\n"
                f"  1. Download datasets:\n"
                f"     wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip --no-check-certificate\n"
                f"     unzip KGs.zip\n"
                f"  2. Use absolute path: --dataset_dir /absolute/path/to/KGs/UMLS\n"
                f"  3. Check current directory: {os.getcwd()}\n"
            )
        # Check whether the input parameter leads a standard data format (e.g. FOLDER/train.txt)
        if glob.glob(args.dataset_dir + '/train*'):
            """ all is good we have xxx/train.txt"""
        else:
            raise ValueError(
                f"Dataset directory must contain train.txt file: {args.dataset_dir}\n"
                f"\nExpected structure:\n"
                f"  {args.dataset_dir}/\n"
                f"    ├── train.txt (required)\n"
                f"    ├── valid.txt (optional)\n"
                f"    └── test.txt (optional)\n"
                f"\nFor single file datasets, use: --path_single_kg folder/dataset.owl\n"
                f"For SPARQL endpoints, use: --sparql_endpoint http://localhost:3030/dataset/\n"
            )

        if args.sparql_endpoint is not None or args.path_single_kg is not None:
            #print(f'The sparql_endpoint and path_single_kg arguments '
            #      f'must be None if dataset_dir is given.'
            #      f'***{args.sparql_endpoint}***\n'
            #      f'***{args.path_single_kg}***\n'
            #      f'These two parameters are set to None.')
            args.sparql_endpoint = None
            args.path_single_kg = None


    elif args.dataset_dir is None and args.path_single_kg is None and args.sparql_endpoint is None:
        raise ValueError(
            "No data source specified. You must provide ONE of the following:\n"
            "\nOption 1: Standard dataset folder\n"
            "  --dataset_dir KGs/UMLS\n"
            "\nOption 2: Single RDF/OWL file\n"
            "  --path_single_kg KGs/Family/family.owl --backend rdflib\n"
            "\nOption 3: SPARQL endpoint\n"
            "  --sparql_endpoint http://localhost:3030/mydata/\n"
            "\nFor examples, see: tests/test_different_backends.py\n"
        )
    else:
        raise RuntimeError('Invalid computation flow!')


def sanity_checking_with_arguments(args):
    assert args.embedding_dim > 0, f"embedding_dim must be strictly positive. Currently:{args.embedding_dim}"
    valid_techniques = ["AllvsAll", "1vsSample", "KvsSample", "KvsAll", "FixedNegSample", "NegSample", "1vsAll", "Pyke", "Sentence"]
    assert args.scoring_technique in valid_techniques, f"Invalid training strategy => {args.scoring_technique}."
    assert args.learning_rate > 0, f"Learning rate must be greater than 0. Currently:{args.learning_rate}"
    if args.num_folds_for_cv is None:
        args.num_folds_for_cv = 0
    assert args.num_folds_for_cv >= 0, f"num_folds_for_cv can not be negative. Currently:{args.num_folds_for_cv}"
    validate_knowledge_graph(args)

def sanity_check_callback_args(args):
    """
    Perform sanity checks on callback-related arguments.
    """
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Check if any callbacks are requested

    if (args.trainer == "PL" and gpu_count >= 2) or args.trainer == "torchDDP":
        if args.path_to_store_single_run is None:
            raise NotImplementedError("Path to store experiments must be provided for Multi-GPU training.")
        if args.adaptive_lr:
            raise NotImplementedError("Adaptive learning rate is not supported with Multi-GPU training.")
    has_callbacks = any([args.swa, args.swag, args.ema, args.adaptive_swa, args.twa, args.adaptive_lr, args.eval_every_n_epochs > 0,
                          args.eval_at_epochs is not None])
    if not has_callbacks:
        return  # No callbacks, no checks needed

    # SWA-related checks
    if any([args.swa, args.swag, args.ema, args.twa]):
        args.swa_start_epoch = args.swa_start_epoch or 1
        assert args.swa_start_epoch > 0, "SWA Start Epoch must be greater than 0"

    # TWA/SWAG trainer compatibility
    if any([args.twa, args.swag]) and args.trainer in {"torchDDP"}:
        raise NotImplementedError("TWA and SWAG are not supported with torchDDP trainer.")
