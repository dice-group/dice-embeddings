"""Static utility functions for DICE embeddings.

This module provides utility functions for model initialization, data loading,
serialization, and various helper operations.
"""
import csv
import datetime
import functools
import glob
import json
import os
import pickle
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import polars as pl
import psutil
import requests
import torch

from .models import (
    AConEx, AConvO, AConvQ, CKeci, CoKE, ComplEx, ConEx, ConvO, ConvQ,
    DeCaL, DistMult, DualE, Keci, LFMult, OMult, Pyke, QMult, Shallom, TransE
)
from .models.base_model import BaseKGE
from .models.ensemble import EnsembleKGE
from .models.pykeen_models import PykeenKGE
from .models.transformers import BytE

# Model registry mapping model names to their classes and labelling types
MODEL_REGISTRY: Dict[str, Tuple[Type, str]] = {
    'Shallom': (Shallom, 'RelationPrediction'),
    'ConEx': (ConEx, 'EntityPrediction'),
    'AConEx': (AConEx, 'EntityPrediction'),
    'QMult': (QMult, 'EntityPrediction'),
    'OMult': (OMult, 'EntityPrediction'),
    'ConvQ': (ConvQ, 'EntityPrediction'),
    'AConvQ': (AConvQ, 'EntityPrediction'),
    'ConvO': (ConvO, 'EntityPrediction'),
    'AConvO': (AConvO, 'EntityPrediction'),
    'ComplEx': (ComplEx, 'EntityPrediction'),
    'DistMult': (DistMult, 'EntityPrediction'),
    'TransE': (TransE, 'EntityPrediction'),
    'Pyke': (Pyke, 'EntityPrediction'),
    'Keci': (Keci, 'EntityPrediction'),
    'CKeci': (CKeci, 'EntityPrediction'),
    'BytE': (BytE, 'EntityPrediction'),
    'LFMult': (LFMult, 'EntityPrediction'),
    'DeCaL': (DeCaL, 'EntityPrediction'),
    'DualE': (DualE, 'EntityPrediction'),
    'CoKE': (CoKE, 'EntityPrediction'),
}

def create_recipriocal_triples(df: pd.DataFrame) -> pd.DataFrame:
    """Add inverse triples to a DataFrame.

    For each triple (s, p, o), creates an inverse triple (o, p_inverse, s).

    Args:
        df: DataFrame with 'subject', 'relation', and 'object' columns.

    Returns:
        DataFrame with original and inverse triples concatenated.
    """
    inverse_df = pd.DataFrame({
        'subject': df['object'],
        'relation': df['relation'] + '_inverse',
        'object': df['subject']
    })
    return pd.concat([df, inverse_df], ignore_index=True)


def get_er_vocab(data: np.ndarray, file_path: Optional[str] = None) -> Dict[Tuple[int, int], List[int]]:
    """Build entity-relation to tail vocabulary.

    Args:
        data: Array of triples with shape (n, 3) where columns are (head, relation, tail).
        file_path: Optional path to save the vocabulary as pickle.

    Returns:
        Dictionary mapping (head, relation) pairs to list of tail entities.
    """
    er_vocab: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for h, r, t in data:
        er_vocab[(h, r)].append(t)
    if file_path:
        save_pickle(data=er_vocab, file_path=file_path)
    return er_vocab


def get_re_vocab(data: np.ndarray, file_path: Optional[str] = None) -> Dict[Tuple[int, int], List[int]]:
    """Build relation-entity (tail) to head vocabulary.

    Args:
        data: Array of triples with shape (n, 3) where columns are (head, relation, tail).
        file_path: Optional path to save the vocabulary as pickle.

    Returns:
        Dictionary mapping (relation, tail) pairs to list of head entities.
    """
    re_vocab: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for h, r, t in data:
        re_vocab[(r, t)].append(h)
    if file_path:
        save_pickle(data=re_vocab, file_path=file_path)
    return re_vocab


def get_ee_vocab(data: np.ndarray, file_path: Optional[str] = None) -> Dict[Tuple[int, int], List[int]]:
    """Build entity-entity to relation vocabulary.

    Args:
        data: Array of triples with shape (n, 3) where columns are (head, relation, tail).
        file_path: Optional path to save the vocabulary as pickle.

    Returns:
        Dictionary mapping (head, tail) pairs to list of relations.
    """
    ee_vocab: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for h, r, t in data:
        ee_vocab[(h, t)].append(r)
    if file_path:
        save_pickle(data=ee_vocab, file_path=file_path)
    return ee_vocab


def timeit(func: Callable) -> Callable:
    """Decorator to measure and print execution time and memory usage.

    Args:
        func: Function to be timed.

    Returns:
        Wrapped function that prints timing information.
    """
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        total_time = time.perf_counter() - start_time
        memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1_000_000
        print(f'Took {total_time:.4f} secs | Current Memory Usage {memory_mb:.5f} MB')
        return result
    return timeit_wrapper

def save_pickle(*, data: Optional[object] = None, file_path: str) -> None:
    """Save data to a pickle file.

    Note: Consider using more portable formats (JSON, Parquet) for new code.

    Args:
        data: Object to serialize. If None, nothing is saved.
        file_path: Path where the pickle file will be saved.
    """
    if data is not None:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        print("Input data is None. Nothing to save.")


def load_pickle(file_path: str) -> object:
    """Load data from a pickle file.

    Note: Consider using more portable formats (JSON, Parquet) for new code.

    Args:
        file_path: Path to the pickle file.

    Returns:
        Deserialized object from the pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_term_mapping(file_path: str) -> Union[dict, pl.DataFrame]:
    """Load term-to-index mapping from pickle or CSV file.

    Attempts to load from pickle first, falls back to CSV if not found.

    Args:
        file_path: Base path without extension.

    Returns:
        Dictionary or Polars DataFrame containing the mapping.
    """
    pickle_path = f"{file_path}.p"
    try:
        return load_pickle(file_path=pickle_path)
    except FileNotFoundError:
        print(f"Pickle file not found: {pickle_path}, loading from CSV")
        return pl.read_csv(f"{file_path}.csv")

# @TODO: Could these funcs can be merged?
def select_model(args: dict, is_continual_training: bool = None, storage_path: str = None):
    isinstance(args, dict)
    assert len(args) > 0
    assert isinstance(is_continual_training, bool)
    assert isinstance(storage_path, str)
    if is_continual_training:
        # Check whether we have tensor parallelized KGE.
        files_under_storage_path = [f for f in os.listdir(storage_path) if os.path.isfile(os.path.join(storage_path, f))]
        num_of_partial_models_for_tensor_parallel= len([ i for i in files_under_storage_path if "partial" in i ])
        if num_of_partial_models_for_tensor_parallel >= 1:
            models=[]
            labelling_flag=None
            for i in range(num_of_partial_models_for_tensor_parallel):
                model, labelling_flag = intialize_model(args)
                weights = torch.load(storage_path + f'/model_partial_{i}.pt', torch.device('cpu'),weights_only=False)
                model.load_state_dict(weights)
                for parameter in model.parameters():
                    parameter.requires_grad = True
                model.train()
                models.append(model)
            return EnsembleKGE(pretrained_models=models), labelling_flag
        else:
            print('Loading pre-trained model...')
            model, labelling_flag = intialize_model(args)
            try:
                weights = torch.load(storage_path + '/model.pt', torch.device('cpu'))
                model.load_state_dict(weights)
                for parameter in model.parameters():
                    parameter.requires_grad = True
                model.train()
            except FileNotFoundError as e:
                print(f"{storage_path}/model.pt is not found. The model will be trained with random weights")
                raise e
            return model, labelling_flag
    else:
        if args["trainer"]=="TP":
            # If it is tensor parallelized KGE, then we need to create ensemble of models.
            models = []
            labelling_flag = None
            for i in range(torch.cuda.device_count()):
                args["random_seed"] = i
                model, labelling_flag = intialize_model(args)
                models.append(model)
            model = EnsembleKGE(models=models)
        else:
            model, labelling_flag = intialize_model(args)

    return model, labelling_flag

def load_model(path_of_experiment_folder: str, model_name='model.pt',verbose=0) -> Tuple[object, Tuple[dict, dict]]:
    """ Load weights and initialize pytorch module from namespace arguments"""
    if verbose>0:
        print(f'Loading model {model_name}...', end=' ')
    start_time = time.time()
    # (1) Load weights..
    weights = torch.load(path_of_experiment_folder + f'/{model_name}', torch.device('cpu'))
    configs = load_json(path_of_experiment_folder + '/configuration.json')
    reports = load_json(path_of_experiment_folder + '/report.json')

    if configs.get("byte_pair_encoding", None):
        num_tokens, ent_dim = weights['token_embeddings.weight'].shape
        # (2) Loading input configuration.
        configs = load_json(path_of_experiment_folder + '/configuration.json')
        report = load_json(path_of_experiment_folder + '/report.json')
        # Load ordered_bpe_entities.p
        configs["ordered_bpe_entities"] = load_pickle(file_path=path_of_experiment_folder + "/ordered_bpe_entities.p")
        configs["num_tokens"] = num_tokens
        configs["max_length_subword_tokens"] = report["max_length_subword_tokens"]
    else:

        num_ent = reports["num_entities"]
        num_rel = reports["num_relations"]
        # Update the training configuration
        configs["num_entities"] = num_ent
        configs["num_relations"] = num_rel
    if verbose>0:
        print(f'Done! It took {time.time() - start_time:.3f}')
    # (4) Select the model
    model, _ = intialize_model(configs,verbose)
    # (5) Put (1) into (4)
    if isinstance(weights,torch.jit._script.RecursiveScriptModule):
        model.load_state_dict(weights.state_dict())
    else:
        model.load_state_dict(weights)
    # (6) Set it into eval model.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    start_time = time.time()
    if configs.get("byte_pair_encoding", None):
        return model, None
    else:
        if verbose>0:
            print('Loading entity and relation indexes...', end=' ')
    
        entity_to_idx = { v["entity"]:k for k,v in pd.read_csv(f"{path_of_experiment_folder}/entity_to_idx.csv",index_col=0,dtype=str).to_dict(orient='index').items()}

        relation_to_idx = { v["relation"]:k for k,v in pd.read_csv(f"{path_of_experiment_folder}/relation_to_idx.csv",index_col=0,dtype=str).to_dict(orient='index').items()}


        if verbose > 0:
            print(f'Done! It took {time.time() - start_time:.4f}')
        return model, (entity_to_idx, relation_to_idx)


def load_model_ensemble(path_of_experiment_folder: str) -> Tuple[BaseKGE, Tuple[pd.DataFrame, pd.DataFrame]]:
    """ Construct Ensemble Of weights and initialize pytorch module from namespace arguments

    (1) Detect models under given path
    (2) Accumulate parameters of detected models
    (3) Normalize parameters
    (4) Insert (3) into model.
    """
    print('Constructing Ensemble of ', end=' ')
    start_time = time.time()
    # (1) Detect models under given path.
    paths_for_loading = glob.glob(path_of_experiment_folder + '/model*')
    print(f'{len(paths_for_loading)} models...')
    assert len(paths_for_loading) > 0
    num_of_models = len(paths_for_loading)
    weights = None
    # (2) Accumulate parameters of detected models.
    while len(paths_for_loading):
        p = paths_for_loading.pop()
        print(f'Model: {p}...')
        if weights is None:
            weights = torch.load(p, torch.device('cpu'))
        else:
            five_weights = torch.load(p, torch.device('cpu'))
            # (2.1) Accumulate model parameters
            for k, _ in weights.items():
                if 'weight' in k:
                    weights[k] = (weights[k] + five_weights[k])
    # (3) Normalize parameters.
    for k, _ in weights.items():
        if 'weight' in k:
            weights[k] /= num_of_models
    # (4) Insert (3) into model
    # (4.1) Load report and configuration to initialize model.
    configs = load_json(path_of_experiment_folder + '/configuration.json')
    report = load_json(path_of_experiment_folder + '/report.json')
    configs["num_entities"] = report["num_entities"]
    configs["num_relations"] = report["num_relations"]
    print(f'Done! It took {time.time() - start_time:.2f} seconds.')
    # (4.2) Select the model
    model, _ = intialize_model(configs)
    # (4.3) Put (3) into their places
    model.load_state_dict(weights, strict=True)
    # (6) Set it into eval model.
    print('Setting Eval mode & requires_grad params to False')
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    start_time = time.time()
    print('Loading entity and relation indexes...', end=' ')
    # TODO: CD: We do not need to keep the mapping in memory
    # TODO:CD: Deprecate the pickle usage for data serialization.

    entity_to_idx = {v["entity"]: k for k, v in
                     pd.read_csv(f"{path_of_experiment_folder}/entity_to_idx.csv", index_col=0).to_dict(
                         orient='index').items()}
    relation_to_idx = {v["relation"]: k for k, v in
                     pd.read_csv(f"{path_of_experiment_folder}/relation_to_idx.csv", index_col=0).to_dict(
                         orient='index').items()}

    assert isinstance(entity_to_idx, dict)
    assert isinstance(relation_to_idx, dict)
    print(f'Done! It took {time.time() - start_time:.4f}')
    return model, (entity_to_idx, relation_to_idx)


def save_numpy_ndarray(*, data: np.ndarray, file_path: str):
    with open(file_path, 'wb') as f:
        np.save(f, data)


def numpy_data_type_changer(train_set: np.ndarray, num: int) -> np.ndarray:
    """
    Detect most efficient data type for a given triples
    :param train_set:
    :param num:
    :return:
    """
    assert isinstance(num, int)
    if np.iinfo(np.int8).max > num:
        # print(f'Setting int8,\t {np.iinfo(np.int8).max}')
        train_set = train_set.astype(np.int8)
    elif np.iinfo(np.int16).max > num:
        # print(f'Setting int16,\t {np.iinfo(np.int16).max}')
        train_set = train_set.astype(np.int16)
    elif np.iinfo(np.int32).max > num:
        # print(f'Setting int32,\t {np.iinfo(np.int32).max}')
        train_set = train_set.astype(np.int32)
    else:
        raise TypeError('Int64?')
    return train_set


def save_checkpoint_model(model, path: str) -> None:
    """ Store Pytorch model into disk"""
    if isinstance(model, BaseKGE):
        torch.save(model.state_dict(), path)
    elif isinstance(model, EnsembleKGE):
        # path comes with ../model_...
        for i, partial_model in enumerate(model):
            new_path=path.replace("model.pt",f"model_partial_{i}.pt")
            torch.save(partial_model.state_dict(), new_path)
    else:
        torch.save(model.model.state_dict(), path)


def store(trained_model, model_name: str = 'model', full_storage_path: str = None,
          save_embeddings_as_csv=False) -> None:
    assert full_storage_path is not None
    assert isinstance(model_name, str)
    assert len(model_name) > 1
    save_checkpoint_model(model=trained_model, path=full_storage_path + f'/{model_name}.pt')

    if save_embeddings_as_csv:
        entity_emb, relation_ebm = trained_model.get_embeddings()
        print("Saving entity embeddings...")
        entity=pd.read_csv(f"{full_storage_path}/entity_to_idx.csv",index_col=0)["entity"]
        assert entity.index.is_monotonic_increasing
        save_embeddings(entity_emb.numpy(), indexes=entity.to_list(), path=full_storage_path + '/' + trained_model.name + '_entity_embeddings.csv')
        del entity, entity_emb
        if relation_ebm is not None:
            print("Saving relation embeddings...")
            relations = pd.read_csv(f"{full_storage_path}/relation_to_idx.csv", index_col=0)["relation"]
            assert relations.index.is_monotonic_increasing
            save_embeddings(relation_ebm.numpy(), indexes=relations, path=full_storage_path + '/' + trained_model.name + '_relation_embeddings.csv')
        else:
            pass

def add_noisy_triples(train_set: pd.DataFrame, add_noise_rate: float) -> pd.DataFrame:
    """
    Add randomly constructed triples
    :param train_set:
    :param add_noise_rate:
    :return:
    """
    num_triples = len(train_set)
    num_noisy_triples = int(num_triples * add_noise_rate)
    print(f'[4 / 14] Generating {num_noisy_triples} noisy triples for training data...')

    list_of_entities = pd.unique(train_set[['subject', 'object']].values.ravel())

    train_set = pd.concat([train_set,
                           # Noisy triples
                           pd.DataFrame(
                               {'subject': np.random.choice(list_of_entities, num_noisy_triples),
                                'relation': np.random.choice(
                                    pd.unique(train_set[['relation']].values.ravel()),
                                    num_noisy_triples),
                                'object': np.random.choice(list_of_entities, num_noisy_triples)}
                           )
                           ], ignore_index=True)

    del list_of_entities

    assert num_triples + num_noisy_triples == len(train_set)
    return train_set

def read_or_load_kg(args, cls):
    print('*** Read or Load Knowledge Graph  ***')
    start_time = time.time()
    kg = cls(dataset_dir=args.dataset_dir,
             byte_pair_encoding=args.byte_pair_encoding,
             padding=True if args.byte_pair_encoding and args.model != "BytE" else False,
             add_noise_rate=args.add_noise_rate,
             sparql_endpoint=args.sparql_endpoint,
             path_single_kg=args.path_single_kg,
             add_reciprocal=args.apply_reciprical_or_noise,
             eval_model=args.eval_model,
             read_only_few=args.read_only_few,
             sample_triples_ratio=args.sample_triples_ratio,
             path_for_serialization=args.full_storage_path,
             path_for_deserialization=args.path_experiment_folder if hasattr(args, 'path_experiment_folder') else None,
             backend=args.backend,
             training_technique=args.scoring_technique,
             separator=args.separator)
    print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
    # (2) Share some info about data for easy access.
    print(kg.description_of_input)
    return kg


def intialize_model(args: Dict, verbose: int = 0) -> Tuple[BaseKGE, str]:
    """Initialize a knowledge graph embedding model.

    Args:
        args: Dictionary containing model configuration including 'model' key.
        verbose: Verbosity level. If > 0, prints initialization message.

    Returns:
        Tuple of (initialized model, form of labelling string).

    Raises:
        ValueError: If the model name is not recognized.
    """
    if verbose > 0:
        print(f"Initializing {args['model']}...")
    model_name = args['model']

    # Handle PyKEEN models
    if "pykeen" in model_name.lower():
        return PykeenKGE(args=args), "EntityPrediction"

    # Use model registry for standard models
    if model_name in MODEL_REGISTRY:
        model_class, form_of_labelling = MODEL_REGISTRY[model_name]
        return model_class(args=args), form_of_labelling

    raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")


# Keep backward compatibility - this is now handled by the registry
def _legacy_intialize_model(args: dict, verbose: int = 0) -> Tuple[object, str]:
    """Legacy model initialization (deprecated, use intialize_model instead)."""
    if verbose > 0:
        print(f"Initializing {args['model']}...")
    model_name = args['model']
    if "pykeen" in model_name.lower():
        model = PykeenKGE(args=args)
        form_of_labelling = "EntityPrediction"
    elif model_name == 'Shallom':
        model = Shallom(args=args)
        form_of_labelling = 'RelationPrediction'
    elif model_name == 'ConEx':
        model = ConEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'AConEx':
        model = AConEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'QMult':
        model = QMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'OMult':
        model = OMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'ConvQ':
        model = ConvQ(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'AConvQ':
        model = AConvQ(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'ConvO':
        model = ConvO(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'AConvO':
        model = AConvO(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'ComplEx':
        model = ComplEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'DistMult':
        model = DistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'TransE':
        model = TransE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'Pyke':
        model = Pyke(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'Keci':
        model = Keci(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'CKeci':
        model = CKeci(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'BytE':
        model = BytE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'LFMult':
        model = LFMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'DeCaL':
        model =DeCaL(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'DualE':
        model =DualE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'CoKE':
        model = CoKE(args=args)
        form_of_labelling = 'EntityPrediction'
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return model, form_of_labelling


def load_json(path: str) -> Dict:
    """Load JSON file into a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary containing the JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_embeddings(embeddings: np.ndarray, indexes: List, path: str) -> None:
    """Save embeddings to a CSV file.

    Args:
        embeddings: NumPy array of embeddings with shape (n_items, embedding_dim).
        indexes: List of index labels (entity/relation names).
        path: Output file path.
    """
    try:
        pd.DataFrame(embeddings, index=indexes).to_csv(path)
    except (KeyError, AttributeError) as e:
        print(f'Exception occurred while saving embeddings: {e}')
        print('Computation will continue.')


@timeit
def vocab_to_parquet(vocab_to_idx, name, path_for_serialization, print_into):
    # @TODO: This function should take any DASK/Pandas DataFrame or Series.
    print(print_into)
    vocab_to_idx.to_parquet(path_for_serialization + f'/{name}', compression='gzip', engine='pyarrow')
    print('Done !\n')


def create_experiment_folder(folder_name: str = 'Experiments') -> str:
    """Create a timestamped experiment folder.

    Args:
        folder_name: Base directory name for experiments.

    Returns:
        Full path to the created experiment folder.
    """
    directory = os.path.join(os.getcwd(), folder_name)
    timestamp = str(datetime.datetime.now()).replace(":", "-")
    path_of_folder = os.path.join(directory, timestamp)
    os.makedirs(path_of_folder)
    return path_of_folder


def continual_training_setup_executor(executor) -> None:
    # TODO:CD:Deprecate it
    if executor.is_continual_training:
        # (4.1) If it is continual, then store new models on previous path.
        executor.storage_path = executor.args.full_storage_path
    else:
        # Create a single directory containing KGE and all related data
        if executor.args.path_to_store_single_run:
            os.makedirs(executor.args.path_to_store_single_run, exist_ok=False)
            executor.args.full_storage_path = executor.args.path_to_store_single_run
        else:
            # Create a parent and subdirectory.
            executor.args.full_storage_path = create_experiment_folder(folder_name=executor.args.storage_path)
        executor.storage_path = executor.args.full_storage_path
        with open(executor.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(executor.args)
            json.dump(temp, file_descriptor, indent=3)


def exponential_function(x: np.ndarray, lam: float, ascending_order=True) -> torch.FloatTensor:
    # A sequence in exponentially decreasing order
    result = np.exp(-lam * x) / np.sum(np.exp(-lam * x))
    assert 0.999 < sum(result) < 1.0001
    result = np.flip(result) if ascending_order else result
    return torch.tensor(result.tolist())


@timeit
def load_numpy(path) -> np.ndarray:
    print('Loading indexed training data...', end='')
    with open(path, 'rb') as f:
        data = np.load(f)
    return data


def evaluate(entity_to_idx, scores, easy_answers, hard_answers):
    """
    # @TODO: CD: Renamed this function
    Evaluate multi hop query answering on different query types
    """
    # Calculate MRR considering the hard and easy answers
    total_mrr = 0
    total_h1 = 0
    total_h3 = 0
    total_h10 = 0
    num_queries = len(scores)
    # @TODO: Dictionary keys do not need to be in order, zip(entity_to_idx.keys(), entity_score) is not a viable solution
    # @TODO: Although it is working
    # @TODO: Use pytorch to obtain the entities sorted in the descending order of scores
    for query, entity_score in scores.items():
        entity_scores = [(ei, s) for ei, s in zip(entity_to_idx.keys(), entity_score)]
        entity_scores = sorted(entity_scores, key=lambda x: x[1], reverse=True)

        # Extract corresponding easy and hard answers
        easy_ans = easy_answers[query]
        hard_ans = hard_answers[query]
        easy_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in easy_ans]
        hard_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in hard_ans]

        answer_indices = easy_answer_indices + hard_answer_indices

        cur_ranking = np.array(answer_indices)

        # Sort by position in the ranking; indices for (easy + hard) answers
        cur_ranking, indices = np.sort(cur_ranking), np.argsort(cur_ranking)
        num_easy = len(easy_ans)
        num_hard = len(hard_ans)

        # Indices with hard answers only
        masks = indices >= num_easy

        # Reduce ranking for each answer entity by the amount of (easy+hard) answers appearing before it
        answer_list = np.arange(num_hard + num_easy, dtype=float)
        cur_ranking = cur_ranking - answer_list + 1

        # Only take indices that belong to the hard answers
        cur_ranking = cur_ranking[masks]
        # print(cur_ranking)
        mrr = np.mean(1.0 / cur_ranking)
        h1 = np.mean((cur_ranking <= 1).astype(float))
        h3 = np.mean((cur_ranking <= 3).astype(float))
        h10 = np.mean((cur_ranking <= 10).astype(float))
        total_mrr += mrr
        total_h1 += h1
        total_h3 += h3
        total_h10 += h10
    # average for all queries of a type
    avg_mrr = total_mrr / num_queries
    avg_h1 = total_h1 / num_queries
    avg_h3 = total_h3 / num_queries
    avg_h10 = total_h10 / num_queries

    return avg_mrr, avg_h1, avg_h3, avg_h10



def download_file(url, destination_folder="."):
    response = requests.get(url, stream=True)
    # lazy import
    from urllib.parse import urlparse

    if response.status_code == 200:
        filename = os.path.join(destination_folder, os.path.basename(urlparse(url).path))
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")


def download_files_from_url(base_url:str, destination_folder=".")->None:
    """

    Parameters
    ----------
    base_url: e.g. "https://files.dice-research.org/projects/DiceEmbeddings/KINSHIP-Keci-dim128-epoch256-KvsAll"

    destination_folder: e.g. "KINSHIP-Keci-dim128-epoch256-KvsAll"

    Returns
    -------

    """
    # lazy import
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        print("Please install the 'beautifulsoup4' package by running: pip install beautifulsoup4")
        raise
    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the table with id "list"
        table = soup.find('table', {'id': 'list'})
        # Extract all hrefs under the table
        hrefs = [a['href'] for a in table.find_all('a', href=True)]
        # To remove '?C=N&O=A', '?C=N&O=D', '?C=S&O=A', '?C=S&O=D', '?C=M&O=A', '?C=M&O=D', '../'
        hrefs = [i for i in hrefs if len(i) > 3 and "." in i]
        for file_url in hrefs:
            download_file(base_url + "/" + file_url, destination_folder)
    else:
        print("ERROR:", response.status_code)

def download_pretrained_model(url: str) -> str:
    assert url[-1] != "/"
    dir_name = url[url.rfind("/") + 1:]
    url_to_download_from = f"https://files.dice-research.org/projects/DiceEmbeddings/{dir_name}"
    if os.path.exists(dir_name):
        print("Path exists", dir_name)
    else:
        os.mkdir(dir_name)
        download_files_from_url(url_to_download_from, destination_folder=dir_name)
    return dir_name

def write_csv_from_model_parallel(path: str) :
    """Create"""
    assert os.path.exists(path), "Path does not exist"
    # Detect files that start with model_ and end with .pt
    model_files = [f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt")]
    model_files.sort()  # Sort to maintain order if necessary (e.g., model_0.pt, model_1.pt)

    entity_embeddings=[]
    relation_embeddings=[]

    # Process each model file
    for model_file in model_files:
        model_path = os.path.join(path, model_file)
        # Load model
        model = torch.load(model_path)
        # Assuming model has a get_embeddings method
        entity_emb, relation_emb = model["_orig_mod.entity_embeddings.weight"], model["_orig_mod.relation_embeddings.weight"]
        entity_embeddings.append(entity_emb)
        relation_embeddings.append(relation_emb)

    return torch.cat(entity_embeddings, dim=1), torch.cat(relation_embeddings, dim=1)


def from_pretrained_model_write_embeddings_into_csv(path: str) -> None:
    """ """
    assert os.path.exists(path), "Path does not exist"
    config = load_json(path + '/configuration.json')
    entity_csv_path = os.path.join(path, f"{config['model']}_entity_embeddings.csv")
    relation_csv_path = os.path.join(path, f"{config['model']}_relation_embeddings.csv")

    if config["trainer"]=="TP":
        entity_emb, relation_emb = write_csv_from_model_parallel(path)
    else:
        # Load model
        model = torch.load(os.path.join(path, "model.pt"))
        # Assuming model has a get_embeddings method
        entity_emb, relation_emb = model["entity_embeddings.weight"], model["relation_embeddings.weight"]
    str_entity = pd.read_csv(f"{path}/entity_to_idx.csv", index_col=0)["entity"]
    assert str_entity.index.is_monotonic_increasing
    str_entity=str_entity.to_list()
    # Write entity embeddings with headers and indices
    with open(entity_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Add header (e.g., "", "0", "1", ..., "N")
        headers = [""] + [f"{i}" for i in range(entity_emb.size(1))]
        writer.writerow(headers)
        # Add rows with index
        for i_row, (name,row) in enumerate(zip(str_entity,entity_emb)):
            writer.writerow([name] + row.tolist())
    str_relations = pd.read_csv(f"{path}/relation_to_idx.csv", index_col=0)["relation"]
    assert str_relations.index.is_monotonic_increasing

    # Write relation embeddings with headers and indices
    with open(relation_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Add header (e.g., "", "0", "1", ..., "N")
        headers = [""] + [f"{i}" for i in range(relation_emb.size(1))]
        writer.writerow(headers)
        # Add rows with index
        for i_row, (name, row) in enumerate(zip(str_relations,relation_emb)):
            writer.writerow([name]+ row.tolist())

    """
    
    # Write entity embeddings directly to CSV
    with open(entity_csv_path, "w") as f:
        for row in entity_emb:
            f.write(",".join(map(str, row.tolist())) + "\n")

    # Write relation embeddings directly to CSV
    with open(relation_csv_path, "w") as f:
        for row in relation_emb:
            f.write(",".join(map(str, row.tolist())) + "\n")

    # Convert to numpy
    pd.DataFrame(entity_emb.numpy()).to_csv(entity_csv_path, index=True, header=False)
    # If CSV files do not exist, create them
    pd.DataFrame(relation_emb.numpy()).to_csv(relation_csv_path, index=True, header=False)
    """