import os
from typing import AnyStr, Tuple
import numpy as np
import torch
import datetime
import logging
from collections import defaultdict
import pytorch_lightning as pl
import sys
from .helper_classes import CustomArg
from .models import *
import time
import pandas as pd
import json
import glob
import dask.dataframe as dd
from .sanity_checkers import sanity_checking_with_arguments, config_kge_sanity_checking


def index_triples(train_set, entity_to_idx: dict, relation_to_idx: dict):
    """
    :param train_set: dask dataframe/pandas dataframe
    :param entity_to_idx:
    :param relation_to_idx:
    :return:
    """
    train_set['subject'] = train_set['subject'].map(
        lambda x: entity_to_idx[x] if entity_to_idx.get(x) else None)
    train_set['relation'] = train_set['relation'].map(
        lambda x: relation_to_idx[x] if relation_to_idx.get(x) else None)
    train_set['object'] = train_set['object'].map(
        lambda x: entity_to_idx[x] if entity_to_idx.get(x) else None)
    train_set = train_set.dropna()
    train_set = train_set.astype(int)
    return train_set


def add_noisy_triples(train_set, add_noise_rate: float) -> pd.DataFrame:
    """
    Add randomly constructed triples
    :param train_set:
    :param add_noise_rate:
    :return:
    """
    # Can not be applied on large
    train_set = train_set.compute()

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


def store_kge(trained_model, path: str) -> None:
    torch.save(trained_model.state_dict(), path)


def create_recipriocal_triples_from_dask(x):
    """
    Add inverse triples into dask dataframe
    :param x:
    :return:
    """
    # x dask dataframe
    return dd.concat([x, x['object'].to_frame(name='subject').join(
        x['relation'].map(lambda x: x + '_inverse').to_frame(name='relation')).join(
        x['subject'].to_frame(name='object'))], ignore_index=True)


def save_embeddings(embeddings: np.ndarray, indexes, path: str) -> None:
    """

    :param embeddings:
    :param indexes:
    :param path:
    :return:
    """
    try:
        df = pd.DataFrame(embeddings, index=indexes)
        del embeddings
        num_mb = df.memory_usage(index=True, deep=True).sum() / (10 ** 6)
        if num_mb > 10 ** 6:
            df = dd.from_pandas(df, npartitions=len(df) / 100)
            # PARQUET wants columns to be stn
            df.columns = df.columns.astype(str)
            df.to_parquet(path)
        else:
            df.to_csv(path)
    except KeyError or AttributeError as e:
        print('Exception occurred at saving entity embeddings. Computation will continue')
        print(e)
    del df


def read_input_data(args, cls):
    """ Read & Parse input data for training and testing"""
    print('*** Read, Parse, and Serialize Knowledge Graph  ***')
    start_time = time.time()
    # 1. Read & Parse input data
    kg = cls(data_dir=args.path_dataset_folder,
             large_kg_parse=args.large_kg_parse,
             add_reciprical=args.add_reciprical,
             eval_model=args.eval,
             read_only_few=args.read_only_few,
             sample_triples_ratio=args.sample_triples_ratio,
             path_for_serialization=args.full_storage_path,
             add_noise_rate=args.add_noise_rate,
             min_freq_for_vocab=args.min_freq_for_vocab
             )
    print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
    print(kg.description_of_input)
    return kg


def reload_input_data(storage_path: str, cls):
    print('*** Reload Knowledge Graph  ***')
    start_time = time.time()
    kg = cls(deserialize_flag=storage_path)
    print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
    print(kg.description_of_input)
    return kg


def performance_debugger(func_name):
    def func_decorator(func):
        def debug(*args, **kwargs):
            starT = time.time()
            print('\n######', func_name, ' ', end='')
            r = func(*args, **kwargs)
            print(f' took  {time.time() - starT:.3f}  seconds')
            return r

        return debug

    return func_decorator


def preprocesses_input_args(arg):
    # To update the default value of Trainer in pytorch-lightnings
    arg.max_epochs = arg.num_epochs
    arg.min_epochs = arg.num_epochs
    if arg.add_noise_rate is not None:
        assert 1. >= arg.add_noise_rate > 0.

    arg.learning_rate = arg.lr
    arg.deterministic = True
    # Below part will be investigated
    arg.check_val_every_n_epoch = 10 ** 6
    # del arg.check_val_every_n_epochs
    arg.checkpoint_callback = False
    arg.logger = False
    arg.eval = True if arg.eval == 1 else False
    arg.eval_on_train = True if arg.eval_on_train == 1 else False
    arg.add_reciprical = True if arg.scoring_technique in ['KvsAll', '1vsAll'] else False
    if arg.sample_triples_ratio is not None:
        assert 1.0 >= arg.sample_triples_ratio >= 0.0
    sanity_checking_with_arguments(arg)
    if arg.num_folds_for_cv > 0:
        arg.eval = True
    if arg.model == 'Shallom':
        arg.scoring_technique = 'KvsAll'
    return arg


def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def create_experiment_folder(folder_name='Experiments'):
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder


def select_model(args: dict) -> Tuple[pl.LightningModule, AnyStr]:
    print('Select model...', end=' ')
    start_time = time.time()
    model_name = args['model']
    if model_name == 'KronELinear':
        model = KronELinear(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KPDistMult':
        model = KPDistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KPFullDistMult':
        # Full compression of entities and relations.
        model = KPFullDistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KronE':
        model = KronE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KronE_wo_f':
        model = KronE_wo_f(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'BaseKronE':
        model = BaseKronE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'Shallom':
        model = Shallom(args=args)
        form_of_labelling = 'RelationPrediction'
    elif model_name == 'ConEx':
        model = ConEx(args=args)
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
    elif model_name == 'ConvO':
        model = ConvO(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'ComplEx':
        model = ComplEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'DistMult':
        model = DistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    else:
        raise ValueError
    print(f'Done! It took {time.time() - start_time:.3f}')
    return model, form_of_labelling


def load_model(path_of_experiment_folder, model_path='model.pt') -> Tuple[BaseKGE, pd.DataFrame, pd.DataFrame]:
    """ Load weights and initialize pytorch module from namespace arguments"""
    print(f'Loading model {model_path}...', end=' ')
    start_time = time.time()
    # (1) Load weights..
    weights = torch.load(path_of_experiment_folder + f'/{model_path}', torch.device('cpu'))
    # (2) Loading input configuration..
    configs = load_json(path_of_experiment_folder + '/configuration.json')
    # (3) Loading the report of a training process.
    report = load_json(path_of_experiment_folder + '/report.json')
    configs["num_entities"] = report["num_entities"]
    configs["num_relations"] = report["num_relations"]
    print(f'Done! It took {time.time() - start_time:.3f}')
    # (4) Select the model
    model, _ = select_model(configs)
    # (5) Put (1) into (4)
    model.load_state_dict(weights, strict=False)
    # (6) Set it into eval model.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    start_time = time.time()
    print('Loading entity and relation indexes...', end=' ')
    entity_to_idx = pd.read_parquet(path_of_experiment_folder + '/entity_to_idx.gzip')
    relation_to_idx = pd.read_parquet(path_of_experiment_folder + '/relation_to_idx.gzip')
    print(f'Done! It took {time.time() - start_time:.4f}')
    return model, entity_to_idx, relation_to_idx


def load_model_ensemble(path_of_experiment_folder) -> Tuple[BaseKGE, pd.DataFrame, pd.DataFrame]:
    """ Construct Ensemble Of weights and initialize pytorch module from namespace arguments"""
    print('Constructing Ensemble of ', end=' ')
    start_time = time.time()
    # (1) Load weights..
    paths_for_loading = glob.glob(path_of_experiment_folder + '/model*')
    print(f'{len(paths_for_loading)} models...')
    assert len(paths_for_loading) > 0
    num_of_models = len(paths_for_loading)
    weights = None
    while len(paths_for_loading):
        p = paths_for_loading.pop()
        print(f'Model: {p}...')
        if weights is None:
            weights = torch.load(p, torch.device('cpu'))
        else:
            five_weights = torch.load(p, torch.device('cpu'))
            for k, _ in weights.items():
                if 'weight' in k:
                    weights[k] = (weights[k] + five_weights[k])
    for k, _ in weights.items():
        if 'weight' in k:
            weights[k] /= num_of_models
    # (2) Loading input configuration..
    configs = load_json(path_of_experiment_folder + '/configuration.json')
    # (3) Loading the report of a training process.
    report = load_json(path_of_experiment_folder + '/report.json')
    configs["num_entities"] = report["num_entities"]
    configs["num_relations"] = report["num_relations"]
    print(f'Done! It took {time.time() - start_time:.2f} seconds.')
    # (4) Select the model
    model, _ = select_model(configs)
    # (5) Put (1) into (4)
    model.load_state_dict(weights,strict=False)
    # (6) Set it into eval model.
    print('Setting Eval mode & requires_grad params to False')
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    start_time = time.time()
    print('Loading entity and relation indexes...', end=' ')
    entity_to_idx = pd.read_parquet(path_of_experiment_folder + '/entity_to_idx.gzip')
    relation_to_idx = pd.read_parquet(path_of_experiment_folder + '/relation_to_idx.gzip')
    print(f'Done! It took {time.time() - start_time:.4f}')
    return model, entity_to_idx, relation_to_idx


def extract_model_summary(s):
    return {'NumParam': s.total_parameters, 'EstimatedSizeMB': s.model_size}


def get_er_vocab(data):
    # head entity and relation
    er_vocab = defaultdict(list)
    for triple in data:
        er_vocab[(triple[0], triple[1])].append(triple[2])
    return er_vocab


def get_re_vocab(data):
    # head entity and relation
    re_vocab = defaultdict(list)
    for triple in data:
        re_vocab[(triple[1], triple[2])].append(triple[0])
    return re_vocab


def get_ee_vocab(data):
    # head entity and relation
    ee_vocab = defaultdict(list)
    for triple in data:
        ee_vocab[(triple[0], triple[2])].append(triple[1])
    return ee_vocab


def load_json(p: str) -> dict:
    assert os.path.isfile(p)
    with open(p, 'r') as r:
        args = json.load(r)
    return args


def compute_mrr_based_on_relation_ranking(trained_model, triples, entity_to_idx, relations):
    raise NotImplemented('This function seem to be depricated')
    rel = np.array(relations)  # for easy indexing.

    num_rel = len(rel)
    ranks = []

    predictions_save = []
    for triple in triples:
        s, p, o = triple
        x = (torch.LongTensor([entity_to_idx[s]]), torch.LongTensor([entity_to_idx[o]]))
        preds = trained_model.forward(x)

        # Rank predicted scores
        _, ranked_idx_rels = preds.topk(k=num_rel)
        # Rank all relations based on predicted scores
        ranked_relations = rel[ranked_idx_rels][0]

        # Compute and store the rank of the true relation.
        rank = 1 + np.argwhere(ranked_relations == p)[0][0]
        ranks.append(rank)
        # Store prediction.
        predictions_save.append([s, p, o, ranked_relations[0]])

    raw_mrr = np.mean(1. / np.array(ranks))
    # print(f'Raw Mean reciprocal rank on test dataset: {raw_mrr}')
    """
    for it, t in enumerate(predictions_save):
        s, p, o, predicted_p = t
        print(f'{it}. test triples => {s} {p} {o} \t =>{trained_model.name} => {predicted_p}')
        if it == 10:
            break
    """
    return raw_mrr


def compute_mrr_based_on_entity_ranking(trained_model, triples, entity_to_idx, relation_to_idx, entities):
    raise NotImplemented('This function seem to be depricated')
    #########################################
    # Evaluation mode. Parallelize below computation.
    entities = np.array(entities)  # for easy indexing.
    num_entities = len(entities)
    ranks = []

    predictions_save = []
    for triple in triples:
        s, p, o = triple
        x = (torch.LongTensor([entity_to_idx[s]]),
             torch.LongTensor([relation_to_idx[p]]))
        preds = trained_model.forward(x)

        # Rank predicted scores
        _, ranked_idx_entity = preds.topk(k=num_entities)
        # Rank all relations based on predicted scores
        ranked_entity = entities[ranked_idx_entity][0]

        # Compute and store the rank of the true relation.
        rank = 1 + np.argwhere(ranked_entity == o)[0][0]
        ranks.append(rank)
        # Store prediction.
        predictions_save.append([s, p, o, ranked_entity[0]])

    raw_mrr = np.mean(1. / np.array(ranks))
    """
    for it, t in enumerate(predictions_save):
        s, p, o, predicted_ent = t
        print(f'{it}. test triples => {s} {p} {o} \t =>{trained_model.name} => {predicted_ent}')
        if it == 10:
            break
    """
    return raw_mrr
