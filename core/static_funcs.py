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


def store_kge(trained_model, path: str):
    torch.save(trained_model.state_dict(), path)


def model_fitting(trainer, model, train_dataloaders) -> None:
    trainer.fit(model, train_dataloaders=train_dataloaders)


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
    print('*** Read & Parse input data for training and testing ***')
    start_time = time.time()
    # 1. Read & Parse input data
    kg = cls(data_dir=args.path_dataset_folder,
             large_kg_parse=args.large_kg_parse,
             add_reciprical=args.add_reciprical,
             eval_model=args.eval,
             read_only_few=args.read_only_few,
             sample_triples_ratio=args.sample_triples_ratio,
             path_for_serialization=args.full_storage_path,
             add_noise_rate=args.add_noise_rate)
    print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
    print(kg.description_of_input)
    return kg


def reload_input_data(storage_path: str, cls):
    # 1. Read & Parse input data
    print("1. Reload Parsed Input Data")
    return cls(deserialize_flag=storage_path)


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

    arg.add_reciprical = True if arg.scoring_technique in ['KvsAll', '1vsAll'] else False
    if arg.sample_triples_ratio is not None:
        assert 1.0 >= arg.sample_triples_ratio >= 0.0
    sanity_checking_with_arguments(arg)
    if arg.save_model_at_every_epoch is None:
        arg.save_model_at_every_epoch = arg.max_epochs
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
        assert os.path.isfile(args.path_dataset_folder + '/train.txt')
    except AssertionError:
        print(f'The directory {args.path_dataset_folder} must contain a **train.txt** .')
        raise

    args.eval = bool(args.eval)
    args.large_kg_parse = bool(args.large_kg_parse)


def select_model(args) -> Tuple[pl.LightningModule, AnyStr]:
    if args.model == 'KronELinear':
        model = KronELinear(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'KPDistMult':
        model = KPDistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'KPFullDistMult':
        # Full compression of entities and relations.
        model = KPFullDistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'KronE':
        model = KronE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'KronE_wo_f':
        model = KronE_wo_f(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'BaseKronE':
        model = BaseKronE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'Shallom':
        model = Shallom(args=args)
        form_of_labelling = 'RelationPrediction'
    elif args.model == 'ConEx':
        model = ConEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'QMult':
        model = QMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'OMult':
        model = OMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'ConvQ':
        model = ConvQ(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'ConvO':
        model = ConvO(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'ComplEx':
        model = ComplEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif args.model == 'DistMult':
        model = DistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    else:
        raise ValueError
    return model, form_of_labelling


def load_model(args) -> torch.nn.Module:
    """ Load weights and initialize pytorch module from namespace arguments"""
    # (1) Load weights from experiment repo
    weights = torch.load(args.path_of_experiment_folder + '/model.pt', torch.device('cpu'))
    model, _ = select_model(args)
    model.load_state_dict(weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    entity_to_idx = pd.read_parquet(args.path_of_experiment_folder + '/entity_to_idx.gzip').to_dict()['entity']
    relation_to_idx = pd.read_parquet(args.path_of_experiment_folder + '/relation_to_idx.gzip').to_dict()['relation']
    return model, entity_to_idx, relation_to_idx


def compute_mrr_based_on_relation_ranking(trained_model, triples, entity_to_idx, relations):
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


def load_configuration(p: str) -> CustomArg:
    assert os.path.isfile(p)
    with open(p, 'r') as r:
        args = json.load(r)
    return CustomArg(**args)
