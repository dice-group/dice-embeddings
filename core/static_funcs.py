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
import argparse


def argparse_default(description=None):
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default Trainer param https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods

    # Dataset and storage related
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/Countries-S1',
                        help="The path of a folder containing input data")
    parser.add_argument("--large_kg_parse", type=int, default=0, help='A flag for using all cores at parsing.')
    parser.add_argument("--storage_path", type=str, default='DAIKIRI_Storage',
                        help="Embeddings, model, and any other related data will be stored therein.")
    parser.add_argument("--read_only_few", type=int, default=None, help='READ only first N triples. If 0, read all.')
    parser.add_argument("--sample_triples_ratio", type=float, default=None, help='Sample input data.')

    # Models.
    parser.add_argument("--model", type=str,
                        default='KronE',
                        help="Available models: KronE, ConEx, ConvQ, ConvO,  QMult, OMult, Shallom, ConEx, ComplEx, DistMult")
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=1, help='Number of epochs for training. '
                                                                  'This disables max_epochs and min_epochs of pl.Trainer')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini batch size')
    parser.add_argument("--lr", type=float, default=0.01, help='Learning rate')
    parser.add_argument("--label_smoothing_rate", type=float, default=None, help='None for not using it.')

    # Model Parameters
    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Number of dimensions for an embedding vector. This parameter is used for those models requiring same number of embedding vector for entities and relations.')
    parser.add_argument('--entity_embedding_dim', type=int, default=16,
                        help='Number of dimensions for an entity embedding vector. '
                             'This parameter is used for those model having flexibility of using different sized entity and relation embeddings.')
    parser.add_argument('--rel_embedding_dim', type=int, default=16,
                        help='Number of dimensions for an entity embedding vector. '
                             'This parameter is used for those model having flexibility of using different sized entity and relation embeddings.')
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=1, help="# of output channels in convolution")
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.5,
                        help='The ratio of the size of the affine transformation w.r.t. the size of the embeddings')
    # Flags for computation
    parser.add_argument("--eval", type=int, default=1,
                        help='A flag for using evaluation')
    parser.add_argument("--eval_on_train", type=int, default=1,
                        help='A flag for using train data to evaluation ')
    # Hyperparameters pertaining to regularization.
    parser.add_argument('--input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.0)
    parser.add_argument("--feature_map_dropout_rate", type=int, default=0.0)
    parser.add_argument('--apply_unit_norm', type=bool, default=False)
    # Hyperparameters for training.
    parser.add_argument('--scoring_technique', default='1vsAll', help="1vsAll, KvsAll, NegSample.")
    parser.add_argument('--neg_ratio', type=int, default=1)
    # Data Augmentation.
    parser.add_argument('--num_folds_for_cv', type=int, default=0, help='Number of folds in k-fold cross validation.'
                                                                        'If >2,no evaluation scenario is applied implies no evaluation.')
    # This is a workaround for read
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


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
        print(f'The path does not direct to a file {args.path_train_dataset}')
        raise

    try:
        assert os.path.isfile(args.path_dataset_folder + '/train.txt')
    except AssertionError:
        print(f'The directory {args.path_dataset_folder} must contain a **train.txt** .')
        raise

    args.eval = bool(args.eval)
    args.large_kg_parse = bool(args.large_kg_parse)


def select_model(args) -> Tuple[pl.LightningModule, AnyStr]:
    if args.model == 'KronE':
        model = KronE(args=args)
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
