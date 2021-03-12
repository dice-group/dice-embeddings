import os
from typing import AnyStr, Tuple
from models import *
import numpy as np
import torch
import datetime
import logging

import argparse
import pytorch_lightning as pl
import sys
def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/DBpedia')
    parser.add_argument("--storage_path", type=str, default='DAIKIRI_Storage')

    # Models.
    parser.add_argument("--model", type=str, default='Shallom',
                        help="Available models: ConvQ, OMult, QMult, ConEx, Shallom, ConEx, ComplEx, DistMult")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=25)
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=32, help="# of output channels in convolution")
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.5,
                        help='The ratio of the size of the affine transformation w.r.t. the size of the embeddings')

    # Hyperparameters pertaining to regularization.
    parser.add_argument('--input_dropout_rate', type=float, default=0.2)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.2)
    parser.add_argument("--feature_map_dropout_rate", type=int, default=.3)
    parser.add_argument('--apply_unit_norm', type=bool, default=False)

    # Hyperparameters for training.
    parser.add_argument("--max_num_epochs", type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--check_val_every_n_epochs", type=int, default=1000)

    # Data Augmentation.
    parser.add_argument("--add_reciprical", type=bool, default=False)

    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')
    parser.add_argument('--kvsall', default=True)
    parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_folds_for_cv', type=int, default=0, help='Number of folds in k-fold cross validation.'
                                                                        'If >2,no evaluation scenario is applied implies no evaluation.')
    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)
def preprocesses_input_args(arg):
    # To update the default value of Trainer in pytorch-lightnings
    arg.max_epochs = arg.max_num_epochs
    del arg.max_num_epochs
    arg.check_val_every_n_epoch = arg.check_val_every_n_epochs
    del arg.check_val_every_n_epochs

    arg.checkpoint_callback = False
    arg.logger = False
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

    try:
        assert not (args.kvsall is True and args.negative_sample_ratio > 0)
    except AssertionError:
        print(f'Training  strategy: If args.kvsall is TRUE, args.negative_sample_ratio must be 0'
              f'args.kvsall:{args.kvsall} and args.negative_sample_ratio:{args.negative_sample_ratio}.')
        raise

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


def select_model(args) -> Tuple[pl.LightningModule, AnyStr]:
    if args.model == 'Shallom':
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


def compute_mrr_based_on_relation_ranking(trained_model, triples, entity_to_idx, relations):
    rel = np.array(relations)  # for easy indexing.
    num_rel = len(rel)
    ranks = []

    predictions_save = []
    for triple in triples:
        s, p, o = triple
        preds = trained_model.forward(torch.LongTensor([entity_to_idx[s]]),
                                      torch.LongTensor([entity_to_idx[o]]))

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
        preds = trained_model.forward(torch.LongTensor([entity_to_idx[s]]),
                                      torch.LongTensor([relation_to_idx[p]]))

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
