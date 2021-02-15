import os
from typing import AnyStr, Tuple
from models import *
import numpy as np
import torch
import datetime


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
