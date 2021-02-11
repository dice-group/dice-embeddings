import os
from typing import AnyStr,Tuple
from models import *
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


def select_model(args) -> Tuple[pl.LightningModule,AnyStr]:
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
