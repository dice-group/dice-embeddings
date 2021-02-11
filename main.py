from argparse import ArgumentParser

from dataset import KG, StandardDataModule
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold
from funcs import sanity_checking_with_arguments, select_model
import numpy as np

from pytorch_lightning import loggers as pl_loggers


def standard_training(args, dataset):
    if not args.logging:
        args.checkpoint_callback = False
        args.logger = False
    print(args.logging)
    trainer = pl.Trainer.from_argparse_args(args)
    args.num_entities = dataset.num_entities
    args.num_relations = dataset.num_relations

    model, form_of_labelling = select_model(args)
    dataset = StandardDataModule(dataset=dataset, form=form_of_labelling,
                                 batch_size=args.batch_size, num_workers=args.num_workers)

    trainer.fit(model, train_dataloader=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())
    trainer.test(model, test_dataloaders=dataset.test_dataloader())
    return model


def k_fold_cv_training(args, dataset):
    if args.num_folds_for_cv < 2:
        print(
            f'k-fold cross-validation requires at least one train/test split, but got only ***num_folds_for_cv*** => {args.num_folds_for_cv}.num_folds_for_cv is now set to 10.')
        args.num_folds_for_cv = 10

    kf = KFold(n_splits=args.num_folds_for_cv)

    #    for train_index, test_index in kf.split(dataset):

    """

    for train_index, test_index in kf.split(dataset):

        raise ValueError
        k_fold_loader_training = DataLoader(dataset.create_fold(train_index), batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers)
        k_fold_loader_test = DataLoader(dataset.create_fold(test_index), batch_size=args.batch_size,
                                        num_workers=args.num_workers)

        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, train_dataloader=k_fold_loader_training)
        trainer.test(model, k_fold_loader_test)
    """

    return dataset


def simple_hit_1(trained_model, dataset):
    if  trained_model.name!='Shallom':
        print(f'Not yet implemented for {trained_model.name}')
        return
    # @TODO This needs to integrated into models.
    #########################################
    # Evaluation mode. Parallelize below computation.
    trained_model.eval()
    rel = np.array(dataset.relations)  # for easy indexing.
    num_rel = len(rel)
    ranks = []

    predictions_save = []
    for triple in dataset.test_set:
        s, p, o = triple
        preds = trained_model.forward(torch.LongTensor([dataset.entity_to_idx[s]]),
                                      torch.LongTensor([dataset.entity_to_idx[o]]))
        _, ranked_idx_rels = preds.topk(k=num_rel)
        ranked_relations = rel[ranked_idx_rels][0]

        predictions_save.append([s, p, o, ranked_relations[0]])

        rank = 1 + np.argwhere(ranked_relations == p)[0][0]
        ranks.append(rank)

    print(f'Raw Mean reciprocal rank: {np.mean(1. / np.array(ranks))}')

    for it, t in enumerate(predictions_save):
        s, p, o, predicted_p = t
        print(f'{it}. test triples => {s} {p} {o} \t =>{trained_model.name} => {predicted_p}')
        if it == 10:
            break


def start(args):
    sanity_checking_with_arguments(args)
    dataset = KG(data_dir=args.path_dataset_folder)
    if dataset.is_valid_test_available():
        trained_model = standard_training(args, dataset)
    else:
        trained_model = k_fold_cv_training(args, dataset)

    simple_hit_1(trained_model, dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during loadingIncrease ')
    parser.add_argument('--kvsall', default=True)
    parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_folds_for_cv', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1025)
    parser.add_argument('--embedding_dim', type=int, default=25)
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.1)
    parser.add_argument("--model", type=str, default='Shallom', help="Models:Shallom")
    parser.add_argument("--logging", default=False)
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.0,
                        help='The ratio of the size of the first affine transformation with respect to size of the embeddings')
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS')

    start(parser.parse_args())
