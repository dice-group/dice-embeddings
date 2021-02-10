from argparse import ArgumentParser

from dataset import KG, StandardDataModule
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold
from funcs import sanity_checking_with_arguments
import numpy as np

from models import Shallom

from pytorch_lightning import loggers as pl_loggers


def standard_training(args, dataset):
    if not args.logging:
        args.checkpoint_callback = False
        args.logger = False
    print(args.logging)
    trainer = pl.Trainer.from_argparse_args(args)
    if args.model == 'Shallom':
        args.num_entities = dataset.num_entities
        args.num_relations = dataset.num_relations
        model = Shallom(args=args)
    else:
        # @TODOs ConEx, QMult, OMult etc.
        raise ValueError

    dataset = StandardDataModule(dataset=dataset, form='RelationPrediction', batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    trainer.fit(model, train_dataloader=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())
    trainer.test(model, test_dataloaders=dataset.test_dataloader())
    return model


def k_fold_cv_training(args, dataset):
    if args.num_folds_for_cv < 2:
        print(
            f'k-fold cross-validation requires at least one train/test split, but got only ***num_folds_for_cv*** => {args.num_folds_for_cv}.num_folds_for_cv is now set to 10.')
        args.num_folds_for_cv = 10

    kf = KFold(n_splits=args.num_folds_for_cv)

    print(dataset)

    #    for train_index, test_index in kf.split(dataset):

    exit(1)

    exit(1)
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


def start(args):
    """
    Namespace(accelerator=None, accumulate_grad_batches=1, amp_backend='native', amp_level='O2', auto_lr_find=False,
    auto_scale_batch_size=False, auto_select_gpus=False, automatic_optimization=None, batch_size=1024,
    benchmark=False, check_val_every_n_epoch=3, checkpoint_callback=True, default_root_dir='.',
    deterministic=False, distributed_backend=None, embedding_dim=25, enable_pl_optimizer=None,
    fast_dev_run=False, flush_logs_every_n_steps=100, gpus=<function _gpus_arg_default at 0x7fae797560e0>,
    gradient_clip_val=0, kvsall=True, limit_test_batches=1.0, limit_train_batches=1.0, limit_val_batches=1.0,
    log_every_n_steps=50, log_gpu_memory=None, logger=True, max_epochs=15, max_steps=None, min_epochs=1,
    min_steps=None, model='Shallom', move_metrics_to_cpu=False, negative_sample_ratio=0, num_folds_for_cv=0,
    num_nodes=1, num_processes=1, num_sanity_val_steps=2, num_workers=32, overfit_batches=0.0,
    path_dataset_folder='KGs/UMLS', plugins=None, precision=32, prepare_data_per_node=True, process_position=0,
    profiler=None, progress_bar_refresh_rate=1, reload_dataloaders_every_epoch=False, replace_sampler_ddp=True,
    resume_from_checkpoint=None, shallom_width_ratio_of_emb=1.0, sync_batchnorm=False, terminate_on_nan=False,
    tpu_cores=<function _gpus_arg_default at 0x7fae797560e0>, track_grad_norm=-1, truncated_bptt_steps=None,
    val_check_interval=1.0, val_percent_check=1.0, weights_save_path=None, weights_summary='top')


    auto_lr_find=False => Aut find learning rate.https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#auto-lr-find
    # later call # call tune to find the lr
    # trainer.tune(model)
    :param args:
    :return:
    """
    sanity_checking_with_arguments(args)
    dataset = KG(data_dir=args.path_dataset_folder)
    if dataset.is_valid_test_available():
        model = standard_training(args, dataset)
    else:
        k_fold_cv_training(args, dataset)

    #########################################
    # Evaluation mode. Parallelize below computation.
    model.eval()
    rel = np.array(dataset.relations)  # for easy indexing.
    num_rel = len(rel)
    ranks = []
    for triple in dataset.test_set:
        s, p, o = triple
        preds = model.forward(torch.LongTensor([dataset.entity_to_idx[s]]),
                              torch.LongTensor([dataset.entity_to_idx[o]]))
        _, ranked_idx_rels = preds.topk(k=num_rel)
        ranked_relations = rel[ranked_idx_rels][0]
        rank = 1 + np.argwhere(ranked_relations == p)[0][0]
        ranks.append(rank)

    print(f'Raw Mean reciprocal rank: {np.mean(1. / np.array(ranks))}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # @TODO add dropouts.
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during loadingIncrease ')
    parser.add_argument('--kvsall', default=True)
    parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_folds_for_cv', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1025)
    parser.add_argument('--embedding_dim', type=int, default=25)
    parser.add_argument("--model", type=str, default='Shallom', help="Models:Shallom")
    parser.add_argument("--logging", default=False)
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.0,
                        help='The ratio of the size of the first affine transformation with respect to size of the embeddings')
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS')

    start(parser.parse_args())
