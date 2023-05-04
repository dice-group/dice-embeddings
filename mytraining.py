#!/usr/bin/env python3

# Addditional modules:
# ray[air,tune] bayesian-optimization

import dicee
import dicee.config
import logging
import myconfig
import os
import torch

cwd = os.getcwd()

def trainable(config):
    dataset = config['dataset']
    args = dicee.config.Args()
    if config['gpus']:
        args.trainer = 'PL'
        args.gpus = config['gpus']
    args.num_nodes = 1
    args.path_dataset_folder = f'{cwd}/KGs/{dataset}'
    args.storage_path = f'{cwd}/Experiments/{dataset}'
    print(f'{args.storage_path=}')
    args.model = 'SedE'
    args.embedding_dim = round(config.get('embedding_dim', round(config.get('complex_embedding_dim', 2)) * 16))
    args.num_epochs = int(config.get('num_epochs', 100))
    args.eval_model = 'train_val_test'
    args.neg_ratio = int(config.get('neg_ratio', 1))
    args.weight_decay = config.get('weight_decay', 0.0)
    args.input_dropout_rate = config.get('input_dropout_rate', 0.0)
    args.hidden_dropout_rate = 0.0
    args.feature_map_dropout_rate = 0.0
    args.normalization = None
    args.init_param = 'xavier_normal'
    args.label_smoothing_rate = config.get('label_smoothing_rate', 0.0)
    executor = dicee.Execute(args)
    report = executor.start()
    path = f"{report['path_experiment_folder'].replace(f'{cwd}/Experiments/', '')}"
    return {
        'train_mrr': report['Train']['MRR'],
        'val_mrr': report['Val']['MRR'],
        'test_mrr': report['Test']['MRR'],
        'path': path,
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    gpus = 1 if torch.cuda.device_count() > 0 else 0
    config = {
        'dataset': myconfig.dataset,
        'gpus': gpus,
        'num_epochs': 10,
    }
    logging.info('Config: %s', config)
    metrics = trainable(config)
    logging.info('Result: %s', metrics)
