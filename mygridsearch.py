#!/usr/bin/env python3
from mytraining import trainable
from ray import tune
import logging
import myconfig
import ray
import torch

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('CUDA devices: %d', torch.cuda.device_count())
    ray.init(
        dashboard_host=myconfig.dashboard_host,
    )
    gpus = 1 if torch.cuda.device_count() > 0 else 0
    metric = 'val_mrr',
    mode = 'max'
    param_space = myconfig.param_space | {'dataset': myconfig.dataset, 'gpus': gpus}
    tune_config = tune.TuneConfig(
        max_concurrent_trials=myconfig.max_concurrent_trials,
        metric=metric,
        mode=mode,
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, {'gpu': gpus / myconfig.proc_per_gpu}) if gpus else trainable,
        param_space=param_space,
        tune_config=tune_config,
    )
    results = tuner.fit()
    best_results = results.get_best_result(metric=metric, mode=mode)
    logging.info('Parameters: %s', param_space)
    logging.info('Best results: %s', best_results)
    logging.info('Best config: %s', best_results.config)
