from ray import tune

dashboard_host = '127.0.0.1'
#dashboard_host = '0.0.0.0'

dataset = 'UMLS'
#dataset = 'KINSHIP'
#dataset = 'FB15k-237'
#dataset = 'YAGO3-10'

# Grid search
param_space = {
    'embedding_dim': tune.grid_search([x * 16 for x in [8]]),
    'neg_ratio': tune.grid_search([0, 1, 5, 10]),
    'weight_decay': tune.grid_search([0, 0.01, 0.1]),
    'input_dropout_rate': tune.grid_search([0, 0.1, 0.2]),
    'label_smoothing_rate': tune.grid_search([0, 0.05, 0.1]),
}

# Bayesian
#param_space = {
#    'complex_embedding_dim': tune.uniform(2, 8),
#    'neg_ratio': tune.uniform(0, 10),
#    'weight_decay': tune.uniform(0, 0.1),
#    'input_dropout_rate': tune.uniform(0, 0.4),
#    'label_smoothing_rate': tune.uniform(0, 0.4),
#}

proc_per_gpu = 1

max_concurrent_trials = 4

num_samples = 100
