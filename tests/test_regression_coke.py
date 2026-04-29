from dicee.executer import Execute
import pytest
from dicee.config import Namespace

import os
import torch
import random
import numpy as np

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# AllvsAll tests fail with "MKL_NUM_THREADS" = 2 ( or some other values) 
# due to numerical drift in multi-threaded CPU execution.
# Setting to 1 stabilizes the runs and ensures consistent metrics.

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TestRegressionCoKE:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 512
        args.lr = 0.01
        args.embedding_dim = 16
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.scoring_technique = 'KvsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.30 >= result['Val']['H@1'] >= 0.20
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 512
        args.lr = 0.01
        args.embedding_dim = 16
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.scoring_technique = '1vsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.50 >= result['Val']['H@1'] >= 0.30
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_all_vs_all(self):
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 512
        args.lr = 0.01
        args.embedding_dim = 16
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.scoring_technique = 'AllvsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.25 >= result['Val']['H@1'] >= 0.15
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 512
        args.lr = 0.01
        args.embedding_dim = 16
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.45 >= result['Train']['H@1'] >= 0.25
        assert 0.45 >= result['Test']['H@1'] >= 0.25
        assert 0.45 >= result['Val']['H@1'] >= 0.25