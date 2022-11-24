from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionDistMult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'DistMult'
        args.path_dataset_folder = 'KGs/WN18RR'
        args.optim = 'Adam'
        args.num_epochs = 0
        args.batch_size = 32
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.seed_for_computation = 1
        args.min_freq_for_vocab = None
        args.trainer = 'torchCPUTrainer'
        args.normalization = 'LayerNorm'
        args.scoring_technique = 'KvsAll'
        result = Execute(args).start()
        assert 0.03 >= result['Val']['MRR'] >= 0.00024
        assert 0.03 >= result['Test']['MRR'] >= 0.00035

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_neg_sample(self):
        args = argparse_default([])
        args.model = 'DistMult'
        args.path_dataset_folder = 'KGs/WN18RR'
        args.optim = 'Adam'
        args.num_epochs = 0
        args.batch_size = 32
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.seed_for_computation = 1
        args.min_freq_for_vocab = None
        args.normalization = 'LayerNorm'
        args.trainer = 'torchCPUTrainer'
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        result = Execute(args).start()
        assert 0.03 >= result['Test']['MRR'] >= 0.0001
