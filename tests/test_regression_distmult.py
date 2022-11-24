from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionDistMult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'DistMult'
        args.optim = 'Adam'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.scoring_technique = 'KvsAll'
        args.normalization = 'LayerNorm'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.58 >= result['Val']['H@1'] >= 0.01

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'DistMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.scoring_technique = '1vsAll'
        args.normalization = 'LayerNorm'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.99 >= result['Train']['H@1'] >= 0.30
        assert 0.99 >= result['Test']['H@1'] >= 0.25
        assert 0.99 >= result['Val']['H@1'] >= 0.25

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'DistMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.normalization = 'LayerNorm'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.73 >= result['Train']['H@1'] >= 0.01
        assert 0.73 >= result['Test']['H@1'] >= 0.01
        assert 0.73 >= result['Val']['H@1'] >= 0.01
