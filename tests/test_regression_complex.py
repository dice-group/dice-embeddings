from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionComplEx:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'ComplEx'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.normalization='LayerNorm'
        args.torch_trainer = 'DataParallelTrainer'
        result = Execute(args).start()
        assert 0.84 >= result['Train']['H@1'] >= 0.75
        assert 0.84 >= result['Val']['H@1'] >= 0.60
        assert 0.84 >= result['Test']['H@1'] >= 0.60

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'ComplEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.eval = 'train_val_test'
        args.normalization='LayerNorm'
        args.torch_trainer = 'DataParallelTrainer'
        args.scoring_technique = '1vsAll'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['H@1'] >= 0.30
        assert 0.87 >= result['Val']['H@1'] >= 0.15
        assert 0.87 >= result['Test']['H@1'] >= 0.15

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'ComplEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.normalization='LayerNorm'
        args.optim = 'Adam'
        args.neg_ratio = 1
        args.eval = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.torch_trainer = None
        result = Execute(args).start()
        assert 0.76 >= result['Train']['H@1'] >= .65
        assert 0.71 >= result['Val']['H@1'] >= .57
        assert 0.71 >= result['Test']['H@1'] >= .53
