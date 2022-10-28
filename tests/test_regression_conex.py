from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionConEx:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'ConEx'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 20
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval = True
        args.eval_on_train = True
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.normalization='LayerNorm'
        args.torch_trainer = 'DataParallelTrainer'
        result = Execute(args).start()
        assert 0.46 >= result['Train']['H@1'] >= 0.33
        assert 0.46 >= result['Val']['H@1'] >= 0.33
        assert 0.46 >= result['Test']['H@1'] >= 0.33

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'ConEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval = True
        args.eval_on_train = True
        args.test_mode = True
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.normalization='LayerNorm'
        args.scoring_technique = '1vsAll'
        args.torch_trainer = 'DataParallelTrainer'
        result = Execute(args).start()
        assert 0.75 >= result['Train']['H@1'] > 0.32
        assert 0.75 >= result['Val']['H@1'] >= 0.22
        assert 0.75 >= result['Test']['H@1'] >= 0.22

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'ConEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.eval = True
        args.eval_on_train = True
        args.test_mode = True
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.read_only_few = None
        args.neg_ratio = 1
        args.normalization='LayerNorm'
        args.torch_trainer = 'DataParallelTrainer'
        result = Execute(args).start()
        assert 0.75 >= result['Train']['H@1'] >= .38
        assert 0.67 >= result['Val']['H@1'] >= .30
        assert 0.68 >= result['Test']['H@1'] >= .30
