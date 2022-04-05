from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionConvO:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'ConvO'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = 1
        args.eval_on_train = 1
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        assert 0.61 >= result['Train']['H@1'] >= 0.09
        assert 0.30 >= result['Val']['H@1'] >= 0.09
        assert 0.30 >= result['Test']['H@1'] >= 0.09

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'ConvO'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval = 1
        args.eval_on_train = 1
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.scoring_technique = '1vsAll'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['H@1'] >= 0.1
        assert 0.75 >= result['Val']['H@1'] >= 0.1
        assert 0.75 >= result['Test']['H@1'] >= 0.1

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'ConvO'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.eval = 1
        args.eval_on_train = 1
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        assert 0.61 >= result['Train']['H@1'] >= 0.05
        assert 0.30 >= result['Val']['H@1'] >= 0.04
        assert 0.30 >= result['Test']['H@1'] >= 0.038
