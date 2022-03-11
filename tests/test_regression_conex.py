from main import argparse_default
from core.executer import Execute
import sys
import pytest

class TestRegressionConEx:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'ConEx'
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
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()
        assert 0.25 >= result['Train']['H@1'] >= 0.09
        assert 0.25 >= result['Val']['H@1'] >= 0.09
        assert 0.25 >= result['Test']['H@1'] >= 0.09

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'ConEx'
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
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.scoring_technique = '1vsAll'
        result = Execute(args).start()
        assert 0.75 >= result['Train']['H@1'] > 0.35
        assert 0.75 >= result['Val']['H@1'] >= 0.35
        assert 0.75 >= result['Test']['H@1'] >= 0.35

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'ConEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.eval = 1
        args.eval_on_train = 1
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.neg_ratio = 1
        result = Execute(args).start()
        assert 0.48 >= result['Test']['H@1'] >= .35
