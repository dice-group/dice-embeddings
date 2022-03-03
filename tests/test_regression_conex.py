from main import argparse_default
from core.executer import Execute
import sys


class TestRegressionConEx:
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
        result = Execute(args).start()
        assert 0.11 >= result['Val']['H@1'] >= 0.09

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
        args.scoring_technique = '1vsAll'
        result = Execute(args).start()
        assert 0.82 >= result['Test']['H@1'] >= 0.79

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
        args.neg_ratio = 1
        result = Execute(args).start()
        assert 0.46 >= result['Test']['H@1'] >= .43
