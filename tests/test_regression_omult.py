from main import argparse_default
from core.executer import Execute
import sys


class TestRegressionOmult:
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'OMult'
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
        assert 0.71 >= result['Val']['H@1'] >= 0.24

    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'OMult'
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
        assert 0.72 >= result['Test']['H@1'] >= 0.64

    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'OMult'
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
        assert 0.009 >= result['Test']['H@1'] >= .0001
