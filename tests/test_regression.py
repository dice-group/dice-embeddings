from core.static_funcs import argparse_default
from core.executer import Execute
import sys


class TestDefaultParams:
    def test_distmult(self):
        args = argparse_default([])
        args.model = 'DistMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 10
        args.lr = .1
        args.embedding_dim=200
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        result = Execute(args).start()
        assert 0.48 >= result['Val']['H@1'] >= 0.41

    def test_shallom(self):
        args = argparse_default([])
        args.model = 'Shallom'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 10
        args.lr = .1
        args.embedding_dim=200
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        result = Execute(args).start()
        assert 0.29 >= result['Val']['H@1'] >= 0.22

    def test_krone(self):
        args = argparse_default([])
        args.model = 'KronE'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 10
        args.lr = .1
        args.entity_embedding_dim, args.rel_embedding_dim = 32, 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        result = Execute(args).start()
        assert 0.60 >= result['Val']['H@1'] >= 0.53

