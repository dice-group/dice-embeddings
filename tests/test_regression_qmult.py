from main import argparse_default
from core.executer import Execute
import sys
import pytest

class TestRegressionQmult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = True
        args.eval_on_train = True
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()
        assert 0.76 >= result['Val']['H@1'] >= 0.71
        assert 0.76 >= result['Test']['H@1'] >= 0.71

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = '1vsAll'
        args.eval = True
        args.eval_on_train = True
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()
        assert 0.74 >= result['Test']['H@1'] >= 0.72
        assert 0.75 >= result['Val']['H@1'] >= 0.72
        assert 0.95 >= result['Train']['H@1'] >= 0.72

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.eval = True
        args.eval_on_train = True
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()
        assert 0.53 >= result['Test']['H@1'] >= .50
        assert 0.55 >= result['Val']['H@1'] >= .50
