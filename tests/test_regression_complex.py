from main import argparse_default
from core.executer import Execute
import sys
import pytest

class TestRegressionComplEx:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'ComplEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 100
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval = 1
        args.eval_on_train = 1
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.scoring_technique = 'KvsAll'
        result = Execute(args).start()
        assert 0.70 >= result['Train']['H@1'] >= 0.31
        assert 0.45 >= result['Val']['H@1'] >= 0.31
        assert 0.45 >= result['Test']['H@1'] >= 0.31

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'ComplEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 100
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.eval = 1
        args.eval_on_train = 1
        args.scoring_technique = '1vsAll'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['H@1'] >= 0.75
        assert 0.87 >= result['Val']['H@1'] >= 0.50
        assert 0.87 >= result['Test']['H@1'] >= 0.50

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'ComplEx'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 100
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.eval = 1
        args.eval_on_train = 1
        args.sample_triples_ratio = None
        args.read_only_few = None
        result = Execute(args).start()
        assert 0.66 >= result['Train']['H@1'] >= .20
        assert 0.55 >= result['Val']['H@1'] >= .05
        assert 0.55 >= result['Test']['H@1'] >= .05
