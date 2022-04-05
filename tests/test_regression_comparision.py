from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionAdaptE:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'DistMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
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
        args.num_folds_for_cv = None
        result_distMult = Execute(args).start()

        args = argparse_default([])
        args.model = 'AdaptE'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 300
        args.batch_size = 1024
        args.lr = 0.1
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
        args.num_folds_for_cv = None
        result_adapte = Execute(args).start()

        assert result_adapte['Train']['H@1'] >= result_distMult['Train']['H@1']
        assert result_adapte['Test']['H@1'] >= result_distMult['Test']['H@1']
        assert result_adapte['Val']['H@1'] >= result_distMult['Val']['H@1']

