from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionQmult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_sampe(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsSample'
        args.neg_ratio = 10
        args.test_mode = True
        args.eval = True
        args.eval_on_train = True
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()

        assert result['Train']['H@10'] >= result['Train']['H@3'] >= result['Train']['H@1']
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']
        assert result['Test']['H@10'] >= result['Test']['H@3'] >= result['Test']['H@1']
