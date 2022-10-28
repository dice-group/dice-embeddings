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
        args.num_epochs = 100
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
        args.normalization = 'LayerNorm'
        args.torch_trainer = 'DataParallelTrainer'
        result = Execute(args).start()
        assert 1.00 >= result['Train']['H@1'] >= 0.85
        assert 0.85 >= result['Val']['H@1'] >= 0.70
        assert 0.85 >= result['Test']['H@1'] >= 0.70

        args.scoring_technique = 'KvsAll'
        result2 = Execute(args).start()
        assert 1.00 >= result2['Train']['H@1'] >= 0.85
        assert 0.85 >= result2['Val']['H@1'] >= 0.70
        assert 0.85 >= result2['Test']['H@1'] >= 0.70

        assert result2['Test']['H@1'] == result['Test']['H@1']
