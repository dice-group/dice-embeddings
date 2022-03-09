from continual_training import argparse_default as ct_argparse_default
from main import argparse_default as main_argparse_default
from core.executer import Execute, ContinuousExecute
import pytest
import argparse
import os
class TestRegressionCL:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = main_argparse_default([])
        args.model = 'QMult'
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
        assert 0.71 >= result['Val']['H@1'] >= 0.20
        assert os.path.isdir(result['path_experiment_folder'])
        args = ct_argparse_default([])
        args.path_experiment_folder = result['path_experiment_folder']
        ct_results = ContinuousExecute(args).start()
        assert ct_results['Train']['H@1'] >= result['Train']['H@1']
        assert ct_results['Val']['H@1'] >= result['Val']['H@1']
        assert ct_results['Test']['H@1'] >= result['Test']['H@1']
