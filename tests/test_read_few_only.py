from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestReadFewOnly:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_kvsall(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train'
        args.sample_triples_ratio = None
        args.read_only_few = 10
        args.torch_trainer = 'DataParallelTrainer'
        report = Execute(args).start()
        # as we add negative triples
        assert report['num_train_triples'] == int(args.read_only_few * 2)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_1vsall(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train'
        args.read_only_few = 10
        args.sample_triples_ratio = None
        args.torch_trainer = 'DataParallelTrainer'
        report = Execute(args).start()
        # as we add negative triples
        assert report['num_train_triples'] == int(args.read_only_few * 2)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_neg_sampling(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train'
        args.read_only_few = 10
        args.sample_triples_ratio = None
        args.neg_ratio = 1
        args.torch_trainer = 'DataParallelTrainer'
        report = Execute(args).start()
        # as we add negative triples
        assert report['num_train_triples'] == args.read_only_few
