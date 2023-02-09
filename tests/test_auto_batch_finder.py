from continual_training import argparse_default as ct_argparse_default
from main import argparse_default as main_argparse_default
from core.executer import Execute, ContinuousExecute
from core.knowledge_graph_embeddings import KGE
from core.knowledge_graph import KG
import pytest
import argparse
import os


class TestAutoBatchFinder:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_auto_batch_finder(self):
        args = main_argparse_default([])
        args.model = 'DistMult'
        args.scoring_technique = 'KvsSample'
        args.optim = 'Adam'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 100
        args.batch_size = 32
        args.lr = 0.1
        args.embedding_dim = 64
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.auto_batch_finder = True
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.backend = 'pandas'
        args.trainer = 'torchCPUTrainer'
        result_fast = Execute(args).start()

        args = main_argparse_default([])
        args.model = 'DistMult'
        args.scoring_technique = 'KvsSample'
        args.optim = 'Adam'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 100
        args.batch_size = 32
        args.lr = 0.1
        args.embedding_dim = 64
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.auto_batch_finder = False
        args.backend = 'pandas'
        args.trainer = 'torchCPUTrainer'
        result_slow = Execute(args).start()

        assert result_slow['Runtime'] > result_fast['Runtime']
