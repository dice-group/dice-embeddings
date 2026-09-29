"""Regression tests for the GNN encoder models."""

import pytest

from dicee.config import Namespace
from dicee.executer import Execute


class TestRegressionRGCN:

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_k_vs_all(self):
        args = Namespace()
        args.model = "RGCN"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.05
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.gnn_num_layers = 1
        args.gnn_num_bases = None
        args.scoring_technique = "KvsAll"
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert 0.85 >= result["Train"]["MRR"] >= 0.75
        assert 0.80 >= result["Val"]["MRR"] >= 0.70
        assert 0.80 >= result["Test"]["MRR"] >= 0.70

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_negative_sampling(self):
        args = Namespace()
        args.model = "RGCN"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.gnn_num_layers = 1
        args.gnn_num_bases = None
        args.scoring_technique = "NegSample"
        args.neg_ratio = 1
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert 0.80 >= result["Train"]["MRR"] >= 0.65
        assert 0.75 >= result["Val"]["MRR"] >= 0.60
        assert 0.75 >= result["Test"]["MRR"] >= 0.55


class TestRegressionGATv2:

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_k_vs_all(self):
        args = Namespace()
        args.model = "GATv2"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.05
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.gnn_num_layers = 1
        args.gnn_attn_heads = 2
        args.scoring_technique = "KvsAll"
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert 0.85 >= result["Train"]["MRR"] >= 0.75
        assert 0.85 >= result["Val"]["MRR"] >= 0.75
        assert 0.85 >= result["Test"]["MRR"] >= 0.70
