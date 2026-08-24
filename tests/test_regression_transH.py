"""Regression tests for the TransH model."""

import pytest
from dicee.executer import Execute
from dicee.config import Namespace


class TestRegressionTransH:

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_k_vs_all(self):
        args = Namespace()
        args.model = "TransH"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = "KvsAll"
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert 0.8 >= result["Train"]["MRR"] >= 0.7
        assert 0.75 >= result["Val"]["MRR"] >= 0.65
        assert 0.75 >= result["Test"]["MRR"] >= 0.65

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_neg_sample(self):
        args = Namespace()
        args.model = "TransH"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = "NegSample"
        args.neg_ratio = 10
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert 0.75 >= result["Train"]["MRR"] >= 0.65
        assert 0.75 >= result["Val"]["MRR"] >= 0.65
        assert 0.75 >= result["Test"]["MRR"] >= 0.6


    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_neg_sample_with_layer_norm(self):
        args = Namespace()
        args.model = "TransH"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = "NegSample"
        args.neg_ratio = 10
        args.normalization = "LayerNorm"
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert 0.85 >= result["Val"]["MRR"] >= 0.7
        assert 0.85 >= result["Test"]["MRR"] >= 0.7

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_kvs_sample(self):
        args = Namespace()
        args.model = "TransH"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = "KvsSample"
        args.neg_ratio = 10
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert 0.8 >= result["Train"]["MRR"] >= 0.7
        assert 0.75 >= result["Val"]["MRR"] >= 0.65
        assert 0.75 >= result["Test"]["MRR"] >= 0.65
