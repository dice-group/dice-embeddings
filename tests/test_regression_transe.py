"""Regression tests for the TransE model.

TransE has no dedicated regression test file; existing coverage only
appears inside test_execute_start.py which verifies default parameters
for many models together without tight metric bounds.

These tests verify:
- NegSample scoring produces non-trivial link-prediction results
- KvsAll scoring produces non-trivial link-prediction results
- H@k ordering invariant: H@10 >= H@3 >= H@1
"""

import pytest
from dicee.executer import Execute
from dicee.config import Namespace


class TestRegressionTransE:

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_neg_sample(self):
        args = Namespace()
        args.model = "TransE"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = "NegSample"
        args.neg_ratio = 10
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert result["Train"]["MRR"] > 0.0
        assert result["Val"]["MRR"] > 0.0
        assert result["Test"]["MRR"] > 0.0

        for split in ("Train", "Val", "Test"):
            assert result[split]["H@10"] >= result[split]["H@3"] >= result[split]["H@1"], (
                f"TransE {split}: H@k ordering violated: "
                f"H@1={result[split]['H@1']}, H@3={result[split]['H@3']}, H@10={result[split]['H@10']}"
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_k_vs_all(self):
        args = Namespace()
        args.model = "TransE"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = "KvsAll"
        args.eval_model = "train_val_test"
        args.trainer = "torchCPUTrainer"
        result = Execute(args).start()

        assert result["Train"]["MRR"] > 0.0
        assert result["Val"]["MRR"] > 0.0
        assert result["Test"]["MRR"] > 0.0

        for split in ("Train", "Val", "Test"):
            assert result[split]["H@10"] >= result[split]["H@3"] >= result[split]["H@1"], (
                f"TransE KvsAll {split}: H@k ordering violated"
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_neg_sample_with_layer_norm(self):
        args = Namespace()
        args.model = "TransE"
        args.dataset_dir = "KGs/UMLS"
        args.optim = "Adam"
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.01
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

        assert result["Val"]["MRR"] > 0.0
        assert result["Test"]["MRR"] > 0.0
