"""End-to-end regression coverage for adversarial_temperature / grouped_negative_sampling /
strict_negative_sampling (issue #446).

Before this file, the only coverage of this feature was the unit-level
routing test in tests/test_unit_base_model.py::TestAdversarialTemperatureLossFunction
(added in #441), which checks that BaseKGE.loss_function calls
grouped_adversarial_bce but never trains a real model through
Execute(args).start(). That's exactly the kind of gap that let the
loss_function dead-code bug (fixed in #441) go unnoticed for a while:
grouped/adversarial sampling looked like it worked (no errors, a loss
value came out) while silently using the wrong loss the whole time.

Validation for these options lives in dicee/static_preprocess_funcs.py;
the dataset is built by GroupedNegativeSamplingDataset
(dicee/dataset_classes/_negative_sampling.py), the loss by
grouped_adversarial_bce (dicee/models/sampled_loss.py).
"""
import pytest

from dicee.config import Namespace
from dicee.executer import Execute


def _base_args(**overrides):
    args = Namespace()
    args.model = "DistMult"
    args.dataset_dir = "KGs/UMLS"
    args.num_epochs = 5
    args.batch_size = 1024
    args.lr = 0.01
    args.embedding_dim = 32
    args.scoring_technique = "NegSample"
    args.neg_ratio = 2
    args.eval_model = "train_val_test"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestAdversarialTemperatureTraining:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_trains_and_produces_finite_mrr(self):
        result = Execute(_base_args(adversarial_temperature=1.0)).start()
        assert 0.0 <= result["Val"]["MRR"] <= 1.0
        assert 0.0 <= result["Test"]["MRR"] <= 1.0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_zero_temperature_uniform_weighting_trains(self):
        """temperature=0 takes grouped_adversarial_bce's uniform-weight branch
        (as opposed to the softmax-weighted branch for temperature > 0)."""
        result = Execute(_base_args(adversarial_temperature=0.0)).start()
        assert 0.0 <= result["Val"]["MRR"] <= 1.0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_trains_under_pl_trainer(self):
        result = Execute(_base_args(adversarial_temperature=1.0, trainer="PL")).start()
        assert 0.0 <= result["Val"]["MRR"] <= 1.0


class TestGroupedNegativeSamplingTraining:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_trains_and_produces_finite_mrr(self):
        result = Execute(_base_args(grouped_negative_sampling=True)).start()
        assert 0.0 <= result["Val"]["MRR"] <= 1.0
        assert 0.0 <= result["Test"]["MRR"] <= 1.0


class TestStrictNegativeSamplingTraining:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_trains_and_produces_finite_mrr(self):
        result = Execute(_base_args(strict_negative_sampling=True)).start()
        assert 0.0 <= result["Val"]["MRR"] <= 1.0
        assert 0.0 <= result["Test"]["MRR"] <= 1.0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_combined_with_adversarial_temperature(self):
        result = Execute(_base_args(strict_negative_sampling=True, adversarial_temperature=0.5)).start()
        assert 0.0 <= result["Val"]["MRR"] <= 1.0


class TestGroupedSamplingEdgeCases:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_neg_ratio_minimum_allowed_value_trains(self):
        result = Execute(_base_args(grouped_negative_sampling=True, neg_ratio=1)).start()
        assert 0.0 <= result["Val"]["MRR"] <= 1.0

    def test_neg_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="Grouped sampling requires neg_ratio"):
            Execute(_base_args(grouped_negative_sampling=True, neg_ratio=0)).start()

    def test_negative_adversarial_temperature_raises(self):
        with pytest.raises(ValueError, match="adversarial_temperature must be finite and nonnegative"):
            Execute(_base_args(adversarial_temperature=-1.0)).start()

    def test_nan_adversarial_temperature_raises(self):
        with pytest.raises(ValueError, match="adversarial_temperature must be finite and nonnegative"):
            Execute(_base_args(adversarial_temperature=float("nan"))).start()

    def test_infinite_adversarial_temperature_raises(self):
        with pytest.raises(ValueError, match="adversarial_temperature must be finite and nonnegative"):
            Execute(_base_args(adversarial_temperature=float("inf"))).start()

    def test_wrong_scoring_technique_raises(self):
        with pytest.raises(ValueError, match="Grouped/strict/adversarial sampling requires indexed NegSample"):
            Execute(_base_args(grouped_negative_sampling=True, scoring_technique="KvsAll")).start()

    def test_byte_pair_encoding_raises(self):
        with pytest.raises(ValueError, match="Grouped/strict/adversarial sampling requires indexed NegSample"):
            Execute(_base_args(grouped_negative_sampling=True, byte_pair_encoding=True)).start()

    def test_unsupported_trainer_raises(self):
        # torchDDP/torchFSDP fail earlier (require torchrun launch) before
        # reaching this validation; TP does not, and is otherwise unsupported.
        with pytest.raises(ValueError, match="Grouped sampling currently supports native CPU/single GPU and Lightning trainers"):
            Execute(_base_args(grouped_negative_sampling=True, trainer="TP")).start()
