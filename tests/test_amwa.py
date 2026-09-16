"""Tests for AMWA (Adaptive Momentum Weight Averaging), issue #447.

SWA/SWAG/EMA/TWA are covered by tests/test_swa.py and ASWA by
tests/test_adaptive_swa.py; AMWA had no dedicated coverage of its own
before this file - it was only smoke-tested manually while reviewing
PR #440.

Mirrors that structure: an end-to-end Execute() run confirming
amwa.pt/amwa_history.json are written, direct unit tests of the momentum/
beta_n math (dicee/weight_averaging.py::AMWA), and a check that periodic
evaluation during AMWA training actually evaluates StableNet (trainer.wa_model)
rather than the live BaseNet.
"""
import json
import os
from unittest.mock import patch

import pytest
import torch

from dicee.config import Namespace
from dicee.executer import Execute
from dicee.weight_averaging import AMWA


def _amwa_args(**overrides):
    args = Namespace()
    args.model = "DistMult"
    args.dataset_dir = "KGs/UMLS"
    args.scoring_technique = "KvsAll"
    args.num_epochs = 4
    args.lr = 0.1
    args.embedding_dim = 16
    args.batch_size = 1024
    args.amwa = True
    args.amwa_start_epoch = 0
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestAMWAEndToEnd:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_writes_checkpoint_and_history(self):
        report = Execute(_amwa_args()).start()
        experiment_dir = report["path_experiment_folder"]

        assert os.path.exists(os.path.join(experiment_dir, "amwa.pt"))
        assert os.path.exists(os.path.join(experiment_dir, "amwa_history.json"))

        state_dict = torch.load(os.path.join(experiment_dir, "amwa.pt"), map_location="cpu")
        assert "entity_embeddings.weight" in state_dict

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_history_records_have_expected_fields(self):
        report = Execute(_amwa_args()).start()
        experiment_dir = report["path_experiment_folder"]
        with open(os.path.join(experiment_dir, "amwa_history.json")) as handle:
            history = json.load(handle)

        assert len(history) > 0
        expected_keys = {"epoch", "monitor", "maximize", "base_score", "stable_score",
                         "delta", "beta_n", "momentum", "base_weight"}
        for record in history:
            assert expected_keys.issubset(record.keys())
            assert 0.0 <= record["momentum"] <= 1.0
            assert record["base_weight"] == pytest.approx(1.0 - record["momentum"])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_trains_under_pl_trainer_too(self):
        report = Execute(_amwa_args(trainer="PL")).start()
        experiment_dir = report["path_experiment_folder"]
        assert os.path.exists(os.path.join(experiment_dir, "amwa.pt"))


class TestAMWAMomentumMath:
    """Direct unit tests of AMWA's momentum/beta_n computation - no training
    required, since these methods only depend on scalar history, not a
    running model or trainer."""

    def test_momentum_is_half_at_zero_delta(self):
        amwa = AMWA()
        # m(0) = 1 / (1 + exp(0)) = 0.5
        assert amwa._compute_momentum(delta=0.0, beta_n=1.0) == pytest.approx(0.5)

    def test_momentum_clips_negative_delta_to_zero(self):
        amwa = AMWA()
        # max(0, delta) means any negative delta behaves like delta=0.
        assert amwa._compute_momentum(delta=-5.0, beta_n=1.0) == pytest.approx(0.5)
        assert amwa._compute_momentum(delta=-0.01, beta_n=1.0) == pytest.approx(0.5)

    def test_momentum_increases_monotonically_with_positive_delta(self):
        amwa = AMWA()
        m_small = amwa._compute_momentum(delta=0.1, beta_n=1.0)
        m_large = amwa._compute_momentum(delta=10.0, beta_n=1.0)
        assert 0.5 < m_small < m_large < 1.0

    def test_momentum_approaches_one_for_large_scaled_gap(self):
        amwa = AMWA()
        m = amwa._compute_momentum(delta=1000.0, beta_n=1e-8)
        assert m == pytest.approx(1.0, abs=1e-6)

    def test_beta_n_uses_fixed_beta_when_provided(self):
        amwa = AMWA(beta=0.05)
        amwa.delta_history = [1.0, 2.0, 3.0, 100.0]  # should be ignored entirely
        assert amwa._compute_beta_n() == pytest.approx(0.05)

    def test_beta_n_fixed_beta_respects_floor(self):
        amwa = AMWA(beta=0.0, beta_floor=1e-8)
        assert amwa._compute_beta_n() == pytest.approx(1e-8)

    def test_beta_n_returns_init_when_history_empty(self):
        amwa = AMWA(beta=None, beta_init=0.42)
        assert amwa._compute_beta_n() == pytest.approx(0.42)

    def test_beta_n_returns_init_when_history_has_one_entry(self):
        amwa = AMWA(beta=None, beta_init=0.42)
        amwa.delta_history = [1.0]
        assert amwa._compute_beta_n() == pytest.approx(0.42)

    def test_beta_n_computes_std_of_history_within_window(self):
        amwa = AMWA(beta=None, beta_window=4)
        amwa.delta_history = [1.0, 2.0, 3.0, 4.0]
        expected = torch.std(torch.tensor([1.0, 2.0, 3.0, 4.0]), unbiased=False).item()
        assert amwa._compute_beta_n() == pytest.approx(expected)

    def test_beta_n_only_uses_most_recent_window(self):
        amwa = AMWA(beta=None, beta_window=3)
        # Only the last 3 entries (3, 4, 5) should be used, not the leading 100s.
        amwa.delta_history = [100.0, 100.0, 3.0, 4.0, 5.0]
        expected = torch.std(torch.tensor([3.0, 4.0, 5.0]), unbiased=False).item()
        assert amwa._compute_beta_n() == pytest.approx(expected)

    def test_beta_n_respects_floor_for_zero_variance_history(self):
        amwa = AMWA(beta=None, beta_floor=1e-6)
        amwa.delta_history = [5.0, 5.0, 5.0]
        assert amwa._compute_beta_n() == pytest.approx(1e-6)

    def test_compute_delta_maximize_true(self):
        amwa = AMWA(maximize=True)
        assert amwa._compute_delta(base_score=0.3, stable_score=0.5) == pytest.approx(0.2)
        assert amwa._compute_delta(base_score=0.5, stable_score=0.3) == pytest.approx(-0.2)

    def test_compute_delta_maximize_false(self):
        amwa = AMWA(maximize=False)
        assert amwa._compute_delta(base_score=0.3, stable_score=0.5) == pytest.approx(-0.2)
        assert amwa._compute_delta(base_score=0.5, stable_score=0.3) == pytest.approx(0.2)


class TestAMWAStableNetUpdate:
    """Direct unit tests of the StableNet EMA-style update, using a plain
    nn.Linear as a stand-in running_model (_update_stablenet only calls
    running_model.state_dict(), so a full KGE model isn't needed)."""

    def test_initialize_stablenet_copies_state_dict_to_cpu(self):
        amwa = AMWA()
        model = torch.nn.Linear(2, 2)
        amwa._initialize_stablenet(model)
        assert amwa.stable_state_dict is not None
        torch.testing.assert_close(amwa.stable_state_dict["weight"], model.weight.detach().cpu())
        assert amwa.stable_state_dict["weight"].device.type == "cpu"

    def test_update_stablenet_applies_momentum_formula(self):
        amwa = AMWA()
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        amwa._initialize_stablenet(model)
        stable_before = amwa.stable_state_dict["weight"].clone()

        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3.0, 3.0], [3.0, 3.0]]))
        momentum = 0.25
        amwa._update_stablenet(model, momentum)

        # w_stable <- m * w_stable + (1 - m) * w_base
        expected = momentum * stable_before + (1 - momentum) * model.weight.detach()
        torch.testing.assert_close(amwa.stable_state_dict["weight"], expected)

    def test_update_stablenet_momentum_one_keeps_stablenet_unchanged(self):
        amwa = AMWA()
        model = torch.nn.Linear(2, 2, bias=False)
        amwa._initialize_stablenet(model)
        stable_before = amwa.stable_state_dict["weight"].clone()

        with torch.no_grad():
            model.weight.copy_(torch.randn(2, 2))
        amwa._update_stablenet(model, momentum=1.0)
        torch.testing.assert_close(amwa.stable_state_dict["weight"], stable_before)

    def test_update_stablenet_momentum_zero_copies_base(self):
        amwa = AMWA()
        model = torch.nn.Linear(2, 2, bias=False)
        amwa._initialize_stablenet(model)

        with torch.no_grad():
            model.weight.copy_(torch.tensor([[7.0, 7.0], [7.0, 7.0]]))
        amwa._update_stablenet(model, momentum=0.0)
        torch.testing.assert_close(amwa.stable_state_dict["weight"], model.weight.detach())

    def test_update_stablenet_initializes_if_not_already(self):
        amwa = AMWA()
        model = torch.nn.Linear(2, 2, bias=False)
        assert amwa.stable_state_dict is None
        amwa._update_stablenet(model, momentum=0.5)
        assert amwa.stable_state_dict is not None


class TestAMWAPeriodicEvalUsesStableNet:
    """Confirms PeriodicEvalCallback evaluates/checkpoints AMWA's StableNet
    (via trainer.wa_model) during AMWA training, not the live BaseNet -
    the exact mechanism PeriodicEvalCallback's `if any(model.args.get(k) for
    k in (\"swa\", \"ema\", \"twa\", \"amwa\")): wa_model = getattr(trainer,
    \"wa_model\", None)` branch relies on (dicee/callbacks.py)."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_checkpointed_model_matches_stablenet_snapshot(self):
        captured = {}
        original_build_eval_model = AMWA._build_eval_model

        def spy_build_eval_model(self, running_model, state_dict=None):
            result = original_build_eval_model(self, running_model, state_dict)
            if state_dict is not None:
                # Overwritten each qualifying epoch; the last call per epoch
                # is the one AMWA assigns to trainer.wa_model.
                captured[self.current_epoch] = {k: v.clone() for k, v in result.state_dict().items()}
            return result

        with patch.object(AMWA, "_build_eval_model", spy_build_eval_model):
            report = Execute(_amwa_args(num_epochs=3, eval_every_n_epochs=1, save_every_n_epochs=True)).start()

        assert captured, "AMWA never built a StableNet eval model - periodic eval was never scheduled"

        experiment_dir = report["path_experiment_folder"]
        checkpoint_dir = os.path.join(experiment_dir, "models_n_epochs")
        checkpointed_epochs = sorted(int(f.split("_")[-1].split(".")[0])
                                     for f in os.listdir(checkpoint_dir) if f.startswith("model_at_epoch_"))
        assert checkpointed_epochs, "PeriodicEvalCallback never saved a checkpoint"

        # epoch_counter in PeriodicEvalCallback is 1-indexed per completed
        # epoch; AMWA's current_epoch is 0-indexed, so epoch_counter N
        # corresponds to AMWA's current_epoch N-1.
        for epoch_counter in checkpointed_epochs:
            amwa_epoch = epoch_counter - 1
            assert amwa_epoch in captured, f"No StableNet snapshot captured for epoch {amwa_epoch}"
            checkpoint_state = torch.load(
                os.path.join(checkpoint_dir, f"model_at_epoch_{epoch_counter}.pt"), map_location="cpu",
            )
            for key, stable_tensor in captured[amwa_epoch].items():
                torch.testing.assert_close(checkpoint_state[key], stable_tensor)
