"""Unit tests for dicee/callbacks.py.

Covers:
- AccumulateEpochLossCallback.on_fit_end writes epoch_losses.csv
- PrintCallback on_fit_start / on_fit_end run without error
- KGESaveCallback constructor default interval calculation
- KGESaveCallback.on_epoch_end increments epoch_counter and saves checkpoint
  at the configured interval
"""

import os
import tempfile
import types

import pandas as pd
import pytest
import torch

from dicee.callbacks import AccumulateEpochLossCallback, KGESaveCallback, PrintCallback
from dicee.models.real import DistMult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_with_loss_history(losses):
    """Create a minimal mock model object with a loss_history attribute."""
    model = types.SimpleNamespace()
    model.loss_history = list(losses)
    return model


def _make_trainer():
    return types.SimpleNamespace()


# ---------------------------------------------------------------------------
# AccumulateEpochLossCallback
# ---------------------------------------------------------------------------

class TestAccumulateEpochLossCallback:

    def test_on_fit_end_creates_csv_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = AccumulateEpochLossCallback(path=tmp)
            model = _make_model_with_loss_history([0.8, 0.5, 0.3])
            cb.on_fit_end(trainer=_make_trainer(), model=model)
            csv_path = os.path.join(tmp, "epoch_losses.csv")
            assert os.path.isfile(csv_path), "epoch_losses.csv was not created"

    def test_on_fit_end_csv_has_correct_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = AccumulateEpochLossCallback(path=tmp)
            losses = [0.9, 0.6, 0.4, 0.2]
            model = _make_model_with_loss_history(losses)
            cb.on_fit_end(trainer=_make_trainer(), model=model)
            df = pd.read_csv(os.path.join(tmp, "epoch_losses.csv"), index_col=0)
            assert "EpochLoss" in df.columns

    def test_on_fit_end_csv_has_correct_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = AccumulateEpochLossCallback(path=tmp)
            losses = [1.0, 0.8, 0.5, 0.2, 0.1]
            model = _make_model_with_loss_history(losses)
            cb.on_fit_end(trainer=_make_trainer(), model=model)
            df = pd.read_csv(os.path.join(tmp, "epoch_losses.csv"), index_col=0)
            assert len(df) == len(losses)

    def test_on_fit_end_csv_values_match_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = AccumulateEpochLossCallback(path=tmp)
            losses = [0.7, 0.4, 0.15]
            model = _make_model_with_loss_history(losses)
            cb.on_fit_end(trainer=_make_trainer(), model=model)
            df = pd.read_csv(os.path.join(tmp, "epoch_losses.csv"), index_col=0)
            assert list(df["EpochLoss"]) == pytest.approx(losses)

    def test_on_fit_end_empty_history_writes_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = AccumulateEpochLossCallback(path=tmp)
            model = _make_model_with_loss_history([])
            cb.on_fit_end(trainer=_make_trainer(), model=model)
            df = pd.read_csv(os.path.join(tmp, "epoch_losses.csv"), index_col=0)
            assert len(df) == 0


# ---------------------------------------------------------------------------
# PrintCallback
# ---------------------------------------------------------------------------

class TestPrintCallback:

    def test_on_fit_start_runs_without_error(self, capsys):
        cb = PrintCallback()
        cb.on_fit_start(trainer=_make_trainer(), pl_module=types.SimpleNamespace())
        captured = capsys.readouterr()
        assert "Training is starting" in captured.out

    def test_on_fit_end_prints_runtime(self, capsys):
        cb = PrintCallback()
        cb.on_fit_end(trainer=_make_trainer(), pl_module=types.SimpleNamespace())
        captured = capsys.readouterr()
        assert "Training Runtime" in captured.out

    def test_on_train_batch_end_returns_none(self):
        cb = PrintCallback()
        assert cb.on_train_batch_end() is None

    def test_on_train_epoch_end_returns_none(self):
        cb = PrintCallback()
        assert cb.on_train_epoch_end() is None


# ---------------------------------------------------------------------------
# KGESaveCallback
# ---------------------------------------------------------------------------

class TestKGESaveCallback:

    def test_default_interval_is_half_max_epochs(self):
        cb = KGESaveCallback(every_x_epoch=None, max_epochs=20, path="/tmp")
        assert cb.every_x_epoch == 10

    def test_default_interval_minimum_is_one(self):
        """When max_epochs=1, default interval must be 1 (not 0)."""
        cb = KGESaveCallback(every_x_epoch=None, max_epochs=1, path="/tmp")
        assert cb.every_x_epoch >= 1

    def test_explicit_interval_is_respected(self):
        cb = KGESaveCallback(every_x_epoch=5, max_epochs=20, path="/tmp")
        assert cb.every_x_epoch == 5

    def test_epoch_counter_starts_at_zero(self):
        cb = KGESaveCallback(every_x_epoch=5, max_epochs=20, path="/tmp")
        assert cb.epoch_counter == 0

    def test_on_epoch_end_increments_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = KGESaveCallback(every_x_epoch=10, max_epochs=20, path=tmp)
            model = torch.nn.Linear(2, 2)
            for _ in range(3):
                cb.on_epoch_end(model=model, trainer=_make_trainer())
            assert cb.epoch_counter == 3

    def test_on_epoch_end_saves_checkpoint_at_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            every_x = 2
            cb = KGESaveCallback(every_x_epoch=every_x, max_epochs=10, path=tmp)
            # save_checkpoint_model requires a BaseKGE subclass
            model = DistMult(dict(
                model="DistMult", embedding_dim=8, num_entities=10, num_relations=4,
                learning_rate=0.01, optim="Adam", scoring_technique="KvsAll",
                input_dropout_rate=0.0, hidden_dropout_rate=0.0, normalization=None,
                init_param=None, byte_pair_encoding=False,
            ))
            # Run enough epochs to trigger a save (epoch_counter must be > 1 and divisible)
            for _ in range(5):
                cb.on_epoch_end(model=model, trainer=_make_trainer())
            # A checkpoint file should exist after epoch 2 (counter = 2, 2 % 2 == 0 and 2 > 1)
            pt_files = [f for f in os.listdir(tmp) if f.endswith(".pt")]
            assert len(pt_files) >= 1, (
                f"Expected at least one checkpoint file after 5 epochs (every_x={every_x}), found: {pt_files}"
            )

    def test_no_checkpoint_before_epoch_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = KGESaveCallback(every_x_epoch=1, max_epochs=5, path=tmp)
            model = torch.nn.Linear(2, 2)
            # Only call once: counter goes 0 → 1, condition is counter > 1 which is False
            cb.on_epoch_end(model=model, trainer=_make_trainer())
            pt_files = [f for f in os.listdir(tmp) if f.endswith(".pt")]
            assert len(pt_files) == 0, "Should not save at epoch 1 (counter=1, condition is > 1)"
