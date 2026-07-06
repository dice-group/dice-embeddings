"""Unit tests for dicee/abstracts.py.

Covers:
- AbstractTrainer callback dispatch (on_fit_start/end, on_train_epoch_end/start,
  on_train_batch_end, save_checkpoint)
- InteractiveQueryDecomposition fuzzy-logic operators (t_norm, t_conorm, negnorm,
  tensor_t_norm)
- AbstractCallback default hook implementations (no-op, no exception)
"""

import os
import tempfile
import types

import pytest
import torch

from dicee.abstracts import AbstractCallback, AbstractTrainer, InteractiveQueryDecomposition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trainer_args(seed: int = 0):
    """Return a minimal Namespace-like object with a random_seed attribute."""
    ns = types.SimpleNamespace()
    ns.random_seed = seed
    return ns


class _RecordingCallback(AbstractCallback):
    """Concrete callback that records which hooks were called."""

    def __init__(self):
        super().__init__()
        self.called = []

    def on_fit_start(self, trainer, model):
        self.called.append("on_fit_start")

    def on_fit_end(self, *args, **kwargs):
        self.called.append("on_fit_end")

    def on_train_epoch_start(self, *args, **kwargs):
        self.called.append("on_train_epoch_start")

    def on_train_epoch_end(self, trainer, model):
        self.called.append("on_train_epoch_end")

    def on_train_batch_end(self, *args, **kwargs):
        self.called.append("on_train_batch_end")


# ---------------------------------------------------------------------------
# AbstractTrainer
# ---------------------------------------------------------------------------

class TestAbstractTrainer:
    """Tests for the callback-dispatch mechanism in AbstractTrainer."""

    def _make_trainer(self, callbacks):
        return AbstractTrainer(args=_make_trainer_args(), callbacks=callbacks)

    def test_on_fit_start_dispatches_to_all_callbacks(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        trainer = self._make_trainer([cb1, cb2])
        trainer.on_fit_start("trainer_sentinel", "model_sentinel")
        assert cb1.called == ["on_fit_start"]
        assert cb2.called == ["on_fit_start"]

    def test_on_fit_end_dispatches_to_all_callbacks(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        trainer = self._make_trainer([cb1, cb2])
        trainer.on_fit_end()
        assert cb1.called == ["on_fit_end"]
        assert cb2.called == ["on_fit_end"]

    def test_on_train_epoch_start_dispatches_to_all_callbacks(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        trainer = self._make_trainer([cb1, cb2])
        trainer.on_train_epoch_start()
        assert cb1.called == ["on_train_epoch_start"]
        assert cb2.called == ["on_train_epoch_start"]

    def test_on_train_epoch_end_dispatches_to_all_callbacks(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        trainer = self._make_trainer([cb1, cb2])
        trainer.on_train_epoch_end("trainer_sentinel", "model_sentinel")
        assert cb1.called == ["on_train_epoch_end"]
        assert cb2.called == ["on_train_epoch_end"]

    def test_on_train_batch_end_dispatches_to_all_callbacks(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        trainer = self._make_trainer([cb1, cb2])
        trainer.on_train_batch_end()
        assert cb1.called == ["on_train_batch_end"]
        assert cb2.called == ["on_train_batch_end"]

    def test_dispatch_order_preserves_callback_list_order(self):
        """Callbacks are invoked in registration order."""
        order = []

        class _OrderedCB(AbstractCallback):
            def __init__(self, tag):
                super().__init__()
                self.tag = tag

            def on_fit_start(self, trainer, model):
                order.append(self.tag)

        trainer = self._make_trainer([_OrderedCB("first"), _OrderedCB("second"), _OrderedCB("third")])
        trainer.on_fit_start(None, None)
        assert order == ["first", "second", "third"]

    def test_empty_callback_list_runs_without_error(self):
        trainer = self._make_trainer([])
        trainer.on_fit_start(None, None)
        trainer.on_fit_end()
        trainer.on_train_epoch_start()
        trainer.on_train_epoch_end(None, None)
        trainer.on_train_batch_end()

    def test_save_checkpoint_writes_state_dict(self):
        """save_checkpoint writes a loadable state_dict file."""
        model = torch.nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            AbstractTrainer.save_checkpoint(path, model)
            assert os.path.isfile(path), "Checkpoint file was not created"
            loaded = torch.load(path, weights_only=True)
            # Must have the same keys as the original state_dict
            assert set(loaded.keys()) == set(model.state_dict().keys())

    def test_trainer_initial_rank_attributes(self):
        """Default rank/world-size attributes are correct for single-process training."""
        trainer = self._make_trainer([])
        assert trainer.is_global_zero is True
        assert trainer.global_rank == 0
        assert trainer.local_rank == 0


# ---------------------------------------------------------------------------
# AbstractCallback defaults
# ---------------------------------------------------------------------------

class TestAbstractCallbackDefaults:
    """AbstractCallback default hooks must be no-ops (return None, no exception)."""

    def test_default_on_fit_start_returns_none(self):
        cb = AbstractCallback()
        result = cb.on_fit_start("trainer", "model")
        # Default returns None (return statement without value)
        assert result is None

    def test_default_on_fit_end_returns_none(self):
        cb = AbstractCallback()
        assert cb.on_fit_end() is None

    def test_default_on_train_epoch_end_returns_none(self):
        cb = AbstractCallback()
        assert cb.on_train_epoch_end("trainer", "model") is None

    def test_default_on_train_batch_end_returns_none(self):
        cb = AbstractCallback()
        assert cb.on_train_batch_end() is None

    def test_default_on_init_start_returns_none(self):
        cb = AbstractCallback()
        assert cb.on_init_start() is None

    def test_default_on_init_end_returns_none(self):
        cb = AbstractCallback()
        assert cb.on_init_end() is None


# ---------------------------------------------------------------------------
# InteractiveQueryDecomposition — fuzzy operators
# ---------------------------------------------------------------------------

class _ConcreteQD(InteractiveQueryDecomposition):
    """Minimal concrete subclass so we can instantiate the mixin."""
    pass


class TestInteractiveQueryDecomposition:
    """Tests for the T-norm / T-conorm / negation-norm operators."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.qd = _ConcreteQD()
        # Simple 1-D score tensors with values in [0, 1]
        self.a = torch.tensor([0.9, 0.4, 0.1])
        self.b = torch.tensor([0.6, 0.7, 0.8])

    # --- t_norm ---

    def test_t_norm_min_returns_element_wise_min(self):
        result = self.qd.t_norm(self.a, self.b, tnorm="min")
        expected = torch.min(self.a, self.b)
        assert torch.allclose(result, expected)

    def test_t_norm_prod_returns_element_wise_product(self):
        result = self.qd.t_norm(self.a, self.b, tnorm="prod")
        expected = self.a * self.b
        assert torch.allclose(result, expected)

    def test_t_norm_min_bounds(self):
        """Min T-norm result must be <= both operands."""
        result = self.qd.t_norm(self.a, self.b, tnorm="min")
        assert torch.all(result <= self.a)
        assert torch.all(result <= self.b)

    def test_t_norm_prod_bounds(self):
        """Product T-norm result must be <= both operands (when inputs in [0,1])."""
        result = self.qd.t_norm(self.a, self.b, tnorm="prod")
        assert torch.all(result <= self.a)
        assert torch.all(result <= self.b)

    # --- t_conorm ---

    def test_t_conorm_min_returns_element_wise_max(self):
        result = self.qd.t_conorm(self.a, self.b, tconorm="min")
        expected = torch.max(self.a, self.b)
        assert torch.allclose(result, expected)

    def test_t_conorm_prod_returns_probabilistic_sum(self):
        result = self.qd.t_conorm(self.a, self.b, tconorm="prod")
        expected = self.a + self.b - self.a * self.b
        assert torch.allclose(result, expected)

    def test_t_conorm_min_bounds(self):
        """Max T-conorm result must be >= both operands."""
        result = self.qd.t_conorm(self.a, self.b, tconorm="min")
        assert torch.all(result >= self.a)
        assert torch.all(result >= self.b)

    def test_t_conorm_result_in_unit_interval(self):
        """T-conorm of values in [0,1] must stay in [0,1]."""
        result = self.qd.t_conorm(self.a, self.b, tconorm="prod")
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    # --- negnorm ---

    def test_negnorm_standard_is_complement(self):
        result = self.qd.negnorm(self.a, lambda_=1.0, neg_norm="standard")
        expected = 1 - self.a
        assert torch.allclose(result, expected)

    def test_negnorm_standard_sums_to_one(self):
        """Standard negation: x + neg(x) == 1."""
        result = self.qd.negnorm(self.a, lambda_=1.0, neg_norm="standard")
        assert torch.allclose(self.a + result, torch.ones_like(self.a))

    def test_negnorm_sugeno_at_lambda_zero_equals_standard(self):
        """Sugeno with lambda=0 reduces to the standard complement."""
        result = self.qd.negnorm(self.a, lambda_=0.0, neg_norm="sugeno")
        expected = 1 - self.a
        assert torch.allclose(result, expected, atol=1e-6)

    def test_negnorm_sugeno_result_in_unit_interval(self):
        result = self.qd.negnorm(self.a, lambda_=2.0, neg_norm="sugeno")
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_negnorm_yager_at_lambda_one_equals_standard(self):
        """Yager with lambda=1 reduces to the standard complement."""
        result = self.qd.negnorm(self.a, lambda_=1.0, neg_norm="yager")
        expected = 1 - self.a
        assert torch.allclose(result, expected, atol=1e-6)

    # --- tensor_t_norm ---

    def test_tensor_t_norm_min_reduces_over_first_dim(self):
        scores = torch.tensor([[0.9, 0.3, 0.6],
                                [0.4, 0.8, 0.2]])  # (2, 3) — 2 hops, 3 entities
        result = self.qd.tensor_t_norm(scores, tnorm="min")
        # torch.min with dim=0 returns a named tuple (values, indices)
        expected_values = torch.min(scores, dim=0).values
        assert torch.allclose(result.values, expected_values)

    def test_tensor_t_norm_unknown_raises(self):
        scores = torch.rand(2, 5)
        with pytest.raises(NotImplementedError):
            self.qd.tensor_t_norm(scores, tnorm="unknown")
