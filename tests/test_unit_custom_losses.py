"""Unit tests for dicee/losses/custom_losses.py (issue #445).

Covers every loss class's forward pass for a finite-scalar result and
gradient flow, plus edge cases (all-positive/all-negative batches, batch
size 1, boundary hyperparameters) and, where the underlying math is
independently verifiable, a correctness check of the class's documented
behavior (see the docstrings added in #444).

WeightedBCELoss has additional coverage in
tests/test_unit_base_model.py::TestWeightedBCELoss (it's also exercised
there as part of BaseKGE.loss_function's current_epoch wiring).
"""
import pytest
import torch

from dicee.losses.custom_losses import (
    ACLS,
    AdaptiveLabelRelaxationLoss,
    AdaptiveLabelSmoothingLoss,
    AggregatedLSandLR,
    CombinedAdaptiveLSandAdaptiveLR,
    CombinedLSandLR,
    ConfidenceBasedAdaptiveLabelRelaxationLoss,
    DefaultBCELoss,
    LabelRelaxationLoss,
    LabelSmoothingLoss,
    WeightedBCELoss,
)


def _binary_batch(batch_size=8, num_classes=10, requires_grad=True):
    pred = torch.randn(batch_size, num_classes, requires_grad=requires_grad)
    target = torch.zeros(batch_size, num_classes)
    target[:, 0] = 1.0
    return pred, target


class TestDefaultBCELoss:
    def test_matches_plain_bce_with_logits(self):
        pred, target = _binary_batch()
        expected = torch.nn.BCEWithLogitsLoss()(pred, target)
        actual = DefaultBCELoss()(pred, target)
        torch.testing.assert_close(actual, expected)

    def test_gradient_flows(self):
        pred, target = _binary_batch()
        loss = DefaultBCELoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_batch_size_one(self):
        pred, target = _binary_batch(batch_size=1)
        loss = DefaultBCELoss()(pred, target)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)


class TestWeightedBCELoss:
    """Minimal smoke coverage; see test_unit_base_model.py::TestWeightedBCELoss
    for the current_epoch/loss_fn wiring tests."""

    def test_batch_size_one(self):
        pred, target = _binary_batch(batch_size=1)
        loss = WeightedBCELoss()(pred, target, current_epoch=0)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_all_negative_target(self):
        pred, target = _binary_batch()
        target.zero_()
        loss = WeightedBCELoss()(pred, target, current_epoch=0)
        assert torch.isfinite(loss)


class TestLabelSmoothingLoss:
    """LabelSmoothingLoss currently does not apply smoothing at all (#453) -
    these tests document that known-buggy current behavior so a fix is
    forced to update them rather than silently changing semantics."""

    @pytest.mark.parametrize("smoothness_ratio", [0.0, 0.5, 1.0])
    def test_currently_ignores_smoothness_ratio(self, smoothness_ratio):
        pred, target = _binary_batch(requires_grad=False)
        expected = torch.nn.BCEWithLogitsLoss()(pred, target)
        actual = LabelSmoothingLoss(smoothness_ratio=smoothness_ratio)(pred, target)
        torch.testing.assert_close(actual, expected)

    def test_gradient_flows(self):
        pred, target = _binary_batch()
        loss = LabelSmoothingLoss(smoothness_ratio=0.1)(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestAdaptiveLabelSmoothingLoss:
    def test_finite_and_gradient_flows(self):
        pred, target = _binary_batch()
        loss = AdaptiveLabelSmoothingLoss()(pred, target, current_epoch=0)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None

    def test_smoothing_factor_grows_when_loss_increases(self):
        criterion = AdaptiveLabelSmoothingLoss(initial_smoothing_factor=0.1, smoothing_factor_step=0.01,
                                               max_smoothing_factor=0.2)
        pred_confident, target = _binary_batch(requires_grad=False)
        target = target.clone()
        pred_confident = torch.full_like(target, -10.0)
        pred_confident[:, 0] = 10.0  # near-perfect prediction -> low loss first
        criterion(pred_confident, target, current_epoch=0)
        before = criterion.smoothing_factor

        pred_bad = torch.full_like(target, 10.0)
        pred_bad[:, 0] = -10.0  # confidently wrong -> higher loss than before
        criterion(pred_bad, target, current_epoch=1)
        assert criterion.smoothing_factor > before

    def test_smoothing_factor_bounded(self):
        criterion = AdaptiveLabelSmoothingLoss(min_smoothing_factor=0.01, max_smoothing_factor=0.2,
                                               smoothing_factor_step=1.0, initial_smoothing_factor=0.1)
        pred, target = _binary_batch(requires_grad=False)
        for epoch in range(10):
            pred_bad = torch.full_like(target, 10.0)
            pred_bad[:, 0] = -10.0
            criterion(pred_bad, target, current_epoch=epoch)
        assert criterion.smoothing_factor <= 0.2

    def test_batch_size_one(self):
        pred, target = _binary_batch(batch_size=1, requires_grad=False)
        loss = AdaptiveLabelSmoothingLoss()(pred, target, current_epoch=0)
        assert torch.isfinite(loss)


class TestLabelRelaxationLoss:
    def test_zero_loss_when_confident_enough(self):
        alpha = 0.1
        target = torch.zeros(4, 5)
        target[:, 0] = 1.0
        # Softmax mass on the true class well above 1 - alpha.
        pred = torch.full((4, 5), -10.0)
        pred[:, 0] = 10.0
        loss = LabelRelaxationLoss(alpha=alpha)(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_loss_when_not_confident(self):
        alpha = 0.1
        target = torch.zeros(4, 5)
        target[:, 0] = 1.0
        pred = torch.zeros(4, 5)  # uniform predictions, far from confident
        loss = LabelRelaxationLoss(alpha=alpha)(pred, target)
        assert loss.item() > 0.0

    def test_gradient_flows(self):
        pred, target = _binary_batch(num_classes=5)
        loss = LabelRelaxationLoss(alpha=0.1)(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_batch_size_one(self):
        pred, target = _binary_batch(batch_size=1, num_classes=5, requires_grad=False)
        loss = LabelRelaxationLoss(alpha=0.1)(pred, target)
        assert torch.isfinite(loss)


class TestAdaptiveLabelRelaxationLoss:
    def test_finite_and_gradient_flows(self):
        pred, target = _binary_batch(num_classes=5)
        loss = AdaptiveLabelRelaxationLoss()(pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None

    def test_alpha_grows_when_loss_increases(self):
        criterion = AdaptiveLabelRelaxationLoss(initial_alpha=0.1, alpha_step=0.01, max_alpha=0.2)
        target = torch.zeros(4, 5)
        target[:, 0] = 1.0
        pred_confident = torch.full((4, 5), -10.0)
        pred_confident[:, 0] = 10.0
        criterion(pred_confident, target)
        before = criterion.alpha

        pred_bad = torch.zeros(4, 5)
        criterion(pred_bad, target)
        assert criterion.alpha > before

    def test_alpha_bounded(self):
        criterion = AdaptiveLabelRelaxationLoss(min_alpha=0.01, max_alpha=0.2, alpha_step=1.0,
                                                initial_alpha=0.1)
        target = torch.zeros(4, 5)
        target[:, 0] = 1.0
        for _ in range(10):
            pred_bad = torch.zeros(4, 5)
            criterion(pred_bad, target)
        assert criterion.alpha <= 0.2

    def test_batch_size_one(self):
        target = torch.zeros(1, 5)
        target[:, 0] = 1.0
        pred = torch.randn(1, 5)
        loss = AdaptiveLabelRelaxationLoss()(pred, target)
        assert torch.isfinite(loss)


class TestConfidenceBasedAdaptiveLabelRelaxationLoss:
    def test_finite_and_gradient_flows(self):
        pred, target = _binary_batch(num_classes=5)
        loss = ConfidenceBasedAdaptiveLabelRelaxationLoss()(pred, target, current_epoch=0)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None

    def test_alpha_shrinks_as_confidence_grows(self):
        criterion = ConfidenceBasedAdaptiveLabelRelaxationLoss(alpha=0.5)
        target = torch.zeros(4, 5)
        target[:, 0] = 1.0
        pred = torch.rand(4, 5)  # softmax mean confidence > 0 -> alpha must shrink
        before = criterion.alpha
        criterion(pred, target, current_epoch=0)
        assert criterion.alpha < before

    def test_batch_size_one(self):
        target = torch.zeros(1, 5)
        target[:, 0] = 1.0
        pred = torch.randn(1, 5)
        loss = ConfidenceBasedAdaptiveLabelRelaxationLoss()(pred, target, current_epoch=0)
        assert torch.isfinite(loss)


class TestCombinedLSandLR:
    def test_uses_label_smoothing_before_epoch_20(self):
        pred, target = _binary_batch(num_classes=5, requires_grad=False)
        expected = LabelSmoothingLoss(smoothness_ratio=0.1)(pred, target)
        actual = CombinedLSandLR(smoothness_ratio=0.1, alpha=0.2)(pred, target, current_epoch=19)
        torch.testing.assert_close(actual, expected)

    def test_uses_label_relaxation_from_epoch_20(self):
        pred, target = _binary_batch(num_classes=5, requires_grad=False)
        expected = LabelRelaxationLoss(alpha=0.2)(pred, target)
        actual = CombinedLSandLR(smoothness_ratio=0.1, alpha=0.2)(pred, target, current_epoch=20)
        torch.testing.assert_close(actual, expected)

    def test_gradient_flows(self):
        pred, target = _binary_batch(num_classes=5)
        loss = CombinedLSandLR(smoothness_ratio=0.1, alpha=0.2)(pred, target, current_epoch=0)
        loss.backward()
        assert pred.grad is not None


class TestCombinedAdaptiveLSandAdaptiveLR:
    def test_finite_before_and_after_switch(self):
        criterion = CombinedAdaptiveLSandAdaptiveLR()
        pred, target = _binary_batch(num_classes=5, requires_grad=False)
        early = criterion(pred, target, current_epoch=0)
        late = criterion(pred, target, current_epoch=100)
        assert torch.isfinite(early)
        assert torch.isfinite(late)

    def test_gradient_flows(self):
        criterion = CombinedAdaptiveLSandAdaptiveLR()
        pred, target = _binary_batch(num_classes=5)
        loss = criterion(pred, target, current_epoch=0)
        loss.backward()
        assert pred.grad is not None


class TestAggregatedLSandLR:
    def test_matches_weighted_average_formula(self):
        pred, target = _binary_batch(num_classes=5, requires_grad=False)
        smoothing = LabelSmoothingLoss(smoothness_ratio=0.1)(pred, target)
        relaxation = LabelRelaxationLoss(alpha=0.2)(pred, target)
        expected = 0.4 * smoothing + 0.6 * relaxation
        actual = AggregatedLSandLR(smoothness_ratio=0.1, alpha=0.2)(pred, target, current_epoch=0)
        torch.testing.assert_close(actual, expected)

    def test_gradient_flows(self):
        pred, target = _binary_batch(num_classes=5)
        loss = AggregatedLSandLR(smoothness_ratio=0.1, alpha=0.2)(pred, target, current_epoch=0)
        loss.backward()
        assert pred.grad is not None

    def test_batch_size_one(self):
        pred, target = _binary_batch(batch_size=1, num_classes=5, requires_grad=False)
        loss = AggregatedLSandLR(smoothness_ratio=0.1, alpha=0.2)(pred, target, current_epoch=0)
        assert torch.isfinite(loss)


class TestACLS:
    def test_finite_and_gradient_flows(self):
        criterion = ACLS(num_classes=5)
        pred = torch.randn(8, 5, requires_grad=True)
        target = torch.randint(0, 5, (8,))
        loss = criterion(pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None

    def test_batch_size_one(self):
        criterion = ACLS(num_classes=5)
        pred = torch.randn(1, 5)
        target = torch.randint(0, 5, (1,))
        loss = criterion(pred, target)
        assert torch.isfinite(loss)

    def test_higher_dim_input_is_reshaped(self):
        """forward reshapes (N, C, H, W)-style input to (N*H*W, C)."""
        criterion = ACLS(num_classes=3)
        pred = torch.randn(2, 3, 4, 4, requires_grad=True)
        target = torch.randint(0, 3, (2, 4, 4))
        loss = criterion(pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None

    def test_confident_correct_prediction_triggers_pos_penalty(self):
        """A logit for the correct class above `margin` should be penalized
        via get_reg's pos term, making the regularized loss exceed plain CE."""
        criterion = ACLS(num_classes=3, margin=1.0, pos_lambda=10.0, neg_lambda=0.0, alpha=1.0)
        pred = torch.tensor([[5.0, 0.0, 0.0]])
        target = torch.tensor([0])
        loss = criterion(pred, target)
        plain_ce = torch.nn.CrossEntropyLoss()(pred, target)
        assert loss.item() > plain_ce.item()
