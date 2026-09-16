"""Alternative loss functions selectable via the ``loss_fn`` config option.

``BaseKGE.__init__`` (``dicee/models/base_model.py``) instantiates one of
these in place of the framework's default loss based on
``args["loss_fn"]``; see that dispatch for the exact string values and
which additional config options (``label_smoothing_rate``,
``label_relaxation_alpha``) feed each one's constructor.

Several of these declare a ``current_epoch`` parameter on ``forward`` so
they can vary their behavior over training; ``BaseKGE.loss_function``
detects this once per model (``self._loss_needs_epoch``, set by inspecting
``self.loss.forward``'s parameters) and forwards the current epoch only to
losses that declare it.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DefaultBCELoss(nn.Module):
    """Plain ``BCEWithLogitsLoss``, selectable via ``loss_fn="BCELoss"``.

    Equivalent to the framework's own default entity/relation-prediction
    loss (see ``BaseKGE.__init__``); provided as an explicit ``loss_fn``
    option for parity with the other choices in this module.
    """

    def __init__(self):
        super(DefaultBCELoss, self).__init__()

    def forward(self, pred, target):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)


        return final_loss

class WeightedBCELoss(nn.Module):
    """BCE weighted by each prediction's own confidence.

    Down-weights low-confidence predictions (``sigmoid(pred)`` near 0.5) and
    up-weights confident ones, clamped to ``[0.5, 1.0]``. Weights are
    detached, so they don't receive gradients themselves. ``current_epoch``
    is accepted for interface consistency with the other ``loss_fn`` options
    but is currently unused.
    """

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, pred, target, current_epoch):

        gamma = 10
        confidence = torch.abs(2 * torch.sigmoid(pred) - 1)
        weights = torch.exp(-gamma * (1 - confidence))
        weights = torch.clamp(weights, min=0.5, max=1.0)

        weights = weights.detach()

        criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
        final_loss = criterion(pred, target)

        return final_loss

class LabelSmoothingLoss(nn.Module):
    """Intended as a fixed-rate label-smoothing loss, selectable via
    ``loss_fn="LS"`` (``smoothness_ratio`` comes from the ``label_smoothing_rate``
    config option).

    .. warning::
        ``forward`` currently ignores ``smoothness_ratio`` entirely and is
        byte-for-byte identical to :class:`DefaultBCELoss` — no smoothing is
        actually applied. See `issue #453
        <https://github.com/dice-group/dice-embeddings/issues/453>`_. This is
        independent of the (correctly implemented) dataset-level smoothing
        controlled by the same ``label_smoothing_rate`` option
        (``dicee/dataset_classes/_label_based.py``, ``_negative_sampling.py``),
        which bakes smoothing into the training targets regardless of
        ``loss_fn``.
    """

    def __init__(self, smoothness_ratio=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothness_ratio = smoothness_ratio

    def forward(self, pred, target):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)


        return final_loss

class AdaptiveLabelSmoothingLoss(nn.Module):
    """KL-divergence label smoothing with a smoothing factor that self-adjusts
    from batch to batch, selectable via ``loss_fn="AdaptiveLabelSmoothingLoss"``.

    Converts ``logits`` to log-probabilities and compares against a smoothed
    target distribution (mass ``smoothing_factor`` redistributed uniformly
    over the non-target classes) via ``KL(pred || smoothed_target)``. After
    each call, ``smoothing_factor`` moves one ``smoothing_factor_step``
    toward ``max_smoothing_factor`` if the loss increased since the previous
    call, or toward ``min_smoothing_factor`` if it decreased — i.e. it backs
    off smoothing while the model is improving and leans into it when
    training stalls. ``current_epoch`` is accepted for interface consistency
    with the other ``loss_fn`` options but is currently unused.

    This module is stateful (``prev_loss``, ``smoothing_factor`` persist
    across calls), so a fresh instance should be used per training run.

    Parameters
    ----------
    min_smoothing_factor : float
        Lower bound the adaptive smoothing factor can decay to.
    max_smoothing_factor : float
        Upper bound the adaptive smoothing factor can grow to.
    smoothing_factor_step : float
        Adjustment applied to the smoothing factor after each call.
    initial_smoothing_factor : float
        Starting value before any adjustment has occurred.
    """

    def __init__(self, min_smoothing_factor=0.01,
                 max_smoothing_factor=0.2,
                 smoothing_factor_step=0.01,
                 initial_smoothing_factor=0.1,
                 ):

        super(AdaptiveLabelSmoothingLoss, self).__init__()
        self.min_smoothing_factor = min_smoothing_factor
        self.max_smoothing_factor = max_smoothing_factor
        self.smoothing_factor_step = smoothing_factor_step
        self.smoothing_factor = initial_smoothing_factor
        self.prev_loss = None
        self.eps = 1e-14

    def forward(self, logits, target, current_epoch):

        pred = F.log_softmax(logits, dim=-1) # scores converted to be used in KL

        num_classes = logits.size(-1)
        #smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor / num_classes
        smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor * (1 - target) / (num_classes - 1)

        kl_loss = F.kl_div(pred, smoothed_target, reduction="batchmean")
        loss = kl_loss

        if self.prev_loss is not None:
            loss_diff = loss.item() - self.prev_loss
            if loss_diff >= 0.0:
                self.smoothing_factor = min(self.smoothing_factor + self.smoothing_factor_step, self.max_smoothing_factor)
                #self.gamma = min(self.gamma + self.gamma_step, self.max_gamma)
            elif loss_diff <= 0.0:
                self.smoothing_factor = max(self.smoothing_factor - self.smoothing_factor_step, self.min_smoothing_factor)

        self.prev_loss = loss.item()

        return loss

class LabelRelaxationLoss(nn.Module):
    """Label relaxation loss, selectable via ``loss_fn="LRLoss"``.

    Based on "From Label Smoothing to Label Relaxation" (Lienen &
    Hullermeier, AAAI 2021): instead of smoothing toward one fixed soft
    target, it constructs a *credal set* of distributions that assign at
    least ``1 - alpha`` probability to the target class(es) and measures the
    KL divergence to the closest point in that set, rather than to a single
    fixed distribution. Predictions that are already confident enough
    (probability mass on the target above ``1 - alpha``) incur zero loss.

    Parameters
    ----------
    alpha : float
        Relaxation strength in ``[0, 1)``; ``0`` recovers ordinary
        cross-entropy-style behavior (no slack), larger values tolerate
        more probability mass elsewhere before penalizing.
    """

    def __init__(self, alpha=0.0):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        # Greater zero threshold
        self.gz_threshold = 0.1
        self.eps = 1e-14

    def forward(self, pred, target):
        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        # Construct credal set
        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        # Calculate divergence
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
        pred = torch.sum(pred * target, dim=-1)
        result = torch.where(torch.gt(pred, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        final_loss = torch.mean(result)

        return final_loss

class AdaptiveLabelRelaxationLoss(nn.Module):
    """:class:`LabelRelaxationLoss` with an ``alpha`` that self-adjusts from
    batch to batch, selectable via ``loss_fn="AdaptiveLabelRelaxationLoss"``.

    Computes the label-relaxation loss once under ``torch.no_grad()`` with
    the current ``alpha`` to decide the adjustment (grow toward
    ``max_alpha`` if the loss increased since the previous call, shrink
    toward ``min_alpha`` if it decreased), then recomputes the loss with the
    updated ``alpha`` for the actual gradient-carrying return value. Like
    :class:`AdaptiveLabelSmoothingLoss`, this module is stateful
    (``prev_loss``, ``alpha`` persist across calls) - use a fresh instance
    per training run.

    Parameters
    ----------
    min_alpha : float
        Lower bound the adaptive ``alpha`` can decay to.
    max_alpha : float
        Upper bound the adaptive ``alpha`` can grow to.
    alpha_step : float
        Adjustment applied to ``alpha`` after each call.
    initial_alpha : float
        Starting value before any adjustment has occurred.
    """

    def __init__(self, min_alpha=0.01, max_alpha=0.2, alpha_step=0.01, initial_alpha=0.1):
        super(AdaptiveLabelRelaxationLoss, self).__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.alpha_step = alpha_step
        self.alpha = initial_alpha
        self.prev_loss = None
        self.eps = 1e-14
        self.gz_threshold = 0.1

    def forward(self, pred, target):
        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

            divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
            predc = torch.sum(pred * target, dim=-1)
            filtered_loss = torch.where(torch.gt(predc, 1. - self.alpha), torch.zeros_like(divergence),
                                        divergence)
            mean_final_loss = torch.mean(filtered_loss)

        if self.prev_loss is not None:
            loss_diff = mean_final_loss - self.prev_loss
            if loss_diff > 0:
                self.alpha = min(self.alpha + self.alpha_step, self.max_alpha)
            elif loss_diff < 0:
                self.alpha = max(self.alpha - self.alpha_step, self.min_alpha)

        self.prev_loss = mean_final_loss

        with torch.no_grad():
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
        predc = torch.sum(pred * target, dim=-1)
        result = torch.where(torch.gt(predc, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        final_loss = torch.mean(result)

        return final_loss


class ConfidenceBasedAdaptiveLabelRelaxationLoss(nn.Module):
    """:class:`LabelRelaxationLoss` where ``alpha`` shrinks as the model's
    own average prediction confidence grows, selectable via
    ``loss_fn="ConfidenceBasedAdaptiveLabelRelaxationLoss"``.

    Each call rescales ``alpha`` by ``(1 - mean(pred))`` before computing the
    loss. Unlike :class:`AdaptiveLabelRelaxationLoss`, this adjustment is
    one-directional and unbounded below: ``alpha`` only ever decays toward 0
    as confidence rises, with no mechanism to grow back if confidence later
    drops. ``current_epoch`` is accepted for interface consistency with the
    other ``loss_fn`` options but is currently unused. Stateful
    (``alpha`` persists and mutates across calls) - use a fresh instance
    per training run.

    Parameters
    ----------
    alpha : float
        Initial relaxation strength before any confidence-based decay.
    """

    def __init__(self, alpha=0.1):
        super(ConfidenceBasedAdaptiveLabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        # Greater zero threshold
        self.gz_threshold = 0.1
        self.eps = 1e-14

    def forward(self, pred, target, current_epoch):
        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        pred_confidence_mean = pred.mean().item()
        new_alpha = self.alpha * (1 - pred_confidence_mean)
        self.alpha = new_alpha

        # Construct credal set
        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        # Calculate divergence
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
        pred = torch.sum(pred * target, dim=-1)
        result = torch.where(torch.gt(pred, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        final_loss = torch.mean(result)

        return final_loss


class CombinedLSandLR(nn.Module):
    """Switches from :class:`LabelSmoothingLoss` to :class:`LabelRelaxationLoss`
    after epoch 20, selectable via ``loss_fn="CombinedLSandLR"``.

    Intended to smooth early training then relax the target distribution
    later on; note :class:`LabelSmoothingLoss`'s current no-op bug (`#453
    <https://github.com/dice-group/dice-embeddings/issues/453>`_) means the
    "smoothing" phase is currently plain BCE.

    Parameters
    ----------
    smoothness_ratio : float
        Forwarded to :class:`LabelSmoothingLoss` for the first 20 epochs.
    alpha : float
        Forwarded to :class:`LabelRelaxationLoss` from epoch 20 onward.
    """

    def __init__(self, smoothness_ratio=0.0, alpha=0.0):
        super(CombinedLSandLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha

    def forward(self, pred, target, current_epoch):
        criterion: nn.Module
        if current_epoch < 20:
            criterion = LabelSmoothingLoss(smoothness_ratio=self.smoothness_ratio)
        else:
            criterion = LabelRelaxationLoss(alpha=self.alpha)
        return criterion(pred, target)


class CombinedAdaptiveLSandAdaptiveLR(nn.Module):
    """Switches from :class:`AdaptiveLabelSmoothingLoss` to
    :class:`AdaptiveLabelRelaxationLoss` after epoch 100, selectable via
    ``loss_fn="CombinedAdaptiveLSandAdaptiveLR"``.

    Both inner losses use their default hyperparameters; there is currently
    no way to configure them through this wrapper.
    """

    def __init__(self):
        super(CombinedAdaptiveLSandAdaptiveLR, self).__init__()
        self.adaptive_label_smoothing = AdaptiveLabelSmoothingLoss()
        self.adaptive_label_relaxation = AdaptiveLabelRelaxationLoss()
        self.criterion = ''

    def forward(self, pred, target, current_epoch):
        final_loss = 0
        if current_epoch < 100:
            final_loss = self.adaptive_label_smoothing(pred, target, current_epoch)
        else:
            final_loss = self.adaptive_label_relaxation(pred, target)
        return final_loss


class AggregatedLSandLR(nn.Module):
    """Fixed 0.4/0.6 weighted average of :class:`LabelSmoothingLoss` and
    :class:`LabelRelaxationLoss`, selectable via ``loss_fn="AggregatedLSandLR"``.

    Unlike :class:`CombinedLSandLR`, both terms are computed every call and
    blended (``0.4 * smoothing + 0.6 * relaxation``) rather than switched
    between by epoch. Note :class:`LabelSmoothingLoss`'s current no-op bug
    (`#453 <https://github.com/dice-group/dice-embeddings/issues/453>`_)
    means the smoothing term is currently plain BCE.

    Parameters
    ----------
    smoothness_ratio : float
        Forwarded to the inner :class:`LabelSmoothingLoss`.
    alpha : float
        Forwarded to the inner :class:`LabelRelaxationLoss`.
    """

    def __init__(self, smoothness_ratio=0.1, alpha=0.1):
        super(AggregatedLSandLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha

    def forward(self, pred, target, current_epoch):
        Smoothing_criterion = LabelSmoothingLoss(smoothness_ratio=self.smoothness_ratio)
        Smoothing_loss = Smoothing_criterion(pred, target)

        Relaxation_criterion = LabelRelaxationLoss(alpha=self.alpha)
        Relaxation_loss = Relaxation_criterion(pred, target)

        w = 0.4
        final_loss = (w * Smoothing_loss) + ((1 - w) * Relaxation_loss)

        return final_loss


class ACLS(nn.Module):
    """Adaptive and Conditional Label Smoothing-style calibration loss,
    selectable via ``loss_fn="ACLS"``.

    Adds a margin-based confidence regularizer to cross-entropy: for the
    predicted (argmax) class, logits above ``margin`` are penalized
    quadratically (discourages over-confidence on the top class), and for
    every other class, logits within ``margin`` of the top logit are also
    penalized quadratically (discourages under-separation from the runner-up
    classes). The two penalties are weighted by ``pos_lambda``/``neg_lambda``
    and added to ``CrossEntropyLoss`` scaled by ``alpha``.

    .. warning::
        ``nn.CrossEntropyLoss`` expects ``targets`` as class indices (or a
        proper probability distribution per row); KGE's ``KvsAll`` labels
        are multi-hot (possibly several true tails per row), which is not
        an equivalent input. ``num_classes`` and ``ignore_index`` are
        currently unused in ``forward``/``get_reg``.

    Parameters
    ----------
    pos_lambda : float
        Weight of the over-confidence penalty on the predicted class.
    neg_lambda : float
        Weight of the under-separation penalty on the other classes.
    alpha : float
        Weight of the regularization term relative to cross-entropy.
    margin : float
        Logit margin used by both penalty terms.
    num_classes : int
        Currently unused.
    ignore_index : int
        Currently unused.
    """

    def __init__(self,
                 pos_lambda: float = 1.0,
                 neg_lambda: float = 0.1,
                 alpha: float = 0.1,
                 margin: float = 10.0,
                 num_classes: int = 200,
                 ignore_index: int = -100):
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "reg"

    def get_reg(self, inputs, targets):
        #print(targets)

        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        indicator = (max_values.clone().detach() == inputs.clone().detach()).float()

        batch_size, num_classes = inputs.size()
        num_pos = batch_size * 1.0
        num_neg = batch_size * (num_classes - 1.0)

        neg_dist = max_values.clone().detach() - inputs

        pos_dist_margin = F.relu(max_values - self.margin)
        neg_dist_margin = F.relu(neg_dist - self.margin)

        pos = indicator * pos_dist_margin ** 2
        neg = (1.0 - indicator) * (neg_dist_margin ** 2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)
        return reg

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = self.cross_entropy(inputs, targets)

        loss_reg = self.get_reg(inputs, targets)
        loss = loss_ce + self.alpha * loss_reg

        return loss
