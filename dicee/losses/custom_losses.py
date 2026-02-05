import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import os
from torch import tensor


class DefaultBCELoss(nn.Module):
    def __init__(self):
        super(DefaultBCELoss, self).__init__()

    def forward(self, pred, target, current_epoch):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)

        return final_loss

class WeightedBCELoss(nn.Module):
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
    def __init__(self, smoothness_ratio=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.log_dir = "./smoothing_rates"

    def forward(self, pred, target, current_epoch):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)

        return final_loss

class AdaptiveLabelSmoothingLoss(nn.Module):
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
        self.log_dir ="./smoothing_rates"

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
    def __init__(self, alpha=0.0):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        # Greater zero threshold
        self.gz_threshold = 0.1
        self.eps = 1e-14

    def forward(self, pred, target, current_epoch):
        probs = torch.softmax(pred, dim=-1)

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
    def __init__(self, min_alpha=0.01, max_alpha=0.2, alpha_step=0.01, initial_alpha=0.1):
        super(AdaptiveLabelRelaxationLoss, self).__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.alpha_step = alpha_step
        self.alpha = initial_alpha
        self.prev_loss = None
        self.eps = 1e-14
        self.gz_threshold = 0.1

    def forward(self, pred, target, current_epoch):
        probs = torch.softmax(pred, dim=-1)

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
    def __init__(self, smoothness_ratio=0.0, alpha=0.0):
        super(CombinedLSandLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha

    def forward(self, pred, target, current_epoch):
        final_loss = 0
        if current_epoch < 20:
            criterion = LabelSmoothingLoss(smoothness_ratio=self.smoothness_ratio)
            final_loss = criterion(pred, target, current_epoch)
        else:
            criterion = LabelRelaxationLoss(alpha=self.alpha)
            final_loss = criterion(pred, target, current_epoch)
        return final_loss


class CombinedAdaptiveLSandAdaptiveLR(nn.Module):
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
            final_loss = self.adaptive_label_relaxation(pred, target, current_epoch)
        return final_loss


class AggregatedLSandLR(nn.Module):
    def __init__(self, smoothness_ratio=0.1, alpha=0.1):
        super(AggregatedLSandLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha

    def forward(self, pred, target, current_epoch):
        final_loss = 0

        Smoothing_criterion = LabelSmoothingLoss(smoothness_ratio=self.smoothness_ratio)
        Smoothing_loss = Smoothing_criterion(pred, target, current_epoch)

        Relaxation_criterion = LabelRelaxationLoss(alpha=self.alpha)
        Relaxation_loss = Relaxation_criterion(pred, target, current_epoch)

        w = 0.4
        final_loss = (w * Smoothing_loss) + ((1- w) * Relaxation_loss)

        return final_loss

"""
class GradientBasedLSLR(nn.Module):
    def __init__(self, smoothness_ratio=0.0, alpha=0.0, check_interval=10, dynamic_threshold_ratio=0.015):
        super(GradientBasedLSLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha
        self.check_interval = check_interval
        self.dynamic_threshold_ratio = dynamic_threshold_ratio
        self.mode = 'smooth'
        self.grad_norm_history = []

        self.LabelSmoothingLoss = LabelSmoothingLoss()
        self.LabelRelaxationLoss = LabelRelaxationLoss()


    def forward(self, pred, target, current_epoch, gradient_norm):
        if len(self.grad_norm_history) == self.check_interval:
            self.grad_norm_history.pop(0)
        self.grad_norm_history.append(gradient_norm)

        avg_norm = sum(self.grad_norm_history) / len(self.grad_norm_history) + 1e-14 # avoid division by zero

        if current_epoch != 0:
            if avg_norm < self.dynamic_threshold_ratio and self.mode == 'smooth':
                self.mode = 'relax'

        #final_loss = 0
        if self.mode == 'smooth':
            final_loss = self.LabelSmoothingLoss(pred, target, current_epoch, gradient_norm)
        else:
            final_loss = self.LabelRelaxationLoss(pred, target, current_epoch, gradient_norm)

        return final_loss


class GradientBasedAdaptiveLSLR(nn.Module):
    def __init__(self, smoothness_ratio=0.0, alpha=0.0, check_interval=10,
                 variability_threshold=0.09):
        super(GradientBasedAdaptiveLSLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha
        self.check_interval = check_interval
        self.variability_threshold = variability_threshold
        self.mode = 'smooth'
        self.grad_norm_history = []
        self.adaptive_label_smoothing = AdaptiveLabelSmoothingLoss()
        self.adaptive_label_relaxation = AdaptiveLabelRelaxationLoss()

    def update_dynamic_threshold(self, current_epoch):
        if len(self.grad_norm_history) < self.check_interval:
            return

        std_dev = np.std(self.grad_norm_history)
        print(std_dev, self.variability_threshold)
        if std_dev < self.variability_threshold and self.mode == 'smooth':
            self.mode = 'relax'

    def forward(self, pred, target, current_epoch, gradient_norm):
        if len(self.grad_norm_history) == self.check_interval:
            self.grad_norm_history.pop(0)
        self.grad_norm_history.append(gradient_norm)

        if current_epoch % self.check_interval == 0:
            self.update_dynamic_threshold(current_epoch)

        if self.mode == 'smooth':
            return self.adaptive_label_smoothing(pred, target, current_epoch, gradient_norm)
        else:
            return self.adaptive_label_relaxation(pred, target, current_epoch, gradient_norm)

"""

class ACLS(nn.Module):

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

    def forward(self, inputs, targets, current_epoch):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = self.cross_entropy(inputs, targets)

        loss_reg = self.get_reg(inputs, targets)
        loss = loss_ce + self.alpha * loss_reg

        return loss
    
#My Work

class UNITEI(nn.Module):
    """
    UNITE-I loss (pattern-aware & noise-resilient) for mini-batch training.
    """

    def __init__(
        self,
        gamma: float = 5,     
        sigma: float = 1000,  
        lambda_q: float = 0.3,  # Regularization term
        clamp_scores: bool = True,
        use_tanh_scale: bool = True,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.lambda_q = float(lambda_q)
        self.clamp_scores = clamp_scores
        self.use_tanh_scale = use_tanh_scale
        self.eps = 1e-12

    def _normalize_scores(self, pred: torch.Tensor) -> torch.Tensor:
        # To keep gamma meaningful and gradients don't die or explode.
        x = pred
        if self.use_tanh_scale:
            # squashes to (-1,1), then scales to (-gamma, gamma)
            x = torch.tanh(x) * self.gamma
        if self.clamp_scores:
            x = torch.clamp(x, min=-self.gamma, max=self.gamma)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch = None):

        target = target.float()
        pred = self._normalize_scores(pred)

        tau_pos = F.relu(self.gamma - pred)  # only where target==1 used
        tau_neg = F.relu(pred + self.gamma)  # only where target==0 used #changed - to +

        #Quadratic penalties
        loss_pos = self.sigma * (tau_pos ** 2)
        loss_neg = self.sigma * (tau_neg ** 2)

        # Constraint penalties with Q(x) = sigmoid(x)
        # Pos: Q(gamma - f) >= Q(tau) → violation = ReLU(Q(tau) - Q(gamma - f)) where f=pred
        # Neg: Q(-gamma - f) >= Q(tau) → violation = ReLU(Q(tau) - Q(-gamma - f))
        Q_tau_pos = torch.sigmoid(tau_pos)
        Q_tau_neg = torch.sigmoid(tau_neg)
        Q_pos = torch.sigmoid(self.gamma - pred)
        Q_neg = torch.sigmoid(-self.gamma - pred) #changed from pred - self.gamma to -self.gamma - pred

        pos_violation = F.relu(Q_tau_pos - Q_pos)
        neg_violation = F.relu(Q_tau_neg - Q_neg)

        # Mask by labels and combine
        pos_mask = target
        neg_mask = 1.0 - target

        loss = (
            pos_mask * (loss_pos + self.lambda_q * pos_violation)
            + neg_mask * (loss_neg + self.lambda_q * neg_violation)
        )

        return loss.mean()


class RoBoSS(nn.Module):

    def __init__(self, a_roboss=5.0, lambda_roboss=1.5, margin=1.0, normalize_scores=False):
        super().__init__()
        self.a_roboss = float(a_roboss)
        self.lambda_roboss = float(lambda_roboss)
        self.margin = float(margin)
        self.normalize_scores = normalize_scores
        self.eps = 1e-8

    def _normalize_scores(self, pred):
        if self.normalize_scores:
            return torch.tanh(pred)
        return pred

    def forward(self, pred, target, current_epoch = None):
        target = target.float()
        print(type(target))
        pred = pred.float()
        print(type(pred))
    
        pred = self._normalize_scores(pred)
        
        # Convert [0, 1] to [-1, 1]
        if target.min() >= 0 and target.max() <= 1:
            target = 2 * target - 1
        
        u = self.margin - (target * pred)
        
        term1 = (self.a_roboss * u) + 1.0
        
        term2 = torch.exp(-self.a_roboss * u)

        loss_values = self.lambda_roboss * (1.0 - (term1 * term2))
        
        # Apply condition: Loss is 0 if u <= 0 (Correctly classified)
        # assign a fixed loss of 0 for all samples with u < 0
        loss = torch.where(u > 0, loss_values, torch.zeros_like(loss_values))
        
        return loss.mean()


class AGCELoss(nn.Module):

    def __init__(self, agce_a=0.1, agce_q=1.0, eps=1e-8, scale=1.0):
        super(AGCELoss, self).__init__()
        self.agce_a = float(agce_a)
        self.agce_q = float(agce_q)
        self.eps = eps
        self.scale = scale
        
        # According to Corollary 1 in the paper:
        # - If q > 1: Loss is "Completely Asymmetric" (Robust).
        # - If q <= 1: Robustness depends on 'a' being large enough.
        if self.agce_q <= 1:
            print(f"Warning: q={agce_q} (<=1). Ensure 'a' is tuned high enough for noise robustness.")

    def forward(self, pred, labels, current_epoch = None):
        y = labels
        
        y_hat = torch.sigmoid(pred)
        
        u = y * y_hat + (1 - y) * (1 - y_hat)
        
        u = torch.clamp(u, min=self.eps, max=1.0 - self.eps)

        term1 = (self.agce_a + 1) ** self.agce_q
        term2 = (self.agce_a + u) **  self.agce_q
        
        loss = (term1 - term2) / self.agce_q
        
        return loss.mean() * self.scale


class AULoss(nn.Module):
    def __init__(self, aul_a=1.5, aul_p=0.9, eps=1e-7, scale=1.0):
        super(AULoss, self).__init__()
        self.aul_a = float(aul_a)
        self.aul_p = float(aul_p)
        self.eps = eps
        self.scale = scale

        # assert self.aul_a > 1.0, "Parameter 'aul_a' must be > 1.0 for AUL."

    def forward(self, pred, labels, current_epoch = None):
        y = labels
        #Map scores/logits to probabilities [0, 1]
        y_hat = torch.sigmoid(pred)
        
        # Probability of the CORRECT classification)
        # p_target = y * P(y=1) + (1-y) * P(y=0)
        u = y * y_hat + (1 - y) * (1 - y_hat)

        # Clamp for numerical stability in power operations
        u = torch.clamp(u, min=self.eps, max=1.0 - self.eps)
        
        # 3. Apply AUL Formula: ((aul_a - p_target)^aul_p - (aul_a - 1)^aul_p) / aul_p
        term1 = (self.aul_a - u) ** self.aul_p
        term2 = (self.aul_a - 1) ** self.aul_p
        
        loss = (term1 - term2) / self.aul_p
        
        return loss.mean()


class AELoss(nn.Module):
    def __init__(self, a_ael=0.5, eps=1e-7, scale=1.0):
        super(AELoss, self).__init__()
        self.a_ael = float(a_ael)
        self.eps = eps
        self.scale = scale

        # Validation based on paper constraints
        assert self.a_ael > 0, "Parameter 'a' must be > 0."

    def forward(self, pred, labels, current_epoch=None):

        pred = torch.sigmoid(pred)
        p_target = labels * pred + (1 - labels) * (1 - pred)

        p_target = torch.clamp(p_target, min=self.eps, max=1.0)
        
        loss = torch.exp(-p_target / self.a_ael)
        
        return loss.mean() * self.scale


class EwLoss(nn.Module):
    """
    Corrected Implementation of EwLoss (Exponential Weighted Loss) from
    "EwLoss: A Exponential Weighted Loss Function for Knowledge Graph Embedding Models"
    (Shen et al., IEEE Access, 2023).

    The loss re-weights the negative samples with an exponential coefficient so that
    hard-to-distinguish negatives contribute more to the objective.

    Paper Reference:
      - "Assigns weights... to decrease the weights of easily discriminated samples" [cite: 205]
      - Formula: EwLoss = -log σ(γ - f_pos) - Σ_j (p_j)^α * log σ(f_neg_j - γ) [cite: 249]
    Parameters
    ----------
    margin : float, default=1.0
        γ in the paper.
    alpha : float, default=0.9
        Exponential weight coefficient applied to the normalized negative scores.
    temperature : float, default=1.0
        Softmax temperature used when computing p_j.
    negative_sample_size : int or None, default=512
        Number of negative scores to consider per (h, r) pair.
    normalize_scores : bool, default=False
        Whether to normalize logits with tanh before computing the loss.
    """
    def __init__(self,
                 margin: float = 3.0,
                 alpha: float = 0.1,
                 temperature: float = 1.0,
                 negative_sample_size: int = 512,
                 sample_mode: str = "random",
                 normalize_scores: bool = False,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0,
                 positive_threshold: float = 0.5,
                 eps: float = 1e-12):
        super().__init__()
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.negative_sample_size = None
        if negative_sample_size is not None and negative_sample_size > 0:
            self.negative_sample_size = int(negative_sample_size)
        self.sample_mode = sample_mode if sample_mode in {"random", "topk"} else "random"
        self.normalize_scores = normalize_scores
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.positive_threshold = float(positive_threshold)
        self.eps = float(eps)

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.normalize_scores:
            return torch.tanh(scores)
        return scores

    def _sample_negatives(self, scores: torch.Tensor) -> torch.Tensor:
        if self.negative_sample_size is None or scores.numel() <= self.negative_sample_size:
            return scores
        
        # Optimization: For hard negative mining, we might want the smallest distances
        if self.sample_mode == "topk":
            # If scores are distances, "hardest" are the smallest values
            topk_vals, _ = torch.topk(scores, self.negative_sample_size, largest=False)
            return topk_vals
        
        # Random sampling (default)
        idx = torch.randperm(scores.numel(), device=scores.device)[:self.negative_sample_size]
        return scores[idx]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch=None) -> torch.Tensor:
        """
        Args:
            pred: Prediction scores (Distances: Lower is better/closer).
            target: Binary labels (1 for positive, 0 for negative).
        """
        target = target.float()
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        pred = pred.float()
        pred = self._normalize_scores(pred)
        positive_mask = target >= self.positive_threshold

        batch_loss = pred.new_tensor(0.0)
        valid_rows = 0

        for row_pred, row_mask in zip(pred, positive_mask):
            pos_scores = row_pred[row_mask]
            neg_scores = row_pred[~row_mask]

            if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                continue

            neg_scores = self._sample_negatives(neg_scores)

            # --- 1. Positive Term ---
            # Minimize distance: push pos_scores < margin
            pos_term = F.softplus(pos_scores - self.margin).mean()

            # --- 2. Calculate Weights ---
            # In distance metrics, Hard Negatives = Small Distance.
            # Easy Negatives = Large Distance.
            # Softmax(x) creates high prob for large x.
            # To weight Hard Negatives higher, we must negate the distance.
            # Eq 17: p(h') = exp(f(h')) / sum(...) where f is a score (inverse of distance)
            scaled_neg_for_weights = -neg_scores / max(self.temperature, self.eps)
            
            probs = torch.softmax(scaled_neg_for_weights, dim=0)
            weights = torch.pow(probs + self.eps, self.alpha)

            # --- 3. Negative Term ---
            # Maximize distance: push neg_scores > margin
            # Eq 18: - sum( weights * log sigmoid(neg_scores - gamma) )
            # Note: logsigmoid(x) is always negative. We want to maximize it towards 0.
            log_sig = F.logsigmoid(neg_scores - self.margin)
            
            # We use a weighted average (dividing by weights.sum()) for numerical stability
            # though the paper Eq 18 implies a direct sum.
            neg_term = -(weights * log_sig).sum() / (weights.sum() + self.eps)

            row_loss = self.pos_weight * pos_term + self.neg_weight * neg_term
            batch_loss = batch_loss + row_loss
            valid_rows += 1

        if valid_rows == 0:
            return pred.new_tensor(0.0, requires_grad=True)

        return batch_loss / valid_rows


class SynMarginLoss(nn.Module):

    def __init__(self,
                 mode: str = "projection",
                 margin: float = 0.1,
                 temperature: float = 1.0,
                 detach_negative: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        assert mode in {"projection", "difference"}, "mode must be projection or difference"
        self.mode = mode
        self.margin = margin
        self.temperature = temperature
        self.detach_negative = detach_negative
        self.eps = eps
        self.embedding_layer = None

    def bind_embedding_layer(self, embedding_layer: nn.Embedding) -> None:
        """
        Attach the entity embedding layer so that the loss can retrieve target vectors.
        """
        self.embedding_layer = embedding_layer

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch=None):
        if self.embedding_layer is None:
            raise RuntimeError("SynMarginLoss requires entity embeddings. Call bind_embedding_layer first.")

        entity_vectors = self.embedding_layer.weight
        num_entities = entity_vectors.shape[0]

        pred = self._reshape_scores(pred, num_entities, "predictions")
        target = self._reshape_scores(target, num_entities, "targets")

        prob = torch.softmax(pred / self.temperature, dim=-1)
        pred_emb = prob @ entity_vectors
        pred_emb = F.normalize(pred_emb, p=2, dim=-1, eps=self.eps)

        target_weights = torch.clamp(target, min=0.0)
        mass = target_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        target_probs = target_weights / mass
        target_emb = target_probs @ entity_vectors
        target_emb = F.normalize(target_emb, p=2, dim=-1, eps=self.eps)

        neg = self._synthesize_negative(pred_emb, target_emb)
        if self.detach_negative:
            neg = neg.detach()

        pos_sim = (pred_emb * target_emb).sum(dim=-1)
        neg_sim = (pred_emb * neg).sum(dim=-1)
        loss = torch.relu(self.margin + neg_sim - pos_sim)
        return loss.mean()

    def _reshape_scores(self, tensor: torch.Tensor, num_entities: int, name: str) -> torch.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)

        if tensor.dim() != 2:
            raise ValueError(f"SynMarginLoss expects 2-D {name}, got shape {tensor.shape}")

        if tensor.shape[1] != num_entities:
            if tensor.numel() % num_entities == 0:
                tensor = tensor.view(-1, num_entities)
            else:
                raise RuntimeError(
                    f"SynMarginLoss requires {name} dimension to match number of entities "
                    f"({num_entities}), got shape {tensor.shape}. "
                    f"Please use a scoring technique that produces scores for every entity (e.g., KvsAll/AllvsAll/1vsAll)."
                )
        return tensor

    def _synthesize_negative(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        if self.mode == "difference":
            neg = pred_emb - target_emb
        else:
            dot = (pred_emb * target_emb).sum(dim=-1, keepdim=True)
            neg = pred_emb - dot * target_emb
            zero_mask = neg.norm(dim=-1, keepdim=True) <= self.eps
            if zero_mask.any():
                fallback = pred_emb - target_emb
                neg = torch.where(zero_mask, fallback, neg)
        neg = F.normalize(neg, p=2, dim=-1, eps=self.eps)
        return neg


class WaveLoss(nn.Module):
    def __init__(self, wave_a = 1.5, lambda_param = 0.5, eps = 1e-8):
        super().__init__()
        self.wave_a = wave_a
        self.lambda_param = lambda_param
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch = None):
        target = target.float()
        pred = pred.float()

        if target.min() >= 0 and target.max() <= 1:
            target = 2 * target - 1 

        u = 1.0 - (target * pred)

        u_sqr = u * u
        exp_term = torch.exp(self.wave_a * u)
        
        denom = 1.0 + self.lambda_param * u_sqr * exp_term + self.eps

        wave_loss = (1.0 / self.lambda_param) * (1.0 - 1.0 / denom)

        return wave_loss.mean()

class NSSALoss(nn.Module):
    """
    Negative Sampling Self-Adversarial (NSSA) loss.
    """

    def __init__(
        self,
        nssa_alpha = 1.0,
        positive_threshold = 0.5, #Semantic threshlod for postive classes
    ):
        super().__init__()
        self.nssa_alpha = nssa_alpha
        self.positive_threshold = positive_threshold

    # def forward(
    #     self,
    #     pred,
    #     target,
    #     current_epoch=None
    # ):

    #     pos_mask = target > self.positive_threshold
    #     neg_mask = ~pos_mask

    #     # Handle 1D targets (single row) vs batched 2D targets.
    #     if target.dim() == 1:
    #         if not (pos_mask.any() and neg_mask.any()):
    #             return pred.new_tensor(0.0, requires_grad=True)
    #     else:
    #         # Masking invalid rows
    #         valid_mask = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    #         if not valid_mask.any():
    #             return pred.new_tensor(0.0, requires_grad=True)

    #         pred = pred[valid_mask]
    #         pos_mask = pos_mask[valid_mask]
    #         neg_mask = neg_mask[valid_mask]

    #     pos_scores = pred.masked_select(pos_mask)
    #     pos_loss = -F.logsigmoid(pos_scores).mean()

    #     neg_scores = pred.masked_select(neg_mask)
    #     weights = F.softmax(neg_scores * self.nssa_alpha, dim=0).detach()
    #     neg_loss = -(weights * F.logsigmoid(-neg_scores)).sum()

    #     return (pos_loss + neg_loss) / 2
    def forward(
        self,
        pred,
        target,
        current_epoch=None
    ):
    
        pos_mask = target > self.positive_threshold
        neg_mask = ~pos_mask
        
        pos_scores = pred[pos_mask]
        pos_loss = -F.logsigmoid(pos_scores).mean()
        
        neg_loss = 0
        batch_size = pred.size(0)
        
        for i in range(batch_size):
            row_neg_scores = pred[i][neg_mask[i]]
            if row_neg_scores.numel() == 0:
                continue
                
            weights = F.softmax(row_neg_scores * self.nssa_alpha, dim=0).detach()
            neg_loss += -(weights * F.logsigmoid(-row_neg_scores)).sum()
            
        return (pos_loss + (neg_loss / batch_size)) / 2

class FocalLoss(nn.Module):

    def __init__(self, gamma = 2.0, alpha = 0.25):
        super().__init__() 

        self.gamma = gamma 
        self.alpha = alpha 

    def forward(self, pred, target, current_epoch = None):

        p = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)

        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss

        return loss.mean()


class LocalTripleLoss(nn.Module):
    """
    Local triple loss with per-triple confidence updates (CKRL local confidence)
    """

    requires_x_batch = True

    def __init__(
        self,
        margin = 1.0,
        alpha = 0.9,
        beta = 1e-4,
        min_conf = 0.0,
        max_conf = 1.0,
        positive_threshold = 0.5,
        use_max_negative = True,
        score_is_distance = False
    ):
        super().__init__()
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_conf = float(min_conf)
        self.max_conf = float(max_conf)
        self.positive_threshold = float(positive_threshold)
        self.use_max_negative = bool(use_max_negative)
        self.score_is_distance = bool(score_is_distance)
        self._confidence = {}


    def forward(self, pred, target, current_epoch = None, x_batch = None):
        if x_batch is None:
            raise RuntimeError("LocalTripleLoss requires x_batch to track per-triple confidence.")
        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)
        if pred.shape[0] != target.shape[0]:
            raise RuntimeError("LocalTripleLoss expects pred and target to have matching shapes.")
        if x_batch.dim() != 2 or x_batch.shape[1] != 3:
            raise RuntimeError("LocalTripleLoss expects x_batch to be a [N, 3] tensor of triples.")
        if x_batch.shape[0] != pred.shape[0]:
            raise RuntimeError("LocalTripleLoss expects x_batch and pred to align on the first dimension.")

        pos_mask = target > self.positive_threshold
        pos_scores = pred[pos_mask]
        neg_scores = pred[~pos_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return pred.new_tensor(0.0, requires_grad = True) 

        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if neg_count % pos_count == 0: #Checks if negatives are evenly aligned with positives
            neg_ratio = neg_count // pos_count 
            neg_scores = neg_scores.view(neg_ratio, pos_count) #neg_scores.shape == [neg_ratio, pos_count]
            if self.use_max_negative:
                neg_per_pos = neg_scores.max(dim = 0).values 
            else:
                neg_per_pos = neg_scores.mean(dim = 0)
        else:
            neg_value = neg_scores.max() if self.use_max_negative else neg_scores.mean()
            neg_per_pos = neg_value.expand_as(pos_scores)
            #Every positive is compared against the same negative score

        if self.score_is_distance:
            e_pos = pos_scores 
            e_neg = neg_per_pos 
        else:
            e_pos = -pos_scores
            e_neg = -neg_per_pos #larger the better assumption

        delta = self.margin + e_pos - e_neg 
        hinge = F.relu(delta) 

        pos_triples = x_batch[pos_mask].detach().cpu().tolist()
        #Initially all triples have a confidence score of 1.0
        conf_values = [self._confidence.get(tuple(triple), 1.0) for triple in pos_triples] 
        conf = pred.new_tensor(conf_values)

        loss = (conf * hinge).mean()

        q_values = (-delta).detach() #Energy value - Equation 5
        for triple, q_val, conf_val in zip(pos_triples, q_values.tolist(), conf_values):
            #Equation 6
            if q_val <= 0:
                new_conf = max(conf_val * self.alpha, self.min_conf) #Bad priples punished
            else:
                new_conf = min(conf_val + self.beta, self.max_conf) #caps good triples back to 1.0
            self._confidence[tuple(triple)] = new_conf 
        
        return loss 

def compute_prior_path_confidence(path_set, rel_path_prior, path_prior, epsilon = 1e-6):
    """
    Compute prior path confidence (PP) for a triple.

    path_set: iterable of (path_id, reliability) where reliability is R(h,p,t)
    rel_path_prior: dict mapping path_id -> P(r, p)
    path_prior: dict mapping path_id -> P(p)
    epsilon: smoothing value (epsilon in the paper)
    """
    pp = 0.0
    for path_id, reliability in path_set:
        p_rp = rel_path_prior.get(path_id, 0.0)
        p_p = path_prior.get(path_id, 0.0)
        q_pp = epsilon + (1.0 - epsilon) * (p_rp / max(p_p, epsilon))
        pp += q_pp * reliability
    return pp


class LocalTripleWithPriorPathLoss(nn.Module):
    """
    local triple confidence (LT) with prior path confidence (PP).
    """

    requires_x_batch = True

    def __init__(
        self,
        margin = 1.0,
        alpha = 0.9,
        beta = 1e-4,
        min_conf = 0.0,
        max_conf = 1.0,
        positive_threshold = 0.5,
        use_max_negative = True,
        score_is_distance = False,
        lambda_1_lt = 1.5,
        lambda_2_pp = 0.1,
        prior_confidence_map = None,
    ):
        super().__init__()
        self.local_loss = LocalTripleLoss(
            margin = margin,
            alpha = alpha,
            beta = beta,
            min_conf = min_conf,
            max_conf = max_conf,
            positive_threshold = positive_threshold,
            use_max_negative = use_max_negative,
            score_is_distance = score_is_distance,
        )
        self.lambda_1_lt = float(lambda_1_lt)
        self.lambda_2_pp = float(lambda_2_pp)
        self._prior_confidence = prior_confidence_map or {}

    def set_prior_confidence_map(self, confidence_map): #stores the PP lookup table inside the loss
        self._prior_confidence = confidence_map or {}

    def _get_prior_confidence(self, pos_triples, device, dtype): #retrieves PP values from the table for the current batch of triples
        if not self._prior_confidence:
            return torch.zeros(len(pos_triples), device = device, dtype = dtype)
        values = [self._prior_confidence.get(tuple(triple), 0.0) for triple in pos_triples]
        return torch.tensor(values, device = device, dtype = dtype)

    def forward(self, pred, target, current_epoch = None, x_batch = None):
        if not torch.is_tensor(target):
            if isinstance(target, (list, tuple)):
                tensors = [item for item in target if torch.is_tensor(item)]
                if tensors:
                    target = None
                    if pred is not None:
                        for item in tensors:
                            if item.numel() == pred.numel():
                                target = item
                                break
                    if target is None:
                        target = tensors[-1]
            if target is None:
                raise RuntimeError("LocalTripleWithPriorPathLoss expects target to be a torch.Tensor.")
        if x_batch is None:
            raise RuntimeError("LocalTripleWithPriorPathLoss requires x_batch to track per-triple confidence.")
        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)
        if pred.shape[0] != target.shape[0]:
            raise RuntimeError("LocalTripleWithPriorPathLoss expects pred and target to have matching shapes.")
        if x_batch.dim() != 2 or x_batch.shape[1] != 3:
            raise RuntimeError("LocalTripleWithPriorPathLoss expects x_batch to be a [N, 3] tensor of triples.")
        if x_batch.shape[0] != pred.shape[0]:
            raise RuntimeError("LocalTripleWithPriorPathLoss expects x_batch and pred to align on the first dimension.")

        pos_mask = target > self.local_loss.positive_threshold
        pos_scores = pred[pos_mask]
        neg_scores = pred[~pos_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return pred.new_tensor(0.0, requires_grad = True)

        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if neg_count % pos_count == 0:
            neg_ratio = neg_count // pos_count
            neg_scores = neg_scores.view(neg_ratio, pos_count)
            if self.local_loss.use_max_negative:
                neg_per_pos = neg_scores.max(dim = 0).values
            else:
                neg_per_pos = neg_scores.mean(dim = 0)
        else:
            neg_value = neg_scores.max() if self.local_loss.use_max_negative else neg_scores.mean()
            neg_per_pos = neg_value.expand_as(pos_scores)

        if self.local_loss.score_is_distance:
            e_pos = pos_scores
            e_neg = neg_per_pos
        else:
            e_pos = -pos_scores
            e_neg = -neg_per_pos

        delta = self.local_loss.margin + e_pos - e_neg
        hinge = F.relu(delta)

        pos_triples = x_batch[pos_mask].detach().cpu().tolist()
        conf_values = [self.local_loss._confidence.get(tuple(triple), 1.0) for triple in pos_triples]
        local_conf = pred.new_tensor(conf_values)
        prior_conf = self._get_prior_confidence(pos_triples, device = pred.device, dtype = pred.dtype)
        combined_conf = self.lambda_1_lt * local_conf + self.lambda_2_pp * prior_conf
        loss = (combined_conf * hinge).mean()

        q_values = (-delta).detach()
        for triple, q_val, conf_val in zip(pos_triples, q_values.tolist(), conf_values):
            if q_val <= 0:
                new_conf = max(conf_val * self.local_loss.alpha, self.local_loss.min_conf)
            else:
                new_conf = min(conf_val + self.local_loss.beta, self.local_loss.max_conf)
            self.local_loss._confidence[tuple(triple)] = new_conf

        return loss


class LocalTripleWithPriorAndAdaptivePathLoss(nn.Module):
    """
    Combine local triple confidence (LT), prior path confidence (PP),
    and adaptive path confidence (AP).
    """

    requires_x_batch = True
    requires_model = True

    def __init__(
        self,
        margin = 1.0,
        alpha = 0.9,
        beta = 1e-4,
        min_conf = 0.0,
        max_conf = 1.0,
        positive_threshold = 0.5,
        use_max_negative = True,
        score_is_distance = False,
        lambda_1_lt = 1.5,
        lambda_2_pp = 0.1,
        lambda_3_ap = 0.4,
        adaptive_use_l1 = True,
        prior_confidence_map = None,
    ):
        super().__init__()
        self.local_loss = LocalTripleLoss(
            margin = margin,
            alpha = alpha,
            beta = beta,
            min_conf = min_conf,
            max_conf = max_conf,
            positive_threshold = positive_threshold,
            use_max_negative = use_max_negative,
            score_is_distance = score_is_distance,
        )
        self.lambda_1_lt = float(lambda_1_lt)
        self.lambda_2_pp = float(lambda_2_pp)
        self.lambda_3_ap = float(lambda_3_ap)
        self.adaptive_use_l1 = bool(adaptive_use_l1)
        self._prior_confidence = prior_confidence_map or {}
        self._path_data = {}

    def set_prior_confidence_map(self, confidence_map):
        self._prior_confidence = confidence_map or {}

    def set_path_data(self, path_data):
        self._path_data = path_data or {}

    def _get_prior_confidence(self, pos_triples, device, dtype):
        if not self._prior_confidence:
            return torch.zeros(len(pos_triples), device = device, dtype = dtype)
        values = [self._prior_confidence.get(tuple(triple), 0.0) for triple in pos_triples]
        return torch.tensor(values, device = device, dtype = dtype)

    def _calc_path_distance(self, rel_id, rel_path, rel_emb, num_relations):
        vec = rel_emb[rel_id].clone()
        for pid in rel_path:
            pid = int(pid)
            if pid < num_relations:
                vec = vec - rel_emb[pid]
            else:
                vec = vec + rel_emb[pid - num_relations]
        if self.adaptive_use_l1:
            return vec.abs().sum()
        return (vec * vec).sum()

    def _get_adaptive_confidence(self, pos_triples, model, device, dtype):
        if not self._path_data or not hasattr(model, "relation_embeddings"):
            return torch.zeros(len(pos_triples), device = device, dtype = dtype)
        rel_emb = model.relation_embeddings.weight.detach()
        num_relations = rel_emb.shape[0]
        values = []
        for triple in pos_triples:
            path_list = self._path_data.get(tuple(triple))
            if not path_list:
                values.append(0.0)
                continue
            all_path_conf = 0.0
            rel_id = int(triple[1])
            for rel_path, pr in path_list:
                dist = self._calc_path_distance(rel_id, rel_path, rel_emb, num_relations)
                all_path_conf += float(pr) / max(float(dist), 1e-12)
            soft_conf = 1.0 / (1.0 + math.exp(-all_path_conf))
            values.append(soft_conf)
        return torch.tensor(values, device = device, dtype = dtype)

    def forward(self, pred, target, current_epoch = None, x_batch = None, model = None):
        if not torch.is_tensor(target):
            if isinstance(target, (list, tuple)):
                tensors = [item for item in target if torch.is_tensor(item)]
                if tensors:
                    target = None
                    if pred is not None:
                        for item in tensors:
                            if item.numel() == pred.numel():
                                target = item
                                break
                    if target is None:
                        target = tensors[-1]
            if target is None:
                raise RuntimeError("LocalTripleWithPriorAndAdaptivePathLoss expects target to be a torch.Tensor.")
        if x_batch is None:
            raise RuntimeError("LocalTripleWithPriorAndAdaptivePathLoss requires x_batch to track per-triple confidence.")
        if model is None:
            raise RuntimeError("LocalTripleWithPriorAndAdaptivePathLoss requires model to compute adaptive path confidence.")
        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)
        if pred.shape[0] != target.shape[0]:
            raise RuntimeError("LocalTripleWithPriorAndAdaptivePathLoss expects pred and target to have matching shapes.")
        if x_batch.dim() != 2 or x_batch.shape[1] != 3:
            raise RuntimeError("LocalTripleWithPriorAndAdaptivePathLoss expects x_batch to be a [N, 3] tensor of triples.")
        if x_batch.shape[0] != pred.shape[0]:
            raise RuntimeError("LocalTripleWithPriorAndAdaptivePathLoss expects x_batch and pred to align on the first dimension.")

        pos_mask = target > self.local_loss.positive_threshold
        pos_scores = pred[pos_mask]
        neg_scores = pred[~pos_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return pred.new_tensor(0.0, requires_grad = True)

        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if neg_count % pos_count == 0:
            neg_ratio = neg_count // pos_count
            neg_scores = neg_scores.view(neg_ratio, pos_count)
            if self.local_loss.use_max_negative:
                neg_per_pos = neg_scores.max(dim = 0).values
            else:
                neg_per_pos = neg_scores.mean(dim = 0)
        else:
            neg_value = neg_scores.max() if self.local_loss.use_max_negative else neg_scores.mean()
            neg_per_pos = neg_value.expand_as(pos_scores)

        if self.local_loss.score_is_distance:
            e_pos = pos_scores
            e_neg = neg_per_pos
        else:
            e_pos = -pos_scores
            e_neg = -neg_per_pos

        delta = self.local_loss.margin + e_pos - e_neg
        hinge = F.relu(delta)

        pos_triples = x_batch[pos_mask].detach().cpu().tolist()
        conf_values = [self.local_loss._confidence.get(tuple(triple), 1.0) for triple in pos_triples]
        local_conf = pred.new_tensor(conf_values)
        prior_conf = self._get_prior_confidence(pos_triples, device = pred.device, dtype = pred.dtype)
        adaptive_conf = self._get_adaptive_confidence(pos_triples, model = model, device = pred.device, dtype = pred.dtype)

        combined_conf = (
            self.lambda_1_lt * local_conf +
            self.lambda_2_pp * prior_conf +
            self.lambda_3_ap * adaptive_conf
        )
        loss = (combined_conf * hinge).mean()

        q_values = (-delta).detach()
        for triple, q_val, conf_val in zip(pos_triples, q_values.tolist(), conf_values):
            if q_val <= 0:
                new_conf = max(conf_val * self.local_loss.alpha, self.local_loss.min_conf)
            else:
                new_conf = min(conf_val + self.local_loss.beta, self.local_loss.max_conf)
            self.local_loss._confidence[tuple(triple)] = new_conf

        return loss
