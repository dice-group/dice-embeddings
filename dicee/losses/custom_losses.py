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

        '''
        pred = tensor([8.5, -3.0, 6.2, -7.5])
        target = tensor([1, 0, 1, 0])  # 1 = positive triple, 0 = negative triple
        Target is a label tensor that routes each sample through the correct loss 
        branch (positive vs negative).
        '''

        loss = (
            pos_mask * (loss_pos + self.lambda_q * pos_violation)
            + neg_mask * (loss_neg + self.lambda_q * neg_violation)
        )

        return loss.mean()


class RoBoSS(nn.Module):
    """
    Implementation of RoBoSS loss function based on Eq. 8 of the paper.
    
    Paper Definition:
    L(u) = lambda * (1 - (a*u + 1) * exp(-a*u))  if u > 0
         = 0                                     if u <= 0
    
    where u = 1 - y * pred (margin violation)
    
    Parameters:
    -----------
    a : float
        Shape parameter controlling intensity of penalty 
    b : float
       Bounding parameter (lambda in paper) 
    margin : float
        Margin value (1.0 for SVM in paper)
    """
    def __init__(self, a_roboss=5.0, lambda_roboss=1.5, margin=1.0, normalize_scores=False):
        super().__init__()
        self.a_roboss = float(a_roboss)
        self.lambda_roboss = float(lambda_roboss)
        self.margin = float(margin)
        self.normalize_scores = normalize_scores
        self.eps = 1e-8

    def _normalize_scores(self, pred: torch.Tensor) -> torch.Tensor:
        if self.normalize_scores:
            return torch.tanh(pred)
        return pred

    def forward(self, pred, target, current_epoch = None):
        target = target.float()
        pred = pred.float()
        
    
        pred = self._normalize_scores(pred)
        
        # Convert [0, 1] to [-1, 1]
        if target.min() >= 0 and target.max() <= 1:
            target = 2 * target - 1
        
        # Calculate u (margin violation)
        # u = 1 - y(w^T x + b). 
        u = self.margin - (target * pred)
        
        # Calculate Loss based on Eq. 8
        # L = b * (1 - (a*u + 1) * exp(-a*u))
        
        # Term 1: (au + 1)
        term1 = (self.a_roboss * u) + 1.0
        
        # Term 2: exp(-au)
        term2 = torch.exp(-self.a_roboss * u)
        
        # lambda * (1 - term1 * term2)
        loss_values = self.lambda_roboss * (1.0 - (term1 * term2))
        
        # Apply condition: Loss is 0 if u <= 0 (Correctly classified)
        # assign a fixed loss of 0 for all samples with u < 0
        loss = torch.where(u > 0, loss_values, torch.zeros_like(loss_values))
        
        return loss.mean()


class AGCELoss(nn.Module):
    '''
    AGCE Loss adapted for Knowledge Graph Embeddings.
    
    Paper: "Asymmetric Loss Functions for Learning with Noisy Labels"
    Formula: L_q(u, i) = ((a+1)^q - (a + u_i)^q) / q
    
    This replaces Softmax with Sigmoid to handle multi-label nature of KGEs.
    '''
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
        """
        Parameters:
        -----------
        pred : torch.Tensor
            Raw scores/logits from KGE model.
            Shape: (batch_size, num_entities)
        labels : torch.Tensor
            Binary labels (0 or 1). 
            Shape: (batch_size, num_entities)
            1 indicates the entity is a valid tail, 0 indicates it is not.
        """
        y = labels
        # 1. Get Probabilities via Sigmoid
        # This maps scores to [0, 1]
        y_hat = torch.sigmoid(pred)
        
        # 2. Calculate 'p_target' (Probability of the CORRECT classification)
        # The paper defines u_i as the probability of the correct label.
        # - If label=1 (True Triplet): We want prob -> 1.
        # - If label=0 (False Triplet): We want prob -> 0 (so probability of "correctly rejecting" is 1-prob).
        
        u = y * y_hat + (1 - y) * (1 - y_hat)

        
        # Clamp for numerical stability
        u = torch.clamp(u, min=self.eps, max=1.0 - self.eps)

        # 3. Apply AGCE Formula
        # Formula: ((a+1)^q - (a + p_target)^q) / q 
        term1 = (self.agce_a + 1) ** self.agce_q
        term2 = (self.agce_a + u) **  self.agce_q
        
        loss = (term1 - term2) / self.agce_q
        
        # Mean over the batch and entities
        return loss.mean() * self.scale


class AULoss(nn.Module):
    '''
    Implementation of Asymmetric Unhinged Loss (AUL) for KGE 1-vs-All training.
    
    AUL (formerly referred to as L_p) is guaranteed to be completely asymmetric (robust) 
    when aul_p <= 1, and requires aul_a > 1 when aul_p > 1.
    Formula: L_p(u, i) = ((aul_a - u_i)^aul_p - (aul_a - 1)^aul_p) / aul_p
    
    Parameters:
    -----------
    aul_a : float, default=1.5
        Shift parameter. Must be > 1.0 (recommended by the paper for safety).
    aul_p : float, default=0.9
        Shape parameter. Setting aul_p <= 1.0 ensures the loss is 'Completely Asymmetric' (robust).
    eps : float
        Numerical stability constant.
    scale : float
        Scaling factor for the final loss value.
    '''
    def __init__(self, aul_a=1.5, aul_p=0.9, eps=1e-7, scale=1.0):
        super(AULoss, self).__init__()
        self.aul_a = float(aul_a)
        self.aul_p = float(aul_p)
        self.eps = eps
        self.scale = scale

        # assert self.aul_a > 1.0, "Parameter 'aul_a' must be > 1.0 for AUL."

    def forward(self, pred, labels, current_epoch = None):
        """
        Computes AUL for KGE by mapping logits to probabilities via Sigmoid.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Raw scores/logits from KGE model. Shape: (batch_size, num_entities)
        labels : torch.Tensor
            Binary labels (0 or 1). Shape: (batch_size, num_entities)
        
        Returns:
        --------
        torch.Tensor
            Scalar loss value.
        """
        y = labels
        # 1. Map scores/logits to probabilities [0, 1]
        y_hat = torch.sigmoid(pred)
        
        # 2. Calculate 'p_target' (Probability of the CORRECT classification)
        # p_target = y * P(y=1) + (1-y) * P(y=0)
        u = y * y_hat + (1 - y) * (1 - y_hat)

        # Clamp for numerical stability in power operations
        u = torch.clamp(u, min=self.eps, max=1.0 - self.eps)
        
        # 3. Apply AUL Formula: ((aul_a - p_target)^aul_p - (aul_a - 1)^aul_p) / aul_p
        term1 = (self.aul_a - u) ** self.aul_p
        term2 = (self.aul_a - 1) ** self.aul_p
        
        loss = (term1 - term2) / self.aul_p
        
        # Mean over all elements in the batch and entities
        return loss.mean()


class AELoss(nn.Module):
    '''
    Correct Implementation of Asymmetric Exponential Loss (AEL).
    
    Formula: L_a(u, i) = exp(-u_i / a)
    
    Parameters:
    -----------
    a : float, default=0.5
        Temperature parameter (must be > 0). It controls the asymmetry ratio 
        r(l) = exp(-1/a). Lower 'a' means higher loss for mistakes.
    eps : float
        Numerical stability constant.
    scale : float
        Scaling factor for the final loss value.
    '''
    def __init__(self, a_ael=0.5, eps=1e-7, scale=1.0):
        super(AELoss, self).__init__()
        self.a_ael = float(a_ael)
        self.eps = eps
        self.scale = scale

        # Validation based on paper constraints
        assert self.a_ael > 0, "Parameter 'a' must be > 0."

    def forward(self, pred, labels, current_epoch=None):
        """
        Computes AEL for KGE by mapping logits to probabilities via Sigmoid.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Raw scores/logits from KGE model. Shape: (batch_size, num_entities)
        labels : torch.Tensor
            Binary labels (0 or 1). Shape: (batch_size, num_entities)
        
        Returns:
        --------
        torch.Tensor
            Scalar loss value.
        """
        # 1. Map scores/logits to probabilities [0, 1]
        pred_probs = torch.sigmoid(pred)
        
        # 2. Calculate 'p_target' (Probability of the CORRECT classification)
        # p_target = y * P(y=1) + (1-y) * P(y=0)
        p_target = labels * pred_probs + (1 - labels) * (1 - pred_probs)

        # Clamp for numerical stability in the exponent
        # p_target is used as the negative exponent input
        p_target = torch.clamp(p_target, min=self.eps, max=1.0)
        
        # 3. Apply AEL Formula: exp(-p_target / a)
        # This function increases as p_target (correct probability) decreases.
        loss = torch.exp(-p_target / self.a_ael)
        
        # Mean over all elements in the batch and entities
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
    """
    Syn-margin loss from "A Margin-based Loss with Synthetic Negative Samples for Continuous-output Machine Translation"
    (Bhat et al., WNGT 2019). This adaptation creates synthetic negatives directly from predicted and target embeddings.
    For KGE models we obtain a predicted embedding by weighting the entity embedding table with the model scores.
    """

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
    """
    Implementation of the Wave Loss Function from "Advancing Supervised Learning 
    with the Wave Loss Function" (Akhtar et al., 2024).

    This implementation adheres to the properties defined in Section 2.1:
    1. Smoothness: Differentiable everywhere (no truncation like Hinge/ReLU)[cite: 486].
    2. Noise Insensitivity: Applies loss even when u <= 0 (correctly classified).
    3. Boundedness: The loss is bounded by 1/lambda.

    Mathematical formulation (Eq 10)[cite: 480]:
        L_wave(u) = (1/λ) * (1 - 1/(1 + λ*u²*exp(wave_a*u)))

    where u (ξ in the paper) is defined as: u = 1 - y * pred.
    """
    def __init__(
        self,
        wave_a: float = 1.5,
        lambda_param: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            wave_a (float): Shape parameter 'a'[cite: 482].
            lambda_param (float): Bounding parameter 'lambda'[cite: 482].
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.wave_a = float(wave_a)
        self.lambda_param = float(lambda_param)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch = None):
        """
        Args:
            pred: Predicted scores (w^T x + b).
            target: Labels. If binary {0,1}, they are converted to {-1,1} as per SVM standard.
        """
        target = target.float()
        pred = pred.float()

        # The paper defines y in {-1, 1}[cite: 262].
        # If targets are [0, 1], convert them.
        if target.min() >= 0 and target.max() <= 1:
            target = 2 * target - 1 

        # Calculate u (denoted as xi in Eq 14 and 15)
        # The standard formulation uses a hard margin of 1.
        u = 1.0 - (target * pred)

        # Implementation of Eq (10)
        u_sq = u * u
        exp_term = torch.exp(self.wave_a * u)
        
        # Denominator: 1 + λ * u^2 * e^(wave_a*u)
        denom = 1.0 + self.lambda_param * u_sq * exp_term + self.eps

        # Final Loss: (1/λ) * (1 - 1/denominator)
        wave_loss = (1.0 / self.lambda_param) * (1.0 - 1.0 / denom)

        # Return mean loss over the batch (standard practice for optimization)
        return wave_loss.mean()

class testLoss(nn.Module):
    def __init__(self, eps = 1e-8):

        super().__init__()
        self.eps = eps

    def forward(self, pred, target, current_epoch = None):

        y_hat = torch.sigmoid(pred)
        y_hat = torch.clamp(y_hat, min = self.eps, max= 1.0 - self.eps)
        
        y = target.float() 
        loss = -(y * torch.log(y_hat)+ (1 - y) * torch.log(1 - y_hat))

        return loss.mean()

class NSSALoss(nn.Module):
    """
    Negative Sampling Self-Adversarial (NSSA) loss.

    """

    def __init__(
        self,
        temperature =  1.0,
        positive_index = 0,
        use_target_argmax = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.positive_index = positive_index
        self.use_target_argmax = use_target_argmax

    def forward(
        self,
        pred,
        target,
        current_epoch=None
    ):
        temperature = self.temperature


        # pos_idx = target.argmax(dim=1)
        
        # ratio = (pos_idx == 0).float().mean().item()
        # print(f"Positive-at-0 ratio: {ratio:.4f}")

        # if pred.size(1) == 1:
        #     pos_pred = pred.squeeze(1)
        #     neg_pred = pred.new_empty((pred.size(0), 0))

        if self.use_target_argmax:
            pos_idx = target.argmax(dim=1)
        else:
            pos_idx = torch.full(
                (pred.size(0),),
                self.positive_index,
                device=pred.device,
                dtype=torch.long,
            )

        pos_pred = pred.gather(1, pos_idx[:, None]).squeeze(1)

        neg_mask = torch.ones_like(pred, dtype=torch.bool)
        neg_mask.scatter_(1, pos_idx[:, None], False)
        neg_pred = pred.masked_select(neg_mask).view(pred.size(0), pred.size(1) - 1)

        # Positive term: -log sigma(pos)
        pos_score = F.logsigmoid(pos_pred)

        # Negative term: self-adversarial weighted log-sigmoid on negatives
        weights = F.softmax(neg_pred * temperature, dim=1).detach()
        neg_score = (weights * F.logsigmoid(-neg_pred)).sum(dim=1)

        pos_loss = -pos_score.mean()
        neg_loss = -neg_score.mean()

        loss = (pos_loss + neg_loss) / 2

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma = 2.0):
        super().__init__() 

        self.gamma = gamma 

    def forward(self, pred, target, current_epoch = None):

        p = pred.sigmoid()

        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        pt = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - pt) ** self.gamma)

        return loss

