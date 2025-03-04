import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class DefaultBCELoss(nn.Module):
    def __init__(self):
        super(DefaultBCELoss, self).__init__()

    def forward(self, pred, target, current_epoch, gradient_norm):
        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)
        return final_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothness_ratio=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothness_ratio = smoothness_ratio

    def forward(self, pred, target, current_epoch, gradient_norm):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)

        return final_loss

class AdaptiveLabelSmoothingLoss(nn.Module):
    def __init__(self, min_smoothing_factor=0.01, max_smoothing_factor=0.2, smoothing_factor_step=0.01, initial_smoothing_factor=0.1):

        super(AdaptiveLabelSmoothingLoss, self).__init__()
        self.min_smoothing_factor = min_smoothing_factor
        self.max_smoothing_factor = max_smoothing_factor
        self.smoothing_factor_step = smoothing_factor_step
        self.smoothing_factor = initial_smoothing_factor
        self.prev_loss = None
        self.eps = 1e-14

    def forward(self, logits, target, current_epoch, gradient_norm):

        pred = F.log_softmax(logits, dim=-1)

        num_classes = logits.size(-1)
        #smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor / num_classes
        smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor * (1 - target) / (num_classes - 1)

        loss = F.kl_div(pred, smoothed_target, reduction="batchmean")

        if self.prev_loss is not None:
            loss_diff = loss.item() - self.prev_loss
            if loss_diff > 0:
                self.smoothing_factor = min(self.smoothing_factor + self.smoothing_factor_step, self.max_smoothing_factor)
            elif loss_diff < 0:
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

    def forward(self, pred, target, current_epoch, gradient_norm):
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

    def forward(self, pred, target, current_epoch, gradient_norm):
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

    def forward(self, pred, target, current_epoch, gradient_norm):
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

    def forward(self, pred, target, current_epoch, gradient_norm):
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

    def forward(self, pred, target, current_epoch, gradient_norm):
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

    def forward(self, pred, target, current_epoch, gradient_norm):
        final_loss = 0

        Smoothing_criterion = LabelSmoothingLoss(smoothness_ratio=self.smoothness_ratio)
        Smoothing_loss = Smoothing_criterion(pred, target, current_epoch)

        Relaxation_criterion = LabelRelaxationLoss(alpha=self.alpha)
        Relaxation_loss = Relaxation_criterion(pred, target, current_epoch)

        w = 0.4
        final_loss = (w * Smoothing_loss) + ((1- w) * Relaxation_loss)

        return final_loss

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
