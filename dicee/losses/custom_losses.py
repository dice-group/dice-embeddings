import torch
from torch import nn
from torch.nn import functional as F
import math

class DefaultBCELoss(nn.Module):
    def __init__(self):
        super(DefaultBCELoss, self).__init__()

    def forward(self, pred, target, current_epoch):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)
        return final_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothness_ratio=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothness_ratio = smoothness_ratio

    def forward(self, pred, target):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)
        print("Smoothness Ratio: ", self.smoothness_ratio)
        return final_loss

class AdaptiveLabelSmoothingLoss(nn.Module):
    def __init__(self, min_smoothing_factor=0.01, max_smoothing_factor=0.2, smoothing_factor_step=0.01, initial_smoothing_factor=0.1):

        super(AdaptiveLabelSmoothingLoss, self).__init__()
        self.min_smoothing_factor = min_smoothing_factor
        self.max_smoothing_factor = max_smoothing_factor
        self.smoothing_factor_step = smoothing_factor_step  # Controls rate of alpha adjustment
        self.smoothing_factor = initial_smoothing_factor  # Initial smoothing factor
        self.prev_loss = None  # Store previous loss
        self.eps = 1e-14  # Small value to prevent log(0)

    def forward(self, logits, target):

        # log_softmax is more numerically stable
        pred = F.log_softmax(logits, dim=-1)

        # Compute smoothed target distribution
        num_classes = logits.size(-1)
        #smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor / num_classes
        smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor * (1 - target) / (num_classes - 1)

        # Compute KL divergence loss
        loss = F.kl_div(pred, smoothed_target, reduction="batchmean")

        # Dynamically adjust alpha based on loss trend
        if self.prev_loss is not None:
            loss_diff = loss.item() - self.prev_loss  # Check improvement
            if loss_diff > 0:  # Loss is increasing (bad) → Increase smoothing
                self.smoothing_factor = min(self.smoothing_factor + self.smoothing_factor_step, self.max_smoothing_factor)
            elif loss_diff < 0:  # Loss is decreasing (good) → Reduce smoothing
                self.smoothing_factor = max(self.smoothing_factor - self.smoothing_factor_step, self.min_smoothing_factor)

        # Store current loss for the next iteration
        self.prev_loss = loss.item()
        print(f"Adaptive smoothness ratio = {self.smoothing_factor}")

        return loss

class LabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.0):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        # Greater zero threshold
        self.gz_threshold = 0.1
        self.eps = 1e-14

    def forward(self, pred, target):
        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        print("Alpha: ", self.alpha)
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
        self.alpha_step = alpha_step  # Controls how fast alpha adjusts
        self.alpha = initial_alpha  # Initial alpha
        self.prev_loss = None  # Store previous loss
        self.eps = 1e-14
        self.gz_threshold = 0.1

    def forward(self, pred, target):
        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        # Compute mean loss
        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

            divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
            predc = torch.sum(pred * target, dim=-1)
            filtered_loss = torch.where(torch.gt(predc, 1. - self.alpha), torch.zeros_like(divergence),
                                        divergence)
            mean_final_loss = torch.mean(filtered_loss)

        # Update alpha based on loss trend
        if self.prev_loss is not None:
            loss_diff = mean_final_loss - self.prev_loss  # Check improvement
            if loss_diff > 0:  # Loss is increasing (worse) → Increase relaxation
                self.alpha = min(self.alpha + self.alpha_step, self.max_alpha)
            elif loss_diff < 0:  # Loss is decreasing (better) → Decrease relaxation
                self.alpha = max(self.alpha - self.alpha_step, self.min_alpha)

        # Store current loss for next iteration
        self.prev_loss = mean_final_loss
        print("Alpha: ", self.alpha, "Epoch: ")

        # Recompute target credal set using updated alpha
        with torch.no_grad():
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        # Calculate divergence
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
        print(current_epoch, self.alpha)

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
            final_loss = criterion(pred, target)
        else:
            criterion = LabelRelaxationLoss(alpha=self.alpha)
            final_loss = criterion(pred, target)
        return final_loss


class CombinedAdaptiveLSandAdaptiveLR(nn.Module):
    def __init__(self):
        super(CombinedAdaptiveLSandAdaptiveLR, self).__init__()
        self.adaptive_label_smoothing = AdaptiveLabelSmoothingLoss()  # Persist instance
        self.adaptive_label_relaxation = AdaptiveLabelRelaxationLoss()
        self.criterion = ''

    def forward(self, pred, target, current_epoch):
        final_loss = 0
        if current_epoch < 100:
            print("Combined Adaptive LS on epoch: ", current_epoch)
            final_loss = self.adaptive_label_smoothing(pred, target)
        else:
            print("Combined Adaptive LR on epoch: ", current_epoch)
            final_loss = self.adaptive_label_relaxation(pred, target)
        return final_loss