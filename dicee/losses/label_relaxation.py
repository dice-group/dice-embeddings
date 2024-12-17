import torch
from torch import nn
from torch.nn import functional as F

class LabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.1, dim=-1, logits_provided=True, one_hot_encode_trgts=False, num_classes=-1):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        self.dim = dim
        self.gz_threshold = 0.1
        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts
        self.num_classes = num_classes

    def forward(self, pred, target):
        if self.logits_provided:
            pred = pred.softmax(dim=self.dim)

        # Apply one-hot encoding to targets
        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=self.num_classes)

        # Construct credal set
        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        # Calculate divergence
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)

        pred = torch.sum(pred * target, dim=-1)

        result = torch.where(torch.gt(pred, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        return torch.mean(result)