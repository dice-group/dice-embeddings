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
        self.eps = 1e-9  # Small epsilon for numerical stability

    def forward(self, pred, target):
        if self.logits_provided:
            pred = F.softmax(pred, dim=self.dim)

        # Ensure predictions are stable (minimal clamping)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        with torch.no_grad():
            sum_y_hat_prime = torch.sum((1.0 - target) * pred, dim=-1)

            sum_y_hat_prime = sum_y_hat_prime + self.eps

            pred_hat = self.alpha * pred / sum_y_hat_prime.unsqueeze(dim=-1)

            target_credal = torch.where(
                target > self.gz_threshold,
                1.0 - self.alpha,
                pred_hat
            )

        target_credal = torch.clamp(target_credal, min=self.eps, max=1.0)

        divergence = F.kl_div(pred.log(), target_credal, log_target=False, reduction="none")
        divergence = torch.sum(divergence, dim=-1)

        pred_alignment = torch.sum(pred * target, dim=-1)

        mask = pred_alignment > (1.0 - self.alpha)
        mask = mask.float()

        result = (1.0 - mask) * divergence

        return torch.mean(result)


