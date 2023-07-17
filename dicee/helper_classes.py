import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomArg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def update(self, x: dict):
        self.kwargs.update(x)

    def __getattr__(self, name):
        return self.kwargs[name]

    def __repr__(self):
        return f'CustomArg at {hex(id(self))}: ' + str(self.kwargs)

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for k, v in self.kwargs.items():
            yield k, v


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

        print("Using Label Smoothing...")

    def forward(self, pred, target):
        # Log softmax is used for numerical stability
        pred = pred.log_softmax(dim=self.dim)

        print(pred.shape)
        print(target.shape)
        exit(1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.1, dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=-1):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        self.dim = dim

        # Greater zero threshold
        self.gz_threshold = 0.1

        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts

        self.num_classes = num_classes

    def forward(self, pred, target):
        if self.logits_provided:
            pred = pred.softmax(dim=self.dim)

        # with torch.no_grad():
        # Apply one-hot encoding to targets
        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=pred.shape[1])

        sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)

        pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
        target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)
        divergence = nn.functional.kl_div(pred.log(), target_credal, log_target=False)

        pred = torch.sum(pred * target, dim=-1)

        result = torch.where(torch.gt(pred, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        return torch.mean(result)
