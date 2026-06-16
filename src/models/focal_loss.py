"""
Focal Loss implementation for severe class imbalance in flare forecasting.

Based on:
    Lin et al. 2017, "Focal Loss for Dense Object Detection", ICCV
    https://arxiv.org/abs/1708.02002

Flare-positive windows are <1% of all windows. Focal loss down-weights
easy negative samples and focuses training on hard, misclassified examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        alpha: class weighting factor (alpha for positive class)
        gamma: focusing parameter (gamma=0 -> CE; gamma>0 focuses on hard examples)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: predicted probabilities [B, H] (after sigmoid)
            targets: binary labels [B, H]
        Returns:
            scalar loss
        """
        p = inputs.clamp(min=1e-7, max=1.0 - 1e-7)
        ce_loss = -targets * torch.log(p) - (1.0 - targets) * torch.log(1.0 - p)

        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        modulating = (1.0 - p_t) ** self.gamma

        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        loss = alpha_t * modulating * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted binary cross-entropy as simpler alternative to focal loss."""

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = targets * self.pos_weight + (1.0 - targets) * 1.0
        loss = F.binary_cross_entropy(inputs, targets, weight=weight)
        return loss
