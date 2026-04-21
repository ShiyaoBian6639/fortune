"""
Regression loss functions for the deeptime pipeline.
"""

import torch
import torch.nn as nn
from scipy.stats import spearmanr


class MultiHorizonHuberLoss(nn.Module):
    """
    Weighted sum of per-horizon Huber losses.

    Huber(delta) is quadratic for |e| < delta, linear for |e| >= delta.
    With delta=1.0 (1% excess return), the loss handles Chinese ±10% limit
    events robustly while converging fast for the typical ±1-3% daily range.
    """

    def __init__(self, delta: float = 1.0, horizon_weights=None, num_horizons: int = 5):
        super().__init__()
        self.criterion = nn.HuberLoss(delta=delta, reduction='mean')
        if horizon_weights is None:
            horizon_weights = [1.0] * num_horizons
        w = torch.tensor(horizon_weights, dtype=torch.float32)
        self.register_buffer('weights', w / w.sum())

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:   (B, H) — predicted excess returns
            targets: (B, H) — actual excess returns
        """
        loss = sum(
            self.weights[h] * self.criterion(preds[:, h], targets[:, h])
            for h in range(preds.size(1))
        )
        return loss


class IC_Loss(nn.Module):
    """
    Rank-correlation loss: 1 - Spearman IC per horizon, averaged.

    Differentiable approximation: 1 - pearson(rank(pred), rank(target)).
    Using soft ranks via sigmoid approximation so gradients flow.
    """

    def __init__(self, num_horizons: int = 5, eps: float = 1e-8):
        super().__init__()
        self.num_horizons = num_horizons
        self.eps = eps

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable rank via pairwise comparisons: sum(x_i > x_j)."""
        # (B, B) matrix of soft comparisons
        diff = x.unsqueeze(0) - x.unsqueeze(1)          # (B, B)
        ranks = torch.sigmoid(diff / 0.1).sum(dim=1)    # (B,)
        return ranks

    def _pearson(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean()
        b = b - b.mean()
        return (a * b).sum() / (a.norm() * b.norm() + self.eps)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for h in range(self.num_horizons):
            p_rank = self._soft_rank(preds[:, h])
            t_rank = self._soft_rank(targets[:, h])
            ic = self._pearson(p_rank, t_rank)
            loss = loss + (1.0 - ic)
        return loss / self.num_horizons


class CombinedRegressionLoss(nn.Module):
    """
    alpha * MultiHorizonHuberLoss + (1-alpha) * IC_Loss.
    Default alpha=0.9 keeps Huber as primary driver with IC as regularizer.
    """

    def __init__(self, delta=1.0, horizon_weights=None, num_horizons=5, alpha=0.9):
        super().__init__()
        self.huber = MultiHorizonHuberLoss(delta, horizon_weights, num_horizons)
        self.ic    = IC_Loss(num_horizons)
        self.alpha = alpha

    def forward(self, preds, targets):
        return self.alpha * self.huber(preds, targets) + (1 - self.alpha) * self.ic(preds, targets)


def create_regression_loss(config: dict) -> nn.Module:
    delta   = config.get('huber_delta', 1.0)
    weights = config.get('horizon_weights', None)
    n_h     = config.get('num_horizons', 5)
    ltype   = config.get('loss_type', 'huber')
    if ltype == 'huber+ic':
        return CombinedRegressionLoss(delta, weights, n_h, alpha=0.9)
    return MultiHorizonHuberLoss(delta, weights, n_h)
