"""
Loss functions for handling class imbalance in multi-horizon prediction.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-class classification.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (Tensor or None)
        gamma: Focusing parameter (default=2.0)
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor (default=0.0)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cast to float32: prevents NaN when float16 logits overflow to inf
        # (which makes softmax produce inf/inf = NaN).  Explicit cast is safe
        # inside torch.autocast — autocast only auto-promotes; explicit casts win.
        inputs = inputs.float()
        num_classes = inputs.size(1)

        # log_softmax uses the log-sum-exp trick internally: numerically stable
        # even when logits span a wide range.  torch.softmax(x).log() is not.
        log_p = torch.nn.functional.log_softmax(inputs, dim=1)
        p     = torch.exp(log_p)   # probabilities, derived from stable log_p

        # Build target distribution (with optional label smoothing)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        if self.label_smoothing > 0:
            targets_one_hot = (
                targets_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        # Cross-entropy via log_p (no log(p+eps) needed — log_p is always finite)
        ce = -(targets_one_hot * log_p)

        # Focal modulating factor
        p_t = (p * targets_one_hot).sum(dim=1, keepdim=True)
        focal_weight = (1 - p_t) ** self.gamma

        # Sum over classes to get per-sample loss, then apply focal weight
        focal_loss = (focal_weight * ce).sum(dim=1)   # (batch,)

        # Apply per-sample class weight for the TRUE class only.
        # Using alpha[targets] (shape: batch) instead of expanding the full
        # alpha vector means the weight is applied once per sample to the
        # aggregated loss — correct for both label_smoothing=0 and >0.
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]   # (batch,)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiHorizonLoss(nn.Module):
    """
    Weighted sum of per-horizon losses for multi-horizon prediction.

    logits: (B, H, C) — stacked output of TransformerClassifier
    labels: (B, H)    — int64 class index per horizon

    Loss = Σ_h  weights[h] * base_loss(logits[:, h, :], labels[:, h])

    Weights are normalised to sum to 1 so the total loss magnitude does
    not scale with the number of horizons.
    """

    def __init__(self, base_loss: nn.Module, horizon_weights: List[float]):
        super().__init__()
        self.base_loss = base_loss
        w = torch.tensor(horizon_weights, dtype=torch.float32)
        self.register_buffer('weights', w / w.sum())

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, H, C)
            labels: (B, H)
        Returns:
            scalar loss
        """
        H = logits.size(1)
        total = torch.zeros(1, device=logits.device, dtype=torch.float32).squeeze()
        for h in range(H):
            total = total + self.weights[h] * self.base_loss(logits[:, h, :], labels[:, h])
        return total


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
    https://arxiv.org/abs/1901.05555

    Args:
        samples_per_class: Number of samples in each class
        beta: Hyperparameter for computing effective number (default=0.9999)
        gamma: Focal loss gamma parameter (default=0.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        samples_per_class: np.ndarray,
        beta: float = 0.9999,
        gamma: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        effective_num = np.where(effective_num == 0, 1e-8, effective_num)

        # Compute weights
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)

        self.weights = torch.FloatTensor(weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()  # FP32 for numerical stability under autocast
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)

        p = torch.softmax(inputs, dim=1)
        weights_for_samples = self.weights[targets]

        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        if self.gamma > 0:
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma
            ce_loss = focal_weight * ce_loss

        cb_loss = weights_for_samples * ce_loss

        if self.reduction == 'mean':
            return cb_loss.mean()
        elif self.reduction == 'sum':
            return cb_loss.sum()
        return cb_loss


class CombinedMultiHorizonLoss(nn.Module):
    """
    Combined loss: absolute-return classification + relative-return auxiliary.

    combined = cls_loss(cls_logits, cls_labels)
             + rel_weight * rel_loss(rel_logits, rel_labels)

    cls_logits: (B, H, C)   — primary absolute-return classification heads
    rel_logits: (B, H, C_r) — auxiliary relative-return (stock − CSI300) heads
    cls_labels: (B, H)      — absolute-return class indices
    rel_labels: (B, H)      — relative-return class indices

    When called, expects model_output to be a 2-tuple (cls_logits, rel_logits).
    """

    def __init__(
        self,
        cls_loss:   nn.Module,
        rel_loss:   nn.Module,
        rel_weight: float = 0.3,
    ):
        super().__init__()
        self.cls_loss   = cls_loss
        self.rel_loss   = rel_loss
        self.rel_weight = rel_weight

    def forward(
        self,
        model_out:  tuple,            # (cls_logits (B,H,C), rel_logits (B,H,C_r))
        cls_labels: torch.Tensor,     # (B, H)
        rel_labels: torch.Tensor,     # (B, H)
    ) -> torch.Tensor:
        cls_logits, rel_logits = model_out
        loss_cls = self.cls_loss(cls_logits, cls_labels)
        loss_rel = self.rel_loss(rel_logits, rel_labels)
        return loss_cls + self.rel_weight * loss_rel


def create_loss_function(
    loss_type: str,
    num_classes: int,
    class_counts: np.ndarray,
    device: str,
    gamma: float = 2.0,
    beta: float = 0.9999,
    label_smoothing: float = 0.0,
    use_class_weights: bool = True,
    horizon_weights: Optional[List[float]] = None,
    num_horizons: int = 1,
    use_relative_head: bool = False,
    num_relative_classes: int = 5,
    relative_head_weight: float = 0.3,
) -> nn.Module:
    """
    Factory function to create the appropriate loss function.

    For multi-horizon training, wraps the base loss in MultiHorizonLoss
    which expects logits (B, H, C) and labels (B, H).

    Args:
        loss_type: 'ce' (CrossEntropy), 'focal' (Focal Loss), or 'cb' (Class-Balanced)
        num_classes: Number of classes
        class_counts: (C,) or (H, C) array of sample counts per class.
                      If 2D, averaged across horizons to compute class weights.
        device: Device to use
        gamma: Focal loss gamma parameter
        beta: Class-balanced loss beta parameter
        label_smoothing: Label smoothing factor
        use_class_weights: Whether to use class weights
        horizon_weights: Per-horizon loss weights (e.g. [1,1,1]). If None,
                         equal weights are used when num_horizons > 1.
        num_horizons: Number of prediction horizons.  When > 1 the base loss
                      is wrapped in MultiHorizonLoss.

    Returns:
        Loss function module (MultiHorizonLoss when num_horizons > 1)
    """
    # Flatten (H, C) class_counts to (C,) for class-weight computation
    counts_1d = np.array(class_counts)
    if counts_1d.ndim == 2:
        counts_1d = counts_1d.sum(axis=0)   # sum across horizons

    # Compute class weights (inverse frequency)
    if use_class_weights:
        weights = 1.0 / (counts_1d + 1)
        weights = weights / weights.sum() * num_classes
        class_weights = torch.FloatTensor(weights).to(device)
    else:
        class_weights = None

    if loss_type == 'focal':
        print(f"Using Focal Loss (gamma={gamma}, label_smoothing={label_smoothing})")
        base_loss = FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing
        )

    elif loss_type == 'cb':
        print(f"Using Class-Balanced Loss (beta={beta}, gamma={gamma})")
        base_loss = ClassBalancedLoss(
            samples_per_class=counts_1d,
            beta=beta,
            gamma=gamma
        )

    else:  # Default: CrossEntropy
        print(f"Using CrossEntropy Loss (label_smoothing={label_smoothing})")
        if label_smoothing > 0:
            base_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
        else:
            base_loss = nn.CrossEntropyLoss(weight=class_weights)

    if num_horizons > 1:
        hw = horizon_weights if horizon_weights is not None else [1.0] * num_horizons
        print(f"Wrapping in MultiHorizonLoss (horizons={num_horizons}, weights={hw})")
        cls_criterion = MultiHorizonLoss(base_loss, hw)
    else:
        cls_criterion = base_loss

    if use_relative_head:
        # Auxiliary relative-return CE loss — no class weights (5 buckets are balanced).
        # Use the same label_smoothing for consistency.
        rel_base = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        rel_hw   = [1.0] * num_horizons
        rel_criterion = MultiHorizonLoss(rel_base, rel_hw) if num_horizons > 1 else rel_base
        print(f"Adding CombinedMultiHorizonLoss (rel_weight={relative_head_weight}, "
              f"rel_classes={num_relative_classes})")
        return CombinedMultiHorizonLoss(cls_criterion, rel_criterion, relative_head_weight)

    return cls_criterion


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced batch sampling.

    Args:
        labels: Array of class labels

    Returns:
        WeightedRandomSampler instance
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

    return sampler
