"""
Training and evaluation logic for stock prediction model.
"""

import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: Optional['torch.amp.GradScaler'] = None,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Handles both single-horizon (logits: B×C, labels: B) and multi-horizon
    (logits: B×H×C, labels: B×H) by auto-detecting the logit shape.

    Args:
        scaler: GradScaler for AMP (FP16) training. Pass None to use FP32.

    Returns:
        Tuple of (average_loss, mean_accuracy_across_horizons)
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    device_type = device.split(':')[0]  # 'cuda' from 'cuda:0'

    _is_tft = getattr(model, '_is_tft', False)

    for batch in dataloader:
        sequences, labels, sectors = batch[0], batch[1], batch[2]
        industries       = batch[3] if len(batch) > 3 else None
        relative_labels  = batch[4] if len(batch) > 4 else None
        # future_inputs is always the last element when present (dim==3, shape (B,5,27))
        future_inputs = None
        if _is_tft and len(batch) >= 5:
            last = batch[-1]
            if isinstance(last, torch.Tensor) and last.dim() == 3:
                future_inputs = last
                # Exclude it from relative_labels if it was mis-assigned
                if relative_labels is future_inputs:
                    relative_labels = None
        sequences = sequences.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)
        sectors   = sectors.to(device, non_blocking=True)
        if industries is not None:
            industries = industries.to(device, non_blocking=True)
        if relative_labels is not None:
            relative_labels = relative_labels.to(device, non_blocking=True)
        if future_inputs is not None:
            future_inputs = future_inputs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=scaler is not None):
            if _is_tft and future_inputs is not None:
                out = model(sequences, future_inputs, sectors, industries)
            else:
                out = model(sequences, sectors, industries)   # (B, H, C) or tuple
            if isinstance(out, tuple):
                logits = out[0]
                loss = criterion(out, labels, relative_labels)
            else:
                logits = out
                loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        # Accuracy: per-horizon accuracy averaged, then averaged over batch
        with torch.no_grad():
            lg = logits.float()
            if lg.dim() == 3:                           # (B, H, C) multi-horizon
                H = lg.size(1)
                correct_h = sum(
                    (lg[:, h, :].argmax(dim=1) == labels[:, h]).float().sum().item()
                    for h in range(H)
                )
                total_correct += correct_h / H
            else:                                       # (B, C) single-horizon
                total_correct += (lg.argmax(dim=1) == labels).float().sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = False,
    temperature_scaler: Optional[nn.Module] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model.

    Handles both single-horizon (logits: B×C) and multi-horizon (logits: B×H×C).

    Args:
        use_amp: Whether to use FP16 autocast during inference.
        temperature_scaler: Optional TemperatureScaler to calibrate logits.

    Returns:
        Tuple of (avg_loss, predictions, labels, probabilities)
        For multi-horizon:
          predictions: (N, H) — argmax class per horizon
          labels:      (N, H)
          probs:       (N, H, C)
        For single-horizon:
          predictions: (N,)
          labels:      (N,)
          probs:       (N, C)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    device_type = device.split(':')[0]

    _is_tft = getattr(model, '_is_tft', False)

    with torch.no_grad():
        for batch in dataloader:
            sequences, labels, sectors = batch[0], batch[1], batch[2]
            industries      = batch[3] if len(batch) > 3 else None
            relative_labels = batch[4] if len(batch) > 4 else None
            future_inputs = None
            if _is_tft and len(batch) >= 5:
                last = batch[-1]
                if isinstance(last, torch.Tensor) and last.dim() == 3:
                    future_inputs = last
                    if relative_labels is future_inputs:
                        relative_labels = None
            sequences = sequences.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)
            sectors   = sectors.to(device, non_blocking=True)
            if industries is not None:
                industries = industries.to(device, non_blocking=True)
            if relative_labels is not None:
                relative_labels = relative_labels.to(device, non_blocking=True)
            if future_inputs is not None:
                future_inputs = future_inputs.to(device, non_blocking=True)

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                if _is_tft and future_inputs is not None:
                    out = model(sequences, future_inputs, sectors, industries)
                else:
                    out = model(sequences, sectors, industries)   # (B, H, C) or tuple
                if isinstance(out, tuple):
                    logits = out[0]
                    loss = criterion(out, labels, relative_labels)
                else:
                    logits = out
                    loss = criterion(logits, labels)

            lg = logits.float()
            if temperature_scaler is not None:
                lg = temperature_scaler(lg)

            total_loss += loss.item()

            if lg.dim() == 3:                           # (B, H, C) multi-horizon
                probs = torch.softmax(lg, dim=2)        # (B, H, C)
                preds = lg.argmax(dim=2)                # (B, H)
            else:                                       # (B, C) single-horizon
                probs = torch.softmax(lg, dim=1)        # (B, C)
                preds = lg.argmax(dim=1)                # (B,)

            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels      = np.concatenate(all_labels,      axis=0)
    all_probs       = np.concatenate(all_probs,        axis=0)

    return avg_loss, all_predictions, all_labels, all_probs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: dict,
    device: str
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Full training loop with early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        config: Configuration dictionary
        device: Device to use

    Returns:
        Tuple of (best_model, training_history)
    """
    use_amp = config.get('use_amp', True) and device.split(':')[0] == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision (FP16) enabled — using AMP GradScaler")

    # betas=(0.9, 0.95): Karpathy's nanoGPT setting.
    # β₂=0.95 decays the second moment faster than the default 0.999,
    # making the optimizer more responsive to recent gradient magnitudes —
    # useful for non-stationary financial time-series data.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    # Linear warmup → cosine decay schedule.
    # Warmup prevents unstable large-gradient updates in the first few epochs
    # when the model hasn't yet formed reliable feature representations.
    # After warmup, cosine decay gives each epoch a meaningful LR.
    warmup_epochs = config.get('warmup_epochs', 5)
    total_epochs  = config['epochs']
    cosine_epochs = max(1, total_epochs - warmup_epochs)

    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,   # start at 10% of LR
        end_factor   = 1.0,
        total_iters  = warmup_epochs,
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-7
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers  = [warmup_sched, cosine_sched],
        milestones  = [warmup_epochs],
    )

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = config.get('early_stopping_patience', 10)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    print("\n" + "-" * 40)
    print("Training")
    print("-" * 40)

    for epoch in range(config['epochs']):
        t0 = time.perf_counter()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate
        val_loss, val_predictions, val_labels_true, _ = evaluate(model, val_loader, criterion, device, use_amp)
        # Multi-horizon: average per-horizon accuracy; single-horizon: standard accuracy
        if val_labels_true.ndim == 2:
            val_acc = float(np.mean([
                accuracy_score(val_labels_true[:, h], val_predictions[:, h])
                for h in range(val_labels_true.shape[1])
            ]))
        else:
            val_acc = accuracy_score(val_labels_true, val_predictions)

        epoch_secs = time.perf_counter() - t0

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Step cosine schedule (no metric needed — fires every epoch)
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        lr_tag = f" lr={new_lr:.2e}"

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"patience={patience_counter}/{patience}{lr_tag} | "
              f"{epoch_secs:.1f}s")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.

    For multi-horizon inputs (2D arrays), metrics are computed per horizon
    and averaged.  The returned dict always has flat keys (accuracy, f1, …)
    plus per-horizon keys when H > 1 (accuracy_h0, accuracy_h1, …).
    """
    if y_true.ndim == 2:                         # (N, H) multi-horizon
        H = y_true.shape[1]
        agg: Dict[str, float] = {'accuracy': 0.0, 'precision': 0.0,
                                  'recall': 0.0,   'f1': 0.0}
        for h in range(H):
            m = {
                'accuracy':  accuracy_score(y_true[:, h], y_pred[:, h]),
                'precision': precision_score(y_true[:, h], y_pred[:, h],
                                             average='weighted', zero_division=0),
                'recall':    recall_score(y_true[:, h], y_pred[:, h],
                                          average='weighted', zero_division=0),
                'f1':        f1_score(y_true[:, h], y_pred[:, h],
                                      average='weighted', zero_division=0),
            }
            for k, v in m.items():
                agg[k] += v / H
            for k, v in m.items():
                agg[f'{k}_h{h}'] = v
        return agg

    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    num_horizons: int = 1,
) -> nn.Module:
    """
    Fit per-horizon temperature scalers on the validation set (post-training).

    Guo et al. (2017) "On Calibration of Modern Neural Networks".
    Minimises NLL on the validation set via LBFGS, keeping the backbone frozen.
    Only the temperature parameters are updated.

    Args:
        model:        Trained TransformerClassifier (frozen during this step).
        val_loader:   Validation DataLoader.
        device:       Device string.
        num_horizons: Number of prediction horizons (H).

    Returns:
        Fitted TemperatureScaler (already on `device`).
    """
    import torch.nn.functional as F
    from torch.optim import LBFGS
    from .models import TemperatureScaler

    temp_scaler = TemperatureScaler(num_horizons).to(device)

    # Collect all raw logits and labels from the validation set.
    # Use no_grad here — gradients are only needed w.r.t. temperatures below.
    _is_tft = getattr(model, '_is_tft', False)
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            sequences, labels, sectors = batch[0], batch[1], batch[2]
            industries    = batch[3] if len(batch) > 3 else None
            future_inputs = None
            if _is_tft and len(batch) >= 5:
                last = batch[-1]
                if isinstance(last, torch.Tensor) and last.dim() == 3:
                    future_inputs = last.to(device)
            sequences = sequences.to(device)
            sectors   = sectors.to(device)
            if industries is not None:
                industries = industries.to(device)
            if _is_tft and future_inputs is not None:
                out = model(sequences, future_inputs, sectors, industries)
            else:
                out = model(sequences, sectors, industries)
            # When using relative head, model returns (cls_logits, rel_logits); take cls only
            cls_logits = out[0] if isinstance(out, tuple) else out
            logits_list.append(cls_logits.float().cpu())
            labels_list.append(labels.cpu())

    all_logits = torch.cat(logits_list, dim=0).to(device)   # (N, H, C) or (N, C)
    all_labels = torch.cat(labels_list, dim=0).to(device)   # (N, H) or (N,)

    optimizer = LBFGS([temp_scaler.temperatures], lr=0.1, max_iter=200,
                      line_search_fn='strong_wolfe')

    def _eval():
        optimizer.zero_grad()
        scaled = temp_scaler(all_logits)
        if scaled.dim() == 3:                           # multi-horizon (N, H, C)
            H = scaled.size(1)
            loss = sum(
                F.cross_entropy(scaled[:, h, :], all_labels[:, h])
                for h in range(H)
            ) / H
        else:                                           # single-horizon (N, C)
            loss = F.cross_entropy(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(_eval)

    temps = temp_scaler.temperatures.detach().tolist()
    print(f"Temperature scaling fitted — T: {[f'{t:.4f}' for t in temps]}")
    return temp_scaler


def save_model(
    model: nn.Module,
    config: dict,
    scaler,
    history: dict,
    metrics: dict,
    save_path: str,
    temperature_scaler: Optional[nn.Module] = None,
):
    """Save model checkpoint (with optional temperature scaler)."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'history': history,
        'metrics': metrics,
    }
    if temperature_scaler is not None:
        checkpoint['temperatures'] = temperature_scaler.temperatures.detach().cpu()
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def load_model(model: nn.Module, checkpoint_path: str, device: str) -> Tuple[nn.Module, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint
