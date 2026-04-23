"""
Training loop for the deeptime regression pipeline.

Key differences from dl/training.py:
  - IC (rank correlation) instead of accuracy as primary metric
  - IC-based early stopping
  - Regression-specific batch unpacking (obs, future, targets, sector, ind, sub, size)
  - Gradient norm tracking for diagnostics
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr, pearsonr

from dl.training import set_seed

from .config import NUM_HORIZONS, FORWARD_WINDOWS, get_horizon_name


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank IC between prediction and actual excess return."""
    if len(pred) < 5:
        return 0.0
    try:
        return float(spearmanr(pred, target)[0])
    except Exception:
        return 0.0


def compute_hit_rate(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of correctly-signed predictions."""
    valid = np.isfinite(pred) & np.isfinite(target)
    if valid.sum() < 2:
        return 0.5
    return float(np.mean(np.sign(pred[valid]) == np.sign(target[valid])))


def compute_regression_metrics(
    all_preds:   np.ndarray,  # (N, H)
    all_targets: np.ndarray,  # (N, H)
) -> Dict:
    """Compute per-horizon and aggregate metrics."""
    metrics = {}
    ics = []
    for h in range(NUM_HORIZONS):
        p = all_preds[:, h]
        t = all_targets[:, h]
        valid = np.isfinite(p) & np.isfinite(t)
        pv, tv = p[valid], t[valid]
        ic   = compute_ic(pv, tv)
        mae  = float(np.mean(np.abs(pv - tv))) if len(pv) else 0.0
        rmse = float(np.sqrt(np.mean((pv - tv) ** 2))) if len(pv) else 0.0
        hr   = compute_hit_rate(pv, tv)
        name = get_horizon_name(h)
        metrics[f'ic_{name}']   = ic
        metrics[f'mae_{name}']  = mae
        metrics[f'rmse_{name}'] = rmse
        metrics[f'hr_{name}']   = hr
        ics.append(ic)
    metrics['ic_mean']  = float(np.mean(ics))
    metrics['mae_mean'] = float(np.mean([metrics[f'mae_{get_horizon_name(h)}'] for h in range(NUM_HORIZONS)]))
    return metrics


# ─── Training step ─────────────────────────────────────────────────────────────

def _unpack_batch(batch):
    """Unpack a batch tuple from RegressionMemmapDataset or ChunkedLoader."""
    obs, future, targets, sector, ind, sub, sz = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
    return obs, future, targets, sector, ind, sub, sz


def train_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    str,
    scaler:    Optional['torch.amp.GradScaler'] = None,
    max_grad_norm: float = 1.0,
    profile:   bool = False,
) -> Tuple[float, float, float, dict]:
    """
    Train for one epoch.

    Returns:
        (avg_loss, avg_ic, avg_grad_norm, profile_stats)
    """
    model.train()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []
    grad_norms  = []
    device_type = device.split(':')[0]

    # Profiling counters
    t_data_total = 0.0
    t_gpu_total  = 0.0
    t_last = time.time()

    # Real GPU utilization via nvidia-smi (pynvml)
    gpu_utils = []
    nvml_handle = None
    if profile:
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    for batch in loader:
        # Measure data loading time
        t_data_end = time.time()
        t_data_total += (t_data_end - t_last)
        if len(batch) < 7:
            continue
        def _t(x): return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
        obs     = _t(batch[0])
        future  = _t(batch[1])
        targets = _t(batch[2])
        sector  = _t(batch[3])
        ind     = _t(batch[4])
        sub     = _t(batch[5])
        sz      = _t(batch[6])
        # New static fields (indices 7-9); zero-fill if old cache without them
        area    = _t(batch[7])  if len(batch) > 7  else torch.zeros_like(sector)
        board   = _t(batch[8])  if len(batch) > 8  else torch.zeros_like(sector)
        ipo_age = _t(batch[9])  if len(batch) > 9  else torch.zeros_like(sector)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(scaler is not None)):
            preds = model(obs, future, sector, ind, sub, sz, area, board, ipo_age)
            loss  = criterion(preds, targets)

        # Skip batch if loss is NaN/inf (bad data or numerical instability)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
            # Skip optimizer step if gradients exploded
            if np.isfinite(gn):
                scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
            if np.isfinite(gn):
                optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        grad_norms.append(gn)

        all_preds.append(preds.detach().float().cpu().numpy())
        all_targets.append(targets.detach().float().cpu().numpy())

        # Measure GPU compute time (sync to get accurate timing)
        if profile and n_batches <= 100:  # only profile first 100 batches
            torch.cuda.synchronize()
            # Sample real GPU utilization
            if nvml_handle is not None:
                try:
                    import pynvml
                    util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                    gpu_utils.append(util.gpu)
                except Exception:
                    pass
        t_gpu_end = time.time()
        t_gpu_total += (t_gpu_end - t_data_end)
        t_last = t_gpu_end

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics     = compute_regression_metrics(all_preds, all_targets)

    avg_loss    = total_loss / max(n_batches, 1)
    # Use nanmean so a single inf/nan batch doesn't obscure the real average.
    # Also log the fraction of bad batches for diagnostics.
    finite_norms = [g for g in grad_norms if np.isfinite(g)]
    avg_gn = float(np.mean(finite_norms)) if finite_norms else float('nan')
    bad_frac = 1.0 - len(finite_norms) / max(len(grad_norms), 1)
    if bad_frac > 0.05:   # warn if >5% of batches had inf/nan gradients
        print(f"    [WARN] {bad_frac*100:.1f}% of batches had inf/nan gradients (skipped optimizer step)")

    # Profile stats
    profile_stats = {
        't_data': t_data_total,
        't_gpu': t_gpu_total,
        'n_batches': n_batches,
        'gpu_util': t_gpu_total / max(t_data_total + t_gpu_total, 1e-6) * 100,
        'real_gpu_util': float(np.mean(gpu_utils)) if gpu_utils else None,
    }
    return avg_loss, metrics.get('ic_mean', 0.0), avg_gn, profile_stats


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> Dict:
    """Evaluate on val or test split, return full metrics dict."""
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []

    for batch in loader:
        if len(batch) < 7:
            continue
        def _t(x): return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
        obs    = _t(batch[0]); future = _t(batch[1]); tgt = _t(batch[2])
        sec    = _t(batch[3]); ind    = _t(batch[4]); sub = _t(batch[5]); sz  = _t(batch[6])
        area   = _t(batch[7])  if len(batch) > 7  else torch.zeros_like(sec)
        board  = _t(batch[8])  if len(batch) > 8  else torch.zeros_like(sec)
        ipo    = _t(batch[9])  if len(batch) > 9  else torch.zeros_like(sec)

        preds = model(obs, future, sec, ind, sub, sz, area, board, ipo)
        loss  = criterion(preds, tgt)
        total_loss += loss.item()
        n_batches  += 1
        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(tgt.float().cpu().numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics     = compute_regression_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / max(n_batches, 1)
    return metrics, all_preds, all_targets


# ─── Full training loop ────────────────────────────────────────────────────────

def train_model(
    model:       nn.Module,
    train_loader,
    val_loader,
    config:      dict,
) -> Dict:
    """
    Full training loop with cosine LR, warmup, early stopping on validation IC.

    Returns:
        history dict with per-epoch losses, ICs, grad norms
    """
    from .losses import create_regression_loss

    device     = config.get('device', 'cpu')
    lr         = config.get('learning_rate', 5e-5)
    epochs     = config.get('epochs', 50)
    patience   = config.get('early_stopping_patience', 15)
    warmup_ep  = config.get('warmup_epochs', 8)
    use_amp    = config.get('use_amp', True) and device != 'cpu'
    max_gn     = config.get('max_grad_norm', 0.5)   # tighter than 1.0; see note below
    seed       = config.get('random_seed', 42)
    save_path  = config.get('model_save_path', 'stock_data/deeptime_model.pth')
    weight_decay = config.get('weight_decay', 0.05)  # raised from 0.01 to slow weight growth

    # Note on max_grad_norm:
    #   Raw pre-clip norms growing 0.5→20 during warmup indicate the loss surface
    #   has steep curvature. Clipping at 0.5 (was 1.0) keeps effective step direction
    #   stable and prevents weight-magnitude runaway that causes val-loss divergence.

    set_seed(seed)
    model = model.to(device)
    criterion = create_regression_loss(config).to(device)

    # ── Two-dimension LR scaling ──────────────────────────────────────────────
    #
    # 1. Batch-size scaling (sqrt): calibrated at batch=192.
    #    batch=512 → ×1.63; batch=192 → ×1.0
    #
    # 2. Dataset-size scaling (quartic-root): smaller datasets need lower peak LR.
    #    Evidence: 200 stocks best val IC at epoch 4 (LR=1.65e-5), basin escape
    #    above 2e-5. Formula: lr ∝ (n_stocks/ref_stocks)^0.25
    #    n_stocks=200  → ×0.67;  n_stocks=1000 → ×1.0;  n_stocks=5000 → ×1.50
    #    User can override both via --lr to disable auto-scaling.
    # LR scaling: disabled if user sets --no_lr_scale
    no_lr_scale = config.get('no_lr_scale', False)
    if no_lr_scale:
        scaled_lr = lr
        print(f"  LR = {lr:.2e} (scaling disabled)")
    else:
        base_batch   = config.get('base_batch_for_lr', 192)
        batch_size   = config.get('batch_size', 192)
        batch_scale  = (batch_size / base_batch) ** 0.5

        max_stocks   = config.get('max_stocks', 0)
        ref_stocks   = config.get('ref_stocks_for_lr', 1000)
        if max_stocks and max_stocks > 0 and max_stocks < ref_stocks * 10:
            # Only scale down; never scale up beyond the user-specified LR
            dataset_scale = min(1.0, (max_stocks / ref_stocks) ** 0.25)
        else:
            dataset_scale = 1.0   # full dataset or max_stocks=0 → no reduction

        scale     = batch_scale * dataset_scale
        scaled_lr = lr * scale
        if abs(scale - 1.0) > 0.05:
            parts = [f"√({batch_size}/{base_batch})={batch_scale:.2f} (batch)"]
            if abs(dataset_scale - 1.0) > 0.02:
                parts.append(f"({max_stocks}/{ref_stocks})^0.25={dataset_scale:.2f} (dataset size)")
            print(f"  LR scaled {lr:.2e} × {' × '.join(parts)} = {scaled_lr:.2e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scaled_lr,
        betas=(0.9, 0.95), weight_decay=weight_decay,
    )

    # LR schedule. Options:
    #   'cosine' (default): cosine warmup to peak, then cosine decay to 0.
    #   'flat':   cosine warmup to peak, then hold peak LR for the rest of training.
    #             Useful when the loss surface has a narrow basin the model escapes
    #             under decay-to-zero pressure — holding peak lets it converge
    #             inside the basin rather than ramping through it.
    # NOTE: Do NOT combine LambdaLR with ReduceLROnPlateau — LambdaLR resets
    # the LR to base_lr*lambda(epoch) each step, overriding any reduction.
    lr_schedule = str(config.get('lr_schedule', 'cosine')).lower()

    def _lr_lambda(epoch):
        if epoch < warmup_ep:
            # Cosine warmup 10%→100% of peak LR
            progress = epoch / max(warmup_ep, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 - np.cos(np.pi * progress))
        if lr_schedule == 'flat':
            return 1.0
        # cosine decay to 0
        progress = (epoch - warmup_ep) / max(epochs - warmup_ep, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    primary_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    scaler    = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_ic': [],   'val_ic': [],
        'lr': [],         'grad_norm': [],
    }

    best_val_ic  = -np.inf
    best_epoch   = 0
    patience_cnt = 0

    print(f"\nTraining on {device} | epochs={epochs} | lr={scaled_lr:.2e} | "
          f"warmup={warmup_ep} (cosine) | schedule={lr_schedule} | patience={patience}")
    print(f"  AMP={'on' if scaler else 'off'}")

    for epoch in range(epochs):
        t0 = time.time()
        # Profile first epoch to diagnose data loading bottleneck
        do_profile = (epoch == 0)
        tr_loss, tr_ic, gn, pstats = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, max_gn,
            profile=do_profile
        )
        val_metrics, _, _  = evaluate(model, val_loader, criterion, device)
        primary_scheduler.step()

        val_loss = val_metrics['loss']
        val_ic   = val_metrics['ic_mean']
        lr_now   = float(optimizer.param_groups[0]['lr'])

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_ic'].append(tr_ic)
        history['val_ic'].append(val_ic)
        history['lr'].append(lr_now)
        history['grad_norm'].append(gn)

        elapsed = time.time() - t0
        vram_str = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            vram_str = f" | VRAM {alloc:.1f}/{reserved:.1f}GB"

        # Show profile stats on first epoch
        if do_profile:
            time_pct = pstats['gpu_util']
            real_pct = pstats.get('real_gpu_util')
            data_ms = pstats['t_data'] / pstats['n_batches'] * 1000
            gpu_ms  = pstats['t_gpu']  / pstats['n_batches'] * 1000
            real_str = f" | SM util: {real_pct:.0f}%" if real_pct is not None else ""
            print(f"\n  [PROFILE] time ratio: {time_pct:.1f}% | data: {data_ms:.1f}ms/batch | gpu: {gpu_ms:.1f}ms/batch{real_str}")
            if real_pct is not None and real_pct < 50:
                print(f"  [WARN] Low SM util ({real_pct:.0f}%). Model too small for GPU. Try: --tft_hidden 256 --batch_size 2048\n")
            elif time_pct < 50:
                print(f"  [WARN] GPU starving for data! Consider: --num_workers 4 or larger --chunk_samples\n")

        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"loss {tr_loss:.4f}/{val_loss:.4f} | "
              f"IC {tr_ic:.4f}/{val_ic:.4f} | "
              f"lr={lr_now:.2e} | gn={gn:.2f} | {elapsed:.1f}s{vram_str}")

        # Per-horizon val IC
        for h in range(NUM_HORIZONS):
            hn = get_horizon_name(h)
            print(f"    {hn}: IC={val_metrics.get('ic_'+hn, 0):.4f}  "
                  f"MAE={val_metrics.get('mae_'+hn, 0):.4f}  "
                  f"HR={val_metrics.get('hr_'+hn, 0):.4f}")

        # Detect training instability (gradient explosion, val divergence)
        if len(history['grad_norm']) >= 3:
            recent_gn = history['grad_norm'][-3:]
            if all(g > 2.0 for g in recent_gn):
                print(f"\n  [WARN] Gradient norms consistently high ({recent_gn})")
                print(f"         Consider: --lr {lr*0.5:.1e} or --max_grad_norm 0.2")
        if len(history['val_loss']) >= 3:
            recent_vl = history['val_loss'][-3:]
            if recent_vl[-1] > recent_vl[0] * 1.1:  # val loss increased 10%+
                print(f"\n  [WARN] Validation loss diverging: {recent_vl[0]:.4f} → {recent_vl[-1]:.4f}")
                print(f"         Consider reducing learning rate or model size")

        # Early stopping on val IC
        if val_ic > best_val_ic + 1e-4:
            best_val_ic  = val_ic
            best_epoch   = epoch + 1
            patience_cnt = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save model weights + scalar metadata
            torch.save({
                'epoch':      epoch + 1,
                'model_state': model.state_dict(),
                'val_ic':     float(best_val_ic),
            }, save_path)
            # Save full history alongside checkpoint for later plotting
            import json as _json
            history_path = save_path.replace('.pth', '_history.json')
            _json.dump({k: [float(v) for v in vals] for k, vals in history.items()
                        if isinstance(vals, list)}, open(history_path, 'w'))
            print(f"    [BEST] val IC={best_val_ic:.4f} -- checkpoint saved")
        else:
            patience_cnt += 1
            print(f"    Patience {patience_cnt}/{patience}")
            if patience_cnt >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (best: epoch {best_epoch}, IC={best_val_ic:.4f})")
                break

    # Load best checkpoint (weights_only=True safe now — only model weights saved)
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        print(f"\n  Loaded best checkpoint (epoch {best_epoch}, val IC={best_val_ic:.4f})")

    history['best_epoch']  = best_epoch
    history['best_val_ic'] = best_val_ic
    return history


import os
