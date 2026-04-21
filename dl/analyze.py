"""
Post-training analysis: feature importance and sector-wise performance.

Usage:
    python -m dl.analyze [--data_dir stock_data] [--device cuda]
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    FEATURE_COLUMNS, NUM_CLASSES, NUM_HORIZONS, get_class_names,
    NUM_RELATIVE_CLASSES,
)
from .data_processing import load_sector_data
from .memmap_dataset import load_memmap_datasets
from .models import create_model, TemperatureScaler
from .training import load_model


def _load_checkpoint(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    return checkpoint, config


def _build_model(config: dict, input_dim: int, num_sectors: int,
                 num_industries: int, device: str) -> nn.Module:
    model = create_model(config, input_dim, num_sectors, num_industries)
    return model.to(device)


def _build_sector_map(data_dir: str) -> Dict[int, str]:
    sector_data = load_sector_data(data_dir)
    if sector_data.empty:
        return {}
    unique_sectors = sorted(sector_data['sector'].unique())
    sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
    sector_to_id['Unknown'] = len(sector_to_id)
    return {v: k for k, v in sector_to_id.items()}


# ---------------------------------------------------------------------------
# 1. Gradient-based feature importance
# ---------------------------------------------------------------------------

def compute_gradient_importance(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    n_batches: int = 50,
) -> np.ndarray:
    """
    Compute per-feature importance as mean |d(loss)/d(input)| over the test set.

    For each batch we backprop through the summed per-horizon cross-entropy loss
    w.r.t. the input sequence tensor. We then take the absolute value, average
    over (batch, sequence_position) and accumulate across batches.

    Returns:
        importance: (n_features,) float32 array
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    accum = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    seen = 0

    for b_idx, batch in enumerate(test_loader):
        if b_idx >= n_batches:
            break

        sequences, labels, sectors = batch[0], batch[1], batch[2]
        industries = batch[3] if len(batch) > 3 else None

        seq = sequences.to(device).requires_grad_(True)
        labels = labels.to(device)
        sectors = sectors.to(device)
        if industries is not None:
            industries = industries.to(device)

        out = model(seq, sectors, industries)
        logits = out[0] if isinstance(out, tuple) else out  # (B, H, C)

        # Sum CE loss across horizons
        loss = sum(
            criterion(logits[:, h, :], labels[:, h])
            for h in range(logits.size(1))
        )
        loss.backward()

        # seq.grad: (B, seq_len, n_features) — average |grad| over B and seq_len
        grad = seq.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy()
        accum += grad
        seen += 1

    if seen > 0:
        accum /= seen
    return accum.astype(np.float32)


def print_feature_importance(importance: np.ndarray, top_k: int = 30):
    ranked = np.argsort(importance)[::-1]
    print(f"\n{'Feature Importance (gradient sensitivity — top ' + str(top_k) + ')'}")
    print("-" * 55)
    print(f"{'Rank':<5}  {'Feature':<35}  {'Score':>8}")
    print("-" * 55)
    for rank, idx in enumerate(ranked[:top_k], 1):
        print(f"{rank:<5}  {FEATURE_COLUMNS[idx]:<35}  {importance[idx]:>8.5f}")

    # Also show bottom 10 (least useful)
    print(f"\n{'Least informative features (bottom 10)'}")
    print("-" * 55)
    for rank, idx in enumerate(ranked[-10:], 1):
        print(f"{rank:<5}  {FEATURE_COLUMNS[idx]:<35}  {importance[idx]:>8.5f}")


# ---------------------------------------------------------------------------
# 2. Sector-wise performance
# ---------------------------------------------------------------------------

def compute_sector_performance(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    id_to_sector: Dict[int, str],
    temperature_scaler=None,
) -> Dict[str, dict]:
    """
    Run inference on the test set and bucket results by sector_id.

    Returns:
        {sector_name: {'correct': int, 'total': int, 'per_class': ...}}
    """
    model.eval()
    # sector_id → [correct_count, total_count]
    sector_stats: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    # sector_id → per_class [correct, total]
    sector_class: Dict[int, Dict[int, List[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    with torch.no_grad():
        for batch in test_loader:
            sequences, labels, sectors_t = batch[0], batch[1], batch[2]
            industries = batch[3] if len(batch) > 3 else None

            sequences = sequences.to(device)
            labels = labels.to(device)
            sectors_t = sectors_t.to(device)
            if industries is not None:
                industries = industries.to(device)

            out = model(sequences, sectors_t, industries)
            logits = out[0] if isinstance(out, tuple) else out
            lg = logits.float()
            if temperature_scaler is not None:
                lg = temperature_scaler(lg)

            # Average prediction across horizons (majority horizon vote)
            if lg.dim() == 3:                          # (B, H, C)
                preds = lg.argmax(dim=2)               # (B, H)
                # Use middle horizon (day-4) as primary
                H = preds.size(1)
                mid_h = H // 2
                pred_h = preds[:, mid_h]               # (B,)
                true_h = labels[:, mid_h]              # (B,)
            else:
                pred_h = lg.argmax(dim=1)
                true_h = labels

            pred_h = pred_h.cpu().numpy()
            true_h = true_h.cpu().numpy()
            secs = sectors_t.cpu().numpy()

            for i, sec_id in enumerate(secs):
                sec_id = int(sec_id)
                correct = int(pred_h[i] == true_h[i])
                sector_stats[sec_id][0] += correct
                sector_stats[sec_id][1] += 1
                sector_class[sec_id][int(true_h[i])][0] += correct
                sector_class[sec_id][int(true_h[i])][1] += 1

    results = {}
    for sec_id, (correct, total) in sorted(sector_stats.items(),
                                            key=lambda x: -x[1][1]):
        name = id_to_sector.get(sec_id, f'sector_{sec_id}')
        acc = correct / total if total > 0 else 0.0
        results[name] = {
            'accuracy': acc,
            'correct': correct,
            'total': total,
            'per_class': {
                c: (v[0] / v[1] if v[1] > 0 else 0.0, v[1])
                for c, v in sector_class[sec_id].items()
            },
        }
    return results


def print_sector_performance(results: Dict[str, dict], class_names: List[str]):
    print(f"\n{'Sector-wise Performance (day-4 horizon)'}")
    print("-" * 65)
    print(f"{'Sector':<30}  {'Accuracy':>8}  {'Samples':>8}  {'vs random':>9}")
    print("-" * 65)
    random_base = 1.0 / NUM_CLASSES
    for name, stats in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        delta = stats['accuracy'] - random_base
        sign = '+' if delta >= 0 else ''
        print(f"{name:<30}  {stats['accuracy']:>7.2%}  {stats['total']:>8,}  "
              f"{sign}{delta:>8.2%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Post-training analysis')
    parser.add_argument('--data_dir', default='stock_data')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--grad_batches', type=int, default=50,
                        help='Number of batches for gradient importance (fewer = faster)')
    parser.add_argument('--top_k', type=int, default=30)
    args = parser.parse_args()

    model_path = os.path.join(args.data_dir, 'transformer_classifier.pth')
    cache_dir  = os.path.join(args.data_dir, 'cache')

    if not os.path.exists(model_path):
        print(f"No checkpoint found at {model_path}")
        return

    print(f"Loading checkpoint: {model_path}")
    checkpoint, config = _load_checkpoint(model_path, args.device)

    # Load test dataset
    print("Opening test dataset (memmap)...")
    _, _, test_dataset, _ = load_memmap_datasets(cache_dir)

    # Reconstruct sector / industry mappings
    sector_data = load_sector_data(args.data_dir)
    num_sectors = len(sector_data['sector'].unique()) if not sector_data.empty else 0
    if not sector_data.empty and 'industry' in sector_data.columns:
        industry_to_id = {ind: i for i, ind in
                          enumerate(sector_data['industry'].dropna().unique())}
    else:
        industry_to_id = {}
    industry_to_id['Unknown'] = len(industry_to_id)
    num_industries = max(0, len(industry_to_id) - 1)

    input_dim = test_dataset.n_features
    model = _build_model(config, input_dim, num_sectors, num_industries, args.device)
    model, _ = load_model(model, model_path, args.device)
    model.eval()

    # Temperature scaler
    temperature_scaler = None
    if 'temperatures' in checkpoint:
        temperature_scaler = TemperatureScaler(
            checkpoint['temperatures'].numel()
        ).to(args.device)
        temperature_scaler.temperatures = nn.Parameter(
            checkpoint['temperatures'].to(args.device)
        )
        print(f"Temperature scaler loaded: T={checkpoint['temperatures'].tolist()}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device.startswith('cuda')),
    )

    # -----------------------------------------------------------------------
    # 1. Feature importance
    # -----------------------------------------------------------------------
    print(f"\nComputing gradient-based feature importance "
          f"({args.grad_batches} batches × {args.batch_size} samples)...")
    importance = compute_gradient_importance(
        model, test_loader, args.device, n_batches=args.grad_batches
    )
    print_feature_importance(importance, top_k=args.top_k)

    # -----------------------------------------------------------------------
    # 2. Sector-wise performance
    # -----------------------------------------------------------------------
    print("\nComputing sector-wise performance...")
    id_to_sector = _build_sector_map(args.data_dir)
    sector_results = compute_sector_performance(
        model, test_loader, args.device, id_to_sector, temperature_scaler
    )
    print_sector_performance(sector_results, get_class_names())


if __name__ == '__main__':
    main()
