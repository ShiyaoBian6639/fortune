"""
Predict 5-day excess returns for all stocks in the deeptime cache.

Usage:
    python -m deeptime.predict_all                          # predict test split
    python -m deeptime.predict_all --split all              # predict all splits
    python -m deeptime.predict_all --split test --top 20    # print top/bottom 20 stocks

Outputs:
    plots/deeptime_results/predictions_<split>.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deeptime.config import get_config, get_horizon_name, NUM_HORIZONS, FORWARD_WINDOWS
from deeptime.model import create_deeptime_model
from deeptime.memmap_dataset import RegressionMemmapDataset, get_cache_info
from torch.utils.data import DataLoader


def load_model(config: dict) -> torch.nn.Module:
    model = create_deeptime_model(config)
    ckpt_path = config.get('model_save_path', 'stock_data/deeptime_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=config['device'], weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(config['device']).eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, val IC={ckpt.get('val_ic', 0):.4f})")
    return model


@torch.no_grad()
def predict_split(model, dataset, config, batch_size=512):
    """Run inference on a dataset split, return (preds, targets, dates, sectors)."""
    device = config['device']
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_targets, all_dates, all_sectors = [], [], [], []

    for batch in loader:
        def _t(x): return x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
        obs    = _t(batch[0]); future = _t(batch[1]); tgt = batch[2]
        sec    = _t(batch[3]); ind    = _t(batch[4])
        sub    = _t(batch[5]); sz     = _t(batch[6])
        area   = _t(batch[7])  if len(batch) > 7  else torch.zeros_like(sec)
        board  = _t(batch[8])  if len(batch) > 8  else torch.zeros_like(sec)
        ipo    = _t(batch[9])  if len(batch) > 9  else torch.zeros_like(sec)
        # anchor_date is index 10 in the 11-tuple
        dates  = batch[10] if len(batch) > 10 else None

        with torch.autocast(device.split(':')[0], torch.float16, enabled=(device != 'cpu')):
            preds = model(obs, future, sec, ind, sub, sz, area, board, ipo)

        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(tgt.numpy() if isinstance(tgt, torch.Tensor) else np.array(tgt))
        if dates is not None:
            all_dates.append(dates.numpy() if isinstance(dates, torch.Tensor) else np.array(dates))
        all_sectors.append(sec.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    dates   = np.concatenate(all_dates,   axis=0) if all_dates else np.zeros(len(preds), dtype=np.int32)
    sectors = np.concatenate(all_sectors, axis=0)
    return preds, targets, dates, sectors


def build_output_df(preds, targets, dates, sectors):
    rows = {'anchor_date': dates, 'sector_id': sectors}
    for h in range(NUM_HORIZONS):
        hn = get_horizon_name(h)
        rows[f'pred_{hn}']   = preds[:, h]
        rows[f'actual_{hn}'] = targets[:, h]
    df = pd.DataFrame(rows)
    df = df.sort_values('anchor_date').reset_index(drop=True)
    return df


def compute_metrics(preds, targets):
    from scipy.stats import spearmanr
    metrics = {}
    ics = []
    for h in range(NUM_HORIZONS):
        p, t = preds[:, h], targets[:, h]
        valid = np.isfinite(p) & np.isfinite(t)
        ic = float(spearmanr(p[valid], t[valid])[0]) if valid.sum() > 5 else 0.0
        mae = float(np.mean(np.abs(p[valid] - t[valid])))
        hr  = float(np.mean(np.sign(p[valid]) == np.sign(t[valid])))
        hn  = get_horizon_name(h)
        metrics[f'ic_{hn}']  = ic
        metrics[f'mae_{hn}'] = mae
        metrics[f'hr_{hn}']  = hr
        ics.append(ic)
    metrics['ic_mean'] = float(np.mean(ics))
    return metrics


def main():
    p = argparse.ArgumentParser(description='deeptime predict_all')
    p.add_argument('--split',    default='test', choices=['train', 'val', 'test', 'all'])
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--top',      type=int, default=0, help='Print top/bottom N stocks by mean pred IC')
    p.add_argument('--batch_size', type=int, default=512)
    args = p.parse_args()

    config = get_config()
    if args.cache_dir:
        config['cache_dir'] = args.cache_dir

    model = load_model(config)
    meta  = get_cache_info(config['cache_dir'])

    out_dir = os.path.join(_ROOT, 'plots', 'deeptime_results')
    os.makedirs(out_dir, exist_ok=True)

    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    for split in splits:
        n = meta['splits'].get(split, {}).get('n_samples', 0)
        if n == 0:
            print(f"  {split}: empty, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Predicting {split} split ({n:,} samples)...")
        dataset = RegressionMemmapDataset(config['cache_dir'], split)
        preds, targets, dates, sectors = predict_split(model, dataset, config, args.batch_size)

        metrics = compute_metrics(preds, targets)
        print(f"\n  Results on {split}:")
        for h in range(NUM_HORIZONS):
            hn = get_horizon_name(h)
            print(f"    {hn}: IC={metrics[f'ic_{hn}']:.4f}  MAE={metrics[f'mae_{hn}']:.4f}  HR={metrics[f'hr_{hn}']:.4f}")
        print(f"    Mean IC = {metrics['ic_mean']:.4f}")

        df = build_output_df(preds, targets, dates, sectors)
        csv_path = os.path.join(out_dir, f'predictions_{split}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}  ({len(df):,} rows)")

        # Top/bottom stocks by sector-average IC
        if args.top > 0 and 'anchor_date' in df.columns:
            print(f"\n  Top {args.top} dates by mean pred IC:")
            date_ic = df.groupby('anchor_date').apply(
                lambda g: float(np.corrcoef(g['pred_day1'], g['actual_day1'])[0, 1])
                if len(g) > 5 else 0.0
            ).sort_values(ascending=False)
            print(date_ic.head(args.top).to_string())


if __name__ == '__main__':
    main()
