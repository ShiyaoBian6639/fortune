"""
Diagnostic plots for the xgbmodel pipeline.

Consumes the CSVs written by train.py (val.csv / test.csv) and the metadata
JSON containing feature importances. Saves PNGs to cfg['plot_dir'].
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

from .config import MODEL_DIR, PLOT_DIR


def _load_meta(cfg: dict) -> Optional[dict]:
    path = os.path.join(cfg.get('model_dir', MODEL_DIR), 'xgb_pct_chg.meta.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_split(cfg: dict, split: str) -> Optional[pd.DataFrame]:
    path = os.path.join(cfg.get('model_dir', MODEL_DIR), 'xgb_preds', f'{split}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['trade_date'])
    return df


def plot_feature_importance(cfg: dict, top_n: int = 40) -> Optional[str]:
    meta = _load_meta(cfg)
    if not meta or not meta.get('feature_importance_top50'):
        return None
    fi = pd.DataFrame(meta['feature_importance_top50']).head(top_n)
    fi = fi.sort_values('total_gain', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 0.25 * len(fi) + 1.5))
    ax.barh(fi['feature'], fi['total_gain'], color='steelblue')
    ax.set_xlabel('Total gain')
    ax.set_title(f'XGBoost feature importance (top {len(fi)})')
    fig.tight_layout()
    out = os.path.join(cfg.get('plot_dir', PLOT_DIR), 'feature_importance.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_pred_vs_actual(cfg: dict, split: str = 'val') -> Optional[str]:
    df = _load_split(cfg, split)
    if df is None or df.empty:
        return None
    # Sub-sample to keep scatter readable
    if len(df) > 50_000:
        df = df.sample(50_000, random_state=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df['target'], df['pred'], s=3, alpha=0.15)
    lo = float(min(df['target'].min(), df['pred'].min()))
    hi = float(max(df['target'].max(), df['pred'].max()))
    ax.plot([lo, hi], [lo, hi], color='red', lw=1, label='y=x')
    corr = df[['pred', 'target']].corr().iloc[0, 1]
    ax.set_title(f'Pred vs actual — {split}  (Pearson={corr:+.3f}, n={len(df):,})')
    ax.set_xlabel('Actual next-day pct_chg (%)')
    ax.set_ylabel('Predicted pct_chg (%)')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(cfg.get('plot_dir', PLOT_DIR), f'pred_vs_actual_{split}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_daily_ic(cfg: dict, split: str = 'val') -> Optional[str]:
    df = _load_split(cfg, split)
    if df is None or df.empty:
        return None
    # Per-day rank IC
    rows = []
    for d, grp in df.groupby('trade_date'):
        if len(grp) < 10:
            continue
        r = grp[['pred', 'target']].rank(method='average')
        ic = r.corr().iloc[0, 1]
        rows.append((d, ic, len(grp)))
    if not rows:
        return None
    daily = pd.DataFrame(rows, columns=['date', 'ic', 'n']).sort_values('date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.bar(daily['date'], daily['ic'], color=['tab:blue' if v >= 0 else 'tab:red' for v in daily['ic']])
    ax1.axhline(0, color='k', lw=0.5)
    mean_ic = daily['ic'].mean()
    ax1.axhline(mean_ic, color='green', lw=1.2, ls='--', label=f'mean IC={mean_ic:+.3f}')
    ax1.set_ylabel('Daily rank IC')
    ax1.set_title(f'Daily cross-sectional rank IC — {split}  ({len(daily)} days)')
    ax1.legend()

    # Rolling 20-day IC
    daily['ic_ma20'] = daily['ic'].rolling(20, min_periods=5).mean()
    ax2.plot(daily['date'], daily['ic_ma20'], color='purple', lw=1.3)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_ylabel('IC MA(20)')
    fig.autofmt_xdate()
    fig.tight_layout()
    out = os.path.join(cfg.get('plot_dir', PLOT_DIR), f'daily_ic_{split}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_error_distribution(cfg: dict, split: str = 'val') -> Optional[str]:
    df = _load_split(cfg, split)
    if df is None or df.empty:
        return None
    err = (df['pred'] - df['target']).values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err, bins=120, color='slateblue', alpha=0.8)
    ax.axvline(0, color='red', lw=1)
    ax.set_xlabel('Prediction error (pred − target, %)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error distribution — {split}  '
                  f'(mean={err.mean():+.3f}, std={err.std():.3f}, n={len(err):,})')
    fig.tight_layout()
    out = os.path.join(cfg.get('plot_dir', PLOT_DIR), f'error_dist_{split}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_all(cfg: dict) -> None:
    print("[xgbmodel.plotting] rendering diagnostic plots ...")
    paths = []
    paths.append(plot_feature_importance(cfg))
    for split in ('val', 'test'):
        paths.append(plot_pred_vs_actual(cfg, split))
        paths.append(plot_daily_ic(cfg, split))
        paths.append(plot_error_distribution(cfg, split))
    for p in paths:
        if p:
            print(f"  saved {p}")
