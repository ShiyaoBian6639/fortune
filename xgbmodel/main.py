"""
CLI entry point for the xgbmodel pipeline.

Examples:
    # Quick smoke test on 100 stocks
    ./venv/Scripts/python -m xgbmodel.main --max_stocks 100

    # Full training run, 2017-now, CPU hist
    ./venv/Scripts/python -m xgbmodel.main

    # GPU training
    ./venv/Scripts/python -m xgbmodel.main --device cuda

    # Excess-return target
    ./venv/Scripts/python -m xgbmodel.main --target excess

    # Predict next-day pct_chg for every stock (uses latest date in panel)
    ./venv/Scripts/python -m xgbmodel.main --mode predict

    # Score the held-out test window
    ./venv/Scripts/python -m xgbmodel.main --mode predict_test

    # Re-render plots from saved preds
    ./venv/Scripts/python -m xgbmodel.main --mode plot
"""

from __future__ import annotations

import argparse
import sys

from .config import get_config
from .train import train as do_train, train_walk_forward
from .predict import predict_latest, predict_test
from .plotting import plot_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('xgbmodel')
    p.add_argument('--mode', choices=['train', 'predict', 'predict_test', 'plot', 'all'],
                   default='all',
                   help='train: fit new model; predict: score latest date; '
                        'predict_test: score test window; plot: re-render plots; '
                        'all: train + plot (default)')

    # Data / split
    p.add_argument('--max_stocks', type=int, default=None,
                   help='0 = all stocks (default from config)')
    p.add_argument('--target', choices=['raw', 'excess'], default=None,
                   help="'raw' = next-day pct_chg; 'excess' = minus CSI300")
    p.add_argument('--forward_window', type=int, default=None,
                   help='Horizon in trading days (default 1)')

    # Fixed (single-split) mode
    p.add_argument('--split_mode', choices=['fixed', 'walk_forward'], default=None,
                   help="'fixed' = single train/val/test cutoffs; "
                        "'walk_forward' = rolling or expanding CV (de Prado §7.4)")
    p.add_argument('--train_start', type=int, default=None)
    p.add_argument('--val_start',   type=int, default=None)
    p.add_argument('--test_start',  type=int, default=None)
    p.add_argument('--min_rows_per_stock', type=int, default=None)

    # Walk-forward CV knobs (apply when --split_mode walk_forward)
    p.add_argument('--fold_train_weeks', type=int, default=None,
                   help='Rolling train window length in weeks (default 12)')
    p.add_argument('--fold_val_weeks',   type=int, default=None,
                   help='Validation window in weeks (default 2)')
    p.add_argument('--fold_test_weeks',  type=int, default=None,
                   help='Out-of-sample test window in weeks (default 2)')
    p.add_argument('--fold_step_weeks',  type=int, default=None,
                   help='How far the cursor advances per fold (default 2 = no overlap)')
    p.add_argument('--purge_days',       type=int, default=None,
                   help='Days dropped at train→val boundary to prevent label leak (default 5)')
    p.add_argument('--embargo_days',     type=int, default=None,
                   help='Extra days dropped at val→test boundary (default 2)')
    p.add_argument('--expanding_train',  action='store_true',
                   help='Use an expanding (growing) train window instead of rolling')
    p.add_argument('--max_folds',        type=int, default=None,
                   help='Run only the first N folds (0 = all)')

    # XGBoost
    p.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--max_depth',     type=int,   default=None)
    p.add_argument('--n_estimators',  type=int,   default=None)
    p.add_argument('--early_stopping_rounds', type=int, default=None)
    p.add_argument('--subsample',        type=float, default=None)
    p.add_argument('--colsample_bytree', type=float, default=None)
    return p


def cfg_from_args(args: argparse.Namespace) -> dict:
    overrides = {}
    simple_keys = (
        'max_stocks', 'forward_window',
        'split_mode', 'train_start', 'val_start', 'test_start',
        'min_rows_per_stock', 'device',
        'fold_train_weeks', 'fold_val_weeks', 'fold_test_weeks', 'fold_step_weeks',
        'purge_days', 'embargo_days', 'max_folds',
    )
    for k in simple_keys:
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    if args.target:
        overrides['target_mode'] = args.target
    if getattr(args, 'expanding_train', False):
        overrides['expanding_train'] = True

    xgb_overrides = {}
    for k in ('learning_rate', 'max_depth', 'n_estimators', 'early_stopping_rounds',
              'subsample', 'colsample_bytree'):
        v = getattr(args, k, None)
        if v is not None:
            xgb_overrides[k] = v
    if xgb_overrides:
        overrides['xgb_params'] = xgb_overrides
    return get_config(**overrides)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg  = cfg_from_args(args)

    print(f"[xgbmodel] mode={args.mode}")
    print(f"[xgbmodel] target_mode={cfg['target_mode']}  forward_window={cfg['forward_window']}  "
          f"device={cfg['device']}")
    print(f"[xgbmodel] split: train<{cfg['val_start']} ≤ val < {cfg['test_start']} ≤ test")

    if args.mode in ('train', 'all'):
        if cfg.get('split_mode') == 'walk_forward':
            train_walk_forward(cfg)
        else:
            do_train(cfg)
        if args.mode == 'all':
            plot_all(cfg)
    elif args.mode == 'predict':
        predict_latest(cfg)
    elif args.mode == 'predict_test':
        predict_test(cfg)
        plot_all(cfg)
    elif args.mode == 'plot':
        plot_all(cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
