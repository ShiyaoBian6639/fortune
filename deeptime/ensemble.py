"""
Seed ensemble for deeptime.

Trains N models with different random seeds, then averages their test
predictions. On non-stationary Chinese equities this typically adds
10-30% to test IC by averaging out per-seed variance in which sectors /
features each model latches onto.

Usage:
    python -m deeptime.ensemble --use_cache --n_seeds 3 --epochs 30 \\
        [any other deeptime.main flag — they're all forwarded]

Output:
    plots/deeptime_results/ensemble_predictions.csv
    (full prediction matrix: per-seed columns + ensemble mean + actuals)
    stock_data/deeptime_ensemble/seed_42.pth, seed_43.pth, ...
"""

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Pre-parse --drop_market (same pattern as main.py) so config picks it up
if '--drop_market' in sys.argv:
    os.environ['DEEPTIME_DROP_MARKET'] = '1'

from dl.training import set_seed
from .config import get_config, DEFAULT_CONFIG, NUM_HORIZONS, get_horizon_name
from .memmap_dataset import cache_exists, get_cache_info, load_regression_datasets
from .model import create_deeptime_model
from .training import train_model, evaluate, compute_regression_metrics
from .losses import create_regression_loss


# ─── Re-use main.py's arg parser plus ensemble-specific flags ─────────────────

def parse_args():
    p = argparse.ArgumentParser(description='deeptime seed ensemble',
                                parents=[_make_parent_parser()],
                                conflict_handler='resolve')
    p.add_argument('--n_seeds',      type=int, default=3,
                   help='Number of seeds to train (default 3)')
    p.add_argument('--base_seed',    type=int, default=42,
                   help='First seed; subsequent seeds are base_seed+1, +2, ...')
    p.add_argument('--ensemble_dir', default=None,
                   help='Directory for per-seed checkpoints '
                        '(default: stock_data/deeptime_ensemble/)')
    p.add_argument('--skip_existing', action='store_true',
                   help='Reuse checkpoints if already present in ensemble_dir')
    return p.parse_args()


def _make_parent_parser() -> argparse.ArgumentParser:
    """
    Mirror of the flags in main.py::parse_args so the ensemble script accepts
    the same CLI. Kept as a small inline duplication (no try/import dance) —
    if you add flags to main.py, add them here too.
    """
    p = argparse.ArgumentParser(add_help=False)
    from .config import DEFAULT_CONFIG as _C
    p.add_argument('--data_dir',    default=_C['data_dir'])
    p.add_argument('--cache_dir',   default=_C['cache_dir'])
    p.add_argument('--max_stocks',  type=int, default=100)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--batch_size',  type=int, default=None)
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--no_lr_scale', action='store_true')
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--max_grad_norm', type=float, default=None)
    p.add_argument('--dropout',     type=float, default=None)
    p.add_argument('--hidden',      type=int, default=None)
    p.add_argument('--heads',       type=int, default=None)
    p.add_argument('--lstm_layers', type=int, default=None)
    p.add_argument('--warmup_epochs', type=int, default=None)
    p.add_argument('--lr_schedule', choices=['cosine', 'flat'], default=None)
    p.add_argument('--no_swa',      action='store_true')
    p.add_argument('--swa_start',   type=int, default=None)
    p.add_argument('--swa_eval_every', type=int, default=None)
    p.add_argument('--horizon_weights', type=str, default=None)
    p.add_argument('--drop_market', action='store_true')
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--target_mode', default='excess', choices=['excess', 'raw'])
    p.add_argument('--loss_type',   default='huber', choices=['huber', 'huber+ic'])
    p.add_argument('--seq_len',     type=int, default=None)
    p.add_argument('--use_cache',   action='store_true')
    p.add_argument('--no_amp',      action='store_true')
    p.add_argument('--chunk_samples', type=int, default=None)
    p.add_argument('--prefetch',    type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--preload',     action='store_true')
    p.add_argument('--max_chunk_gb', type=float, default=None)
    return p


def _build_config(args, seed: int) -> dict:
    """Mirror of main.py's config-building, with seed override."""
    overrides = {
        'max_stocks':      args.max_stocks if args.max_stocks > 0 else 0,
        'epochs':          args.epochs,
        'target_mode':     args.target_mode,
        'loss_type':       args.loss_type,
        'use_amp':         not args.no_amp,
        'random_seed':     seed,
        'data_dir':        args.data_dir,
        'cache_dir':       args.cache_dir,
        'no_lr_scale':     args.no_lr_scale,
        # Per-seed checkpoint path
        'model_save_path': os.path.join(
            args.ensemble_dir or os.path.join(args.data_dir, 'deeptime_ensemble'),
            f'seed_{seed}.pth'
        ),
    }
    if args.batch_size    is not None: overrides['batch_size']       = args.batch_size
    if args.lr            is not None: overrides['learning_rate']    = args.lr
    if args.weight_decay  is not None: overrides['weight_decay']     = args.weight_decay
    if args.max_grad_norm is not None: overrides['max_grad_norm']    = args.max_grad_norm
    if args.dropout       is not None: overrides['tft_dropout']      = args.dropout
    if args.hidden        is not None: overrides['tft_hidden']       = args.hidden
    if args.heads         is not None: overrides['tft_heads']        = args.heads
    if args.lstm_layers   is not None: overrides['tft_lstm_layers']  = args.lstm_layers
    if args.seq_len       is not None: overrides['sequence_length']  = args.seq_len
    if args.warmup_epochs is not None: overrides['warmup_epochs']    = args.warmup_epochs
    if args.lr_schedule   is not None: overrides['lr_schedule']      = args.lr_schedule
    if args.no_swa:                     overrides['use_swa']         = False
    if args.swa_start     is not None: overrides['swa_start_epoch']  = args.swa_start
    if args.swa_eval_every is not None: overrides['swa_eval_every']  = args.swa_eval_every
    if args.horizon_weights is not None:
        overrides['horizon_weights'] = [float(x) for x in args.horizon_weights.split(',')]
    if args.patience      is not None: overrides['early_stopping_patience'] = args.patience
    if args.chunk_samples is not None: overrides['chunk_samples']  = args.chunk_samples
    if args.prefetch      is not None: overrides['prefetch_factor'] = args.prefetch
    if args.num_workers   is not None: overrides['num_workers']    = args.num_workers
    if args.preload:                    overrides['preload']        = True
    if args.max_chunk_gb  is not None: overrides['max_chunk_gb']   = args.max_chunk_gb
    return get_config(**overrides)


def _train_one_seed(args, seed: int, loaders, meta) -> str:
    """Train a single model at the given seed; return checkpoint path."""
    config = _build_config(args, seed)
    ckpt_path = config['model_save_path']

    if args.skip_existing and os.path.exists(ckpt_path):
        print(f"\n[seed {seed}] checkpoint exists at {ckpt_path} — skipping training")
        return ckpt_path

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training ensemble member — seed={seed}")
    print(f"{'='*60}")

    set_seed(seed)
    model = create_deeptime_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.2f}M parameters")

    train_model(model, loaders['train'], loaders['val'], config)
    # train_model saves final checkpoint to model_save_path; we already set
    # that per-seed via _build_config.
    return ckpt_path


@torch.no_grad()
def _predict_test(ckpt_path: str, config: dict, loaders) -> tuple:
    """Load ckpt and run full test-set prediction. Returns (preds, targets)."""
    device = config['device']
    model = create_deeptime_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    criterion = create_regression_loss(config).to(device)
    metrics, preds, targets = evaluate(model, loaders['test'], criterion, device)
    return preds, targets, metrics


def _print_metrics(prefix: str, metrics: dict):
    parts = [prefix]
    for h in range(NUM_HORIZONS):
        hn = get_horizon_name(h)
        parts.append(f"{hn}:{metrics.get('ic_'+hn, 0):.4f}")
    parts.append(f"mean:{metrics.get('ic_mean', 0):.4f}")
    print("  " + "  ".join(parts))


def main():
    args = parse_args()
    seeds: List[int] = [args.base_seed + i for i in range(args.n_seeds)]

    # Use the first seed's config to set up loaders; they're shared across seeds
    base_cfg = _build_config(args, seeds[0])
    ensemble_dir = args.ensemble_dir or os.path.join(args.data_dir, 'deeptime_ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)

    if not cache_exists(base_cfg['cache_dir']):
        raise SystemExit(
            f"No cache at {base_cfg['cache_dir']}. "
            f"Run `python -m deeptime.main --max_stocks 0` first to build it."
        )

    # Load loaders once (shared across seeds — deterministic order per seed)
    print(f"Loading cache from {base_cfg['cache_dir']}...")
    loaders, meta = load_regression_datasets(
        cache_dir       = base_cfg['cache_dir'],
        batch_size      = base_cfg['batch_size'],
        device          = base_cfg['device'],
        chunk_samples   = base_cfg['chunk_samples'],
        prefetch_factor = base_cfg.get('prefetch_factor', 2),
        num_workers     = base_cfg.get('num_workers', 0),
        preload         = base_cfg.get('preload', False),
        max_chunk_gb    = base_cfg.get('max_chunk_gb', None),
        use_chunked     = True,
    )

    # ── Train each seed ────────────────────────────────────────────────────
    ckpt_paths = []
    t_total = time.time()
    for seed in seeds:
        ckpt = _train_one_seed(args, seed, loaders, meta)
        ckpt_paths.append(ckpt)
    print(f"\nTotal training time: {(time.time()-t_total)/60:.1f} min")

    # ── Predict each seed on test set, collect for averaging ───────────────
    print(f"\n{'='*60}")
    print(f"Evaluating {len(seeds)} ensemble members")
    print(f"{'='*60}\n")

    all_preds   = []        # list of (N_test, H) arrays
    all_metrics = []
    targets_ref = None
    for seed, ckpt in zip(seeds, ckpt_paths):
        preds, targets, metrics = _predict_test(ckpt, _build_config(args, seed), loaders)
        all_preds.append(preds)
        all_metrics.append(metrics)
        if targets_ref is None:
            targets_ref = targets
        _print_metrics(f"seed={seed}", metrics)

    # ── Ensemble = mean of per-seed preds ─────────────────────────────────
    stacked   = np.stack(all_preds, axis=0)       # (n_seeds, N, H)
    ens_preds = stacked.mean(axis=0)

    ens_metrics = compute_regression_metrics(ens_preds, targets_ref)
    print()
    _print_metrics("ENSEMBLE", ens_metrics)

    # Delta vs best single model
    single_best_mean_ic = max(m.get('ic_mean', 0) for m in all_metrics)
    ens_mean_ic = ens_metrics.get('ic_mean', 0)
    gain_pct = 100.0 * (ens_mean_ic - single_best_mean_ic) / max(abs(single_best_mean_ic), 1e-8)
    print(f"\n  Best single-seed mean IC: {single_best_mean_ic:.4f}")
    print(f"  Ensemble mean IC:         {ens_mean_ic:.4f}  "
          f"({'+' if gain_pct >= 0 else ''}{gain_pct:.1f}% vs best single)")

    # ── Save prediction CSV ────────────────────────────────────────────────
    out_dir = os.path.join(_ROOT, 'plots', 'deeptime_results')
    os.makedirs(out_dir, exist_ok=True)

    cols = {}
    for i, seed in enumerate(seeds):
        for h in range(NUM_HORIZONS):
            hn = get_horizon_name(h)
            cols[f'seed{seed}_pred_{hn}'] = all_preds[i][:, h]
    for h in range(NUM_HORIZONS):
        hn = get_horizon_name(h)
        cols[f'ensemble_pred_{hn}'] = ens_preds[:, h]
        cols[f'actual_{hn}']        = targets_ref[:, h]

    # Add anchor dates if cache has them
    try:
        cache_dir = base_cfg['cache_dir']
        n_test = meta['splits']['test']['n_samples']
        dates = np.memmap(os.path.join(cache_dir, 'test_dates.npy'),
                          dtype='int32', mode='r', shape=(n_test,))[:]
        cols['anchor_date'] = dates
    except Exception:
        pass

    df = pd.DataFrame(cols)
    out_path = os.path.join(out_dir, 'ensemble_predictions.csv')
    df.to_csv(out_path, index=False)
    print(f"\nEnsemble predictions saved to {out_path}")

    # Save summary metrics JSON
    import json
    summary = {
        'seeds':             seeds,
        'per_seed_metrics':  [{k: float(v) for k, v in m.items()} for m in all_metrics],
        'ensemble_metrics':  {k: float(v) for k, v in ens_metrics.items()},
        'single_best_mean_ic': float(single_best_mean_ic),
    }
    with open(os.path.join(out_dir, 'ensemble_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {os.path.join(out_dir, 'ensemble_summary.json')}")


if __name__ == '__main__':
    main()
