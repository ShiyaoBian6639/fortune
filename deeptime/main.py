"""
deeptime — Entry point for the regression forecasting pipeline.

Usage:
    # Smoke test (100 stocks, build cache)
    python -m deeptime.main --max_stocks 100 --epochs 15

    # Full dataset — first run (builds cache)
    python -m deeptime.main --max_stocks 0 --epochs 50

    # Subsequent runs (use existing cache)
    python -m deeptime.main --use_cache --epochs 100

    # Predict only
    python -m deeptime.main --use_cache --predict_only
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Ensure repo root is in path so `dl.*` imports work
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dl.training import set_seed

from .config import get_config, DEFAULT_CONFIG, NUM_HORIZONS, get_horizon_name
from .memmap_dataset import cache_exists, get_cache_info, load_regression_datasets
from .model import create_deeptime_model
from .training import train_model, evaluate, compute_regression_metrics
from .plotting import plot_all
from .sanity_checks import run_all_checks


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='deeptime regression pipeline')
    p.add_argument('--preset', choices=['rtx5090', 'rtx5090_aggressive'],
                   help='Hardware preset: rtx5090 (batch=512, hidden=256, heads=8), '
                        'rtx5090_aggressive (batch=768, hidden=384)')
    p.add_argument('--data_dir',    default=DEFAULT_CONFIG['data_dir'])
    p.add_argument('--cache_dir',   default=DEFAULT_CONFIG['cache_dir'])
    p.add_argument('--max_stocks',  type=int, default=100,
                   help='0 = use all stocks')
    p.add_argument('--epochs',      type=int, default=50)
    # None = use config default (128); explicit value overrides
    p.add_argument('--batch_size',  type=int, default=None,
                   help='Batch size (default from config: 128)')
    p.add_argument('--lr',             type=float, default=None,
                   help='Learning rate (default: 2e-5)')
    p.add_argument('--no_lr_scale', action='store_true',
                   help='Disable auto LR scaling for batch size (use exact --lr value)')
    p.add_argument('--weight_decay',   type=float, default=None,
                   help='AdamW weight decay (default: 0.05)')
    p.add_argument('--max_grad_norm',  type=float, default=None,
                   help='Gradient clip threshold (default: 0.5)')
    p.add_argument('--dropout',        type=float, default=None,
                   help='Dropout rate (default: 0.15)')
    p.add_argument('--hidden',      type=int, default=None,
                   help='TFT hidden dim (default from config: 128)')
    p.add_argument('--heads',       type=int, default=None,
                   help='Attention heads (default from config: 4)')
    p.add_argument('--lstm_layers', type=int, default=None,
                   help='LSTM layers (default from config: 2)')
    p.add_argument('--warmup_epochs', type=int, default=None,
                   help='LR warmup epochs (default: 2; use 5-8 for large batches)')
    p.add_argument('--patience', type=int, default=None,
                   help='Early stopping patience (default: 15)')
    p.add_argument('--target_mode', default='excess', choices=['excess', 'raw'])
    p.add_argument('--loss_type',   default='huber', choices=['huber', 'huber+ic'])
    p.add_argument('--seq_len',     type=int, default=None,
                   help='Sequence length (default from config: 30)')
    p.add_argument('--use_cache',   action='store_true',
                   help='Skip data processing; load from existing cache')
    p.add_argument('--predict_only', action='store_true',
                   help='Skip training; only run evaluation and prediction')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--no_amp',      action='store_true')
    p.add_argument('--sanity_only', action='store_true',
                   help='Run sanity checks only and exit')
    # Data loading optimization (for high-end GPUs)
    p.add_argument('--chunk_samples', type=int, default=None,
                   help='Samples per I/O chunk (default: auto-scaled based on batch)')
    p.add_argument('--prefetch', type=int, default=None,
                   help='Prefetch depth (2=double-buffer, 3=triple-buffer)')
    p.add_argument('--num_workers', type=int, default=None,
                   help='DataLoader workers (0=chunked loader, 4-8=parallel workers)')
    p.add_argument('--preload', action='store_true',
                   help='Preload entire train set to RAM for max GPU throughput')
    p.add_argument('--max_chunk_gb', type=float, default=None,
                   help='Max GB per chunk (overrides auto-detection). Use if OOM.')
    return p.parse_args()


# ─── Stratified stock sampler ─────────────────────────────────────────────────

def _stratified_sample(
    stock_files: list,
    data_dir: str,
    max_stocks: int,
    seed: int = 42,
) -> list:
    """
    Stratified random sample from stock_files, grouped by SW L1 sector.

    Guarantees that every sector present in the data is represented
    proportionally in the subset.  Falls back to uniform random shuffle
    if sector data (stock_sectors.csv) is not available.

    Algorithm:
      1. Load stock_sectors.csv → build {ts_code: sw_l1_name} map.
      2. Group stock_files by sector.
      3. Sample proportionally: each sector gets ~max_stocks / n_sectors slots,
         with remainders distributed to larger sectors.
      4. Shuffle within each sector before sampling for randomness.

    Args:
        stock_files: [(ts_code, filepath), ...]  — full universe
        data_dir:    path to stock_data/
        max_stocks:  target number of stocks
        seed:        random seed for reproducibility
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)

    if max_stocks >= len(stock_files):
        return stock_files   # no sampling needed

    # Try to load sector info for stratification
    sector_col = None
    sector_map: dict = {}
    sector_path = os.path.join(data_dir, 'stock_sectors.csv')

    if os.path.exists(sector_path):
        try:
            import pandas as pd
            df = pd.read_csv(sector_path, usecols=['ts_code', 'sw_l1_name', 'sector'])
            # Prefer SW L1 (31 sectors); fall back to coarse 'sector' (9 groups)
            if 'sw_l1_name' in df.columns and df['sw_l1_name'].nunique() > 9:
                sector_col = 'sw_l1_name'
            else:
                sector_col = 'sector'
            sector_map = df.set_index('ts_code')[sector_col].to_dict()
            # Also index by bare code
            for code, val in list(sector_map.items()):
                bare = str(code).split('.')[0]
                sector_map[bare] = val
        except Exception as e:
            print(f"  [warn] Could not load sector data for stratification: {e}")

    if not sector_map:
        # Fallback: uniform random shuffle
        sampled = list(stock_files)
        rng.shuffle(sampled)
        return sampled[:max_stocks]

    # Group by sector
    groups: dict = defaultdict(list)
    for ts_code, filepath in stock_files:
        bare   = str(ts_code).split('.')[0]
        sector = sector_map.get(ts_code, sector_map.get(bare, 'Unknown'))
        groups[sector].append((ts_code, filepath))

    n_sectors = len(groups)
    base_per_sector = max_stocks // n_sectors
    remainder       = max_stocks % n_sectors

    # Sort sectors by size descending so larger sectors absorb the remainder first
    sorted_sectors = sorted(groups.keys(), key=lambda s: -len(groups[s]))

    sampled = []
    for i, sector in enumerate(sorted_sectors):
        stocks = list(groups[sector])
        rng.shuffle(stocks)
        quota = base_per_sector + (1 if i < remainder else 0)
        quota = min(quota, len(stocks))   # can't take more than available
        sampled.extend(stocks[:quota])

    # If total < max_stocks (some sectors had fewer stocks than quota),
    # top up with random picks from sectors that had more available
    if len(sampled) < max_stocks:
        already = {ts for ts, _ in sampled}
        pool    = [(ts, fp) for ts, fp in stock_files if ts not in already]
        rng.shuffle(pool)
        sampled.extend(pool[:max_stocks - len(sampled)])

    rng.shuffle(sampled)   # final shuffle so sectors aren't in blocks

    # Print sector distribution
    sector_counts = defaultdict(int)
    for ts_code, _ in sampled:
        bare   = str(ts_code).split('.')[0]
        sector = sector_map.get(ts_code, sector_map.get(bare, 'Unknown'))
        sector_counts[sector] += 1

    print(f"  Stratified by {sector_col or 'random'} across "
          f"{n_sectors} sectors → {len(sampled)} stocks selected")
    top5 = sorted(sector_counts.items(), key=lambda x: -x[1])[:5]
    others = len(sector_counts) - 5
    print("  " + "  |  ".join(f"{s}: {n}" for s, n in top5)
          + (f"  |  +{others} more" if others > 0 else ""))

    return sampled


# ─── Data pipeline ────────────────────────────────────────────────────────────

def build_cache(args, config):
    """Run the full data processing pipeline and write cache."""
    from dl.data_processing import (
        load_sector_data, load_daily_basic_data,
        load_market_context_data, load_index_membership_data,
        load_stk_limit_data, load_moneyflow_data,
        compute_cross_section_tech_stats,
    )
    from .data_processing import (
        load_fina_indicator_data, load_block_trade_data, _pregroup_block_trade,
        prepare_dataset_regression,
    )
    from .extended_features import (
        load_forecast_data, load_express_data,
        load_limit_data_by_stock, load_dragon_tiger_by_stock,
        load_chip_perf_by_stock,
    )

    data_dir = config['data_dir']
    max_stocks = config['max_stocks']

    print("\n" + "="*60)
    print("deeptime — Building Cache")
    print("="*60)

    # ── Stock file list ───────────────────────────────────────────────────
    sh_dir = os.path.join(data_dir, 'sh')
    sz_dir = os.path.join(data_dir, 'sz')
    stock_files = []
    for d in [sh_dir, sz_dir]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.csv'):
                    ts_code = f.replace('.csv', '')
                    stock_files.append((ts_code, os.path.join(d, f)))

    if max_stocks and max_stocks > 0:
        stock_files = _stratified_sample(
            stock_files, data_dir,
            max_stocks, seed=config.get('random_seed', 42),
        )
        print(f"Using {len(stock_files)} stocks (stratified sample, seed={config.get('random_seed', 42)})")
    else:
        print(f"Using all {len(stock_files)} stocks")

    ts_codes = [ts for ts, _ in stock_files]
    # Bare codes set for filtering large DataFrames (RAM saving)
    bare_codes_set = {c.split('.')[0] for c in ts_codes}

    # ── Load auxiliary data ────────────────────────────────────────────────
    print("\nLoading auxiliary data...")

    sector_data = load_sector_data(data_dir)
    print(f"  Sectors: {len(sector_data)} stocks")

    daily_basic = load_daily_basic_data(data_dir)
    # Filter to only the stocks we'll process — saves ~90% RAM on subset runs
    if max_stocks and max_stocks > 0 and len(daily_basic) > 0 and 'ts_code' in daily_basic.columns:
        daily_basic = daily_basic[daily_basic['ts_code'].str.split('.').str[0].isin(bare_codes_set)]
        daily_basic = daily_basic.reset_index(drop=True)
    print(f"  Daily basic: {len(daily_basic):,} rows")

    market_context    = load_market_context_data(data_dir)
    index_membership  = load_index_membership_data(data_dir)

    # ── stk_limit and moneyflow: load subset-filtered to keep peak RAM <16 GB ──
    # These are date-organised files (2254 CSVs each). Loading all rows then
    # filtering is cheaper on disk I/O but spikes RAM. We filter immediately
    # after concat so the peak is (full load) for only one source at a time.
    import gc as _gc
    stk_limit = load_stk_limit_data(data_dir)
    if len(stk_limit) > 0 and 'ts_code' in stk_limit.columns and bare_codes_set:
        stk_limit = stk_limit[stk_limit['ts_code'].str.split('.').str[0].isin(bare_codes_set)].reset_index(drop=True)
    print(f"  stk_limit: {len(stk_limit):,} rows after filter")
    _gc.collect()

    moneyflow = load_moneyflow_data(data_dir)
    if len(moneyflow) > 0 and 'ts_code' in moneyflow.columns and bare_codes_set:
        moneyflow = moneyflow[moneyflow['ts_code'].str.split('.').str[0].isin(bare_codes_set)].reset_index(drop=True)
    print(f"  moneyflow: {len(moneyflow):,} rows after filter")
    _gc.collect()

    # ── Fina indicator ─────────────────────────────────────────────────────
    print("\nLoading fina_indicator data...")
    fina_data = load_fina_indicator_data(data_dir, ts_codes)

    # ── Block trade: filter to our stock set to cap RAM ────────────────────
    print("\nLoading block_trade data (filtered)...")
    block_trade_daily = load_block_trade_data(data_dir)
    # Pre-group immediately and keep only processed stocks
    block_trade_by_stock = _pregroup_block_trade(block_trade_daily)
    del block_trade_daily
    _gc.collect()
    if bare_codes_set:
        block_trade_by_stock = {k: v for k, v in block_trade_by_stock.items() if k in bare_codes_set}
    print(f"  block_trade: {len(block_trade_by_stock)} stocks retained")

    # ── Cross-section tech stats ───────────────────────────────────────────
    print("\nComputing cross-section technical stats (first pass)...")
    cs_tech_stats = compute_cross_section_tech_stats(
        stock_files, min_data_points=config.get('min_data_points', 100)
    )

    # ── Extended data sources ─────────────────────────────────────────────
    print("\nLoading extended data sources...")

    # Earnings forecast and express
    forecast_data = load_forecast_data(data_dir, ts_codes)
    if bare_codes_set:
        forecast_data = {k: v for k, v in forecast_data.items() if k in bare_codes_set}
    print(f"  forecast: {len(forecast_data)} stocks")

    express_data = load_express_data(data_dir, ts_codes)
    if bare_codes_set:
        express_data = {k: v for k, v in express_data.items() if k in bare_codes_set}
    print(f"  express: {len(express_data)} stocks")

    # Limit and dragon-tiger data
    limit_data_by_stock = load_limit_data_by_stock(data_dir)
    if bare_codes_set:
        limit_data_by_stock = {k: v for k, v in limit_data_by_stock.items() if k in bare_codes_set}
    print(f"  limit_data: {len(limit_data_by_stock)} stocks")

    dragon_tiger_data = load_dragon_tiger_by_stock(data_dir)
    if bare_codes_set:
        dragon_tiger_data = {k: v for k, v in dragon_tiger_data.items() if k in bare_codes_set}
    print(f"  dragon_tiger: {len(dragon_tiger_data)} stocks")

    # Chip distribution data
    chip_perf_data = load_chip_perf_by_stock(data_dir)
    if bare_codes_set:
        chip_perf_data = {k: v for k, v in chip_perf_data.items() if k in bare_codes_set}
    print(f"  chip_perf: {len(chip_perf_data)} stocks")

    _gc.collect()

    # ── Run sanity checks (pre-cache) ─────────────────────────────────────
    run_all_checks(data_dir, cache_dir=None, config=config)

    # ── Build cache ───────────────────────────────────────────────────────
    t0 = time.time()
    metadata = prepare_dataset_regression(
        stock_files          = stock_files,
        sector_data          = sector_data,
        daily_basic          = daily_basic,
        output_dir           = config['cache_dir'],
        data_dir             = data_dir,
        config               = config,
        fina_data            = fina_data,
        block_trade_by_stock = block_trade_by_stock,
        market_context       = market_context,
        index_membership     = index_membership,
        stk_limit            = stk_limit,
        moneyflow            = moneyflow,
        cs_tech_stats        = cs_tech_stats,
        split_mode           = config.get('split_mode', 'rolling_window'),
        # Extended data sources
        forecast_data        = forecast_data,
        express_data         = express_data,
        limit_data_by_stock  = limit_data_by_stock,
        dragon_tiger_data    = dragon_tiger_data,
        chip_perf_data       = chip_perf_data,
    )
    print(f"\nCache built in {(time.time()-t0)/60:.1f} min → {config['cache_dir']}")
    return metadata


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Only pass CLI args that were explicitly provided (not None defaults)
    # so config.py defaults are respected when flags are omitted.
    overrides = {
        'max_stocks':   args.max_stocks if args.max_stocks > 0 else 0,
        'epochs':       args.epochs,
        'target_mode':  args.target_mode,
        'loss_type':    args.loss_type,
        'use_amp':      not args.no_amp,
        'random_seed':  args.seed,
        'data_dir':     args.data_dir,
        'cache_dir':    args.cache_dir,
        'model_save_path': os.path.join(args.data_dir, 'deeptime_model.pth'),
    }
    if args.batch_size   is not None: overrides['batch_size']       = args.batch_size
    if args.lr           is not None: overrides['learning_rate']    = args.lr
    if args.weight_decay is not None: overrides['weight_decay']     = args.weight_decay
    if args.max_grad_norm is not None: overrides['max_grad_norm']   = args.max_grad_norm
    if args.dropout      is not None: overrides['tft_dropout']      = args.dropout
    if args.hidden       is not None: overrides['tft_hidden']       = args.hidden
    if args.heads        is not None: overrides['tft_heads']        = args.heads
    if args.lstm_layers  is not None: overrides['tft_lstm_layers']  = args.lstm_layers
    if args.seq_len      is not None: overrides['sequence_length']  = args.seq_len
    if args.warmup_epochs is not None: overrides['warmup_epochs']   = args.warmup_epochs
    if args.patience     is not None: overrides['early_stopping_patience'] = args.patience
    if args.chunk_samples is not None: overrides['chunk_samples']  = args.chunk_samples
    if args.prefetch     is not None: overrides['prefetch_factor'] = args.prefetch
    if args.num_workers  is not None: overrides['num_workers']     = args.num_workers
    if args.preload:                  overrides['preload']         = True
    if args.max_chunk_gb is not None: overrides['max_chunk_gb']    = args.max_chunk_gb
    if args.no_lr_scale:              overrides['no_lr_scale']     = True

    config = get_config(preset=args.preset, **overrides)

    # Log effective config when using preset
    if args.preset:
        print(f"\n{'='*60}")
        print(f"Using {args.preset.upper()} preset")
        print(f"  hidden={config['tft_hidden']}, heads={config['tft_heads']}, "
              f"dropout={config['tft_dropout']}")
        print(f"  batch_size={config['batch_size']}, seq_len={config['sequence_length']}")
        print(f"  lr={config['learning_rate']:.1e}, warmup={config['warmup_epochs']}ep, "
              f"weight_decay={config['weight_decay']}")
        nw = config.get('num_workers', 0)
        print(f"  chunk_samples={config['chunk_samples']:,}, prefetch={config.get('prefetch_factor', 2)}, "
              f"num_workers={nw}")
        print(f"{'='*60}")

    if args.sanity_only:
        run_all_checks(config['data_dir'], config['cache_dir'], config)
        return

    # ── Build or load cache ────────────────────────────────────────────────
    if args.use_cache and cache_exists(config['cache_dir']):
        print(f"\nLoading existing cache from {config['cache_dir']}")
        meta = get_cache_info(config['cache_dir'])
    else:
        if args.use_cache:
            print(f"\nCache not found at {config['cache_dir']} — building from scratch")
        meta = build_cache(args, config)

    # Run post-cache sanity checks
    run_all_checks(config['data_dir'], config['cache_dir'], config)

    # ── GPU memory limit: dedicated VRAM only (RTX 4070 Super = 12 GB) ───────
    if torch.cuda.is_available():
        # Reserve at most 90% of dedicated VRAM; raises OOM instead of spilling
        # into shared memory (which is slow system RAM masquerading as VRAM).
        torch.cuda.set_per_process_memory_fraction(0.90)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        limit_gb = vram_gb * 0.90
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"  Dedicated VRAM: {vram_gb:.1f} GB  |  Hard limit: {limit_gb:.1f} GB (90%)")

    # Print effective training config so there's no ambiguity
    print(f"\nEffective config:")
    print(f"  batch_size={config['batch_size']}  hidden={config['tft_hidden']}  "
          f"heads={config['tft_heads']}  lstm_layers={config['tft_lstm_layers']}")
    print(f"  lr={config['learning_rate']}  weight_decay={config.get('weight_decay',0.05)}  "
          f"max_grad_norm={config.get('max_grad_norm',0.5)}  dropout={config.get('tft_dropout',0.15)}")
    print(f"  seq_len={config['sequence_length']}  lr={config['learning_rate']}  "
          f"epochs={config['epochs']}  AMP={config['use_amp']}")

    # ── Create data loaders ────────────────────────────────────────────────
    print("\nLoading datasets...")
    loaders, meta = load_regression_datasets(
        cache_dir       = config['cache_dir'],
        batch_size      = config['batch_size'],
        device          = config['device'],
        chunk_samples   = config['chunk_samples'],
        prefetch_factor = config.get('prefetch_factor', 2),
        num_workers     = config.get('num_workers', 0),
        preload         = config.get('preload', False),
        max_chunk_gb    = config.get('max_chunk_gb', None),
        use_chunked     = True,
    )

    print(f"  train: {meta['splits']['train']['n_samples']:,} samples")
    print(f"  val:   {meta['splits']['val']['n_samples']:,} samples")
    print(f"  test:  {meta['splits']['test']['n_samples']:,} samples")

    # ── Create model ───────────────────────────────────────────────────────
    model = create_deeptime_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params/1e6:.2f}M parameters")

    if args.predict_only:
        ckpt_path = config['model_save_path']
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=config['device'], weights_only=True)
            model.load_state_dict(ckpt['model_state'])
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print("No checkpoint found — running untrained model")
        # Load saved training history for plotting (saved alongside checkpoint)
        history_path = ckpt_path.replace('.pth', '_history.json')
        if os.path.exists(history_path):
            import json as _json
            history = _json.load(open(history_path))
            print(f"Loaded training history ({len(history.get('train_loss',[]))} epochs)")
        else:
            history = {'train_loss': [], 'val_loss': [], 'train_ic': [], 'val_ic': [], 'lr': [], 'grad_norm': []}
    else:
        # ── Training ──────────────────────────────────────────────────────
        history = train_model(model, loaders['train'], loaders['val'], config)

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    from .losses import create_regression_loss
    criterion = create_regression_loss(config).to(config['device'])
    model = model.to(config['device'])

    test_metrics, test_preds, test_targets = evaluate(
        model, loaders['test'], criterion, config['device']
    )

    print("\nTest Results:")
    for h in range(NUM_HORIZONS):
        hn = get_horizon_name(h)
        print(f"  {hn}: IC={test_metrics.get('ic_'+hn, 0):.4f}  "
              f"MAE={test_metrics.get('mae_'+hn, 0):.4f}  "
              f"RMSE={test_metrics.get('rmse_'+hn, 0):.4f}  "
              f"HR={test_metrics.get('hr_'+hn, 0):.4f}")
    print(f"  Mean IC = {test_metrics.get('ic_mean', 0):.4f}")

    # ── Load test metadata for plotting ───────────────────────────────────
    test_dates   = None
    test_sectors = None
    try:
        import numpy as np
        cache_dir = config['cache_dir']
        n_test = meta['splits']['test']['n_samples']
        test_dates   = np.memmap(os.path.join(cache_dir, 'test_dates.npy'),   dtype='int32', mode='r', shape=(n_test,))[:]
        test_sectors = np.memmap(os.path.join(cache_dir, 'test_sectors.npy'), dtype='int64', mode='r', shape=(n_test,))[:]
    except Exception:
        pass

    # Run the model on a final test batch to populate interpretability buffers
    model.eval()
    try:
        with torch.no_grad():
            for batch in loaders['test']:
                def _t(x): return x.to(config['device']) if isinstance(x, torch.Tensor) else torch.tensor(x, device=config['device'])
                obs    = _t(batch[0]); future = _t(batch[1])
                sec    = _t(batch[3]); ind    = _t(batch[4])
                sub    = _t(batch[5]); sz     = _t(batch[6])
                area   = _t(batch[7])  if len(batch) > 7  else torch.zeros_like(sec)
                board  = _t(batch[8])  if len(batch) > 8  else torch.zeros_like(sec)
                ipo    = _t(batch[9])  if len(batch) > 9  else torch.zeros_like(sec)
                _ = model(obs, future, sec, ind, sub, sz, area, board, ipo)
                break
    except Exception:
        pass

    # ── Visualizations ────────────────────────────────────────────────────
    plot_all(
        history       = history,
        test_preds    = test_preds,
        test_targets  = test_targets,
        test_dates    = test_dates if test_dates is not None else np.zeros(len(test_preds), dtype=np.int32),
        test_sectors  = test_sectors if test_sectors is not None else np.zeros(len(test_preds), dtype=np.int64),
        test_metrics  = test_metrics,
        model         = model,
        sector_names  = None,
    )

    # ── Save predictions CSV ──────────────────────────────────────────────
    import pandas as pd
    pred_df = pd.DataFrame(
        test_preds,
        columns=[f'pred_{get_horizon_name(h)}' for h in range(NUM_HORIZONS)]
    )
    tgt_df = pd.DataFrame(
        test_targets,
        columns=[f'actual_{get_horizon_name(h)}' for h in range(NUM_HORIZONS)]
    )
    out_df = pd.concat([pred_df, tgt_df], axis=1)
    if test_dates is not None:
        out_df.insert(0, 'anchor_date', test_dates)
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'plots', 'deeptime_results', 'test_predictions.csv'
    )
    out_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
