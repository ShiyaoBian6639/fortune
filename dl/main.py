#!/usr/bin/env python3
"""
Main entry point for stock price prediction training and evaluation.

Usage:
    # Test with 100 stocks (default)
    python -m dl.main

    # Full dataset
    python -m dl.main --max_stocks 0

    # Custom settings
    python -m dl.main --max_stocks 500 --epochs 100 --loss_type focal
"""

import os
import gc
import json
import time
import argparse
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from .config import get_config, get_class_names, NUM_CLASSES, NUM_HORIZONS, HORIZON_WEIGHTS, FEATURE_COLUMNS, SPLIT_MODE, NUM_RELATIVE_CLASSES
from .data_processing import (
    load_sector_data, load_stock_data, load_daily_basic_data,
    load_market_context_data, load_index_membership_data,
    load_stk_limit_data, load_moneyflow_data,
    prepare_dataset, normalize_data, split_data, prepare_dataset_to_disk,
    get_stock_files, compute_cross_section_tech_stats,
)
from .models import create_model, StockDataset
from .losses import create_loss_function, create_weighted_sampler
from .training import set_seed, train_model, evaluate, compute_metrics, save_model, fit_temperature
from .plotting import plot_all_results
from .predict import predict_specific_stocks
from .numba_optimizations import warmup as numba_warmup
from .memmap_dataset import (
    MemmapDataset, ChunkedMemmapLoader,
    load_memmap_datasets, load_into_ram,
    cache_exists, get_cache_info, preshuffle_cache,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Price Prediction Training')

    parser.add_argument('--max_stocks', type=int, default=100,
                        help='Max stocks to load (0 for all, default: 100)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['ce', 'focal', 'cb'],
                        help='Loss function type (default: from config.py DEFAULT_CONFIG)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--predict_stocks', type=str, nargs='+', default=['001270', '300788'],
                        help='Stock codes to predict after training')
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Use disk-based data loading to reduce memory usage')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use existing cached data (skip data processing)')
    parser.add_argument('--data_cache_dir', type=str, default=None,
                        help='Directory to cache processed data (default: stock_data/cache)')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'tft'],
                        help='Model architecture: transformer (default) or tft')

    return parser.parse_args()


def collect_tft_interpretability(
    model,
    loader,
    device: str,
    max_batches: int = 20,
) -> dict:
    """
    Run up to max_batches through TFT and average interpretability outputs:
      - enc_vsn: (186,) mean encoder VSN softmax weights across samples + timesteps
      - dec_vsn: (27,)  mean decoder VSN softmax weights across samples + timesteps
      - attn:    (35,35) mean attention weights across samples

    Called once after training on the test loader.
    """
    model.eval()
    enc_list, dec_list, attn_list = [], [], []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n >= max_batches:
                break
            sequences = batch[0].to(device)
            sectors   = batch[2].to(device)
            industries = batch[3].to(device) if len(batch) > 3 else None
            # future_inputs: last element if it's a 3-D tensor
            future_inputs = None
            if len(batch) >= 5:
                last = batch[-1]
                if isinstance(last, torch.Tensor) and last.dim() == 3:
                    future_inputs = last.to(device)
            if future_inputs is None:
                continue
            _ = model(sequences, future_inputs, sectors, industries)
            if model._enc_vsn_weights is not None:
                enc_list.append(model._enc_vsn_weights.float().cpu())
            if model._dec_vsn_weights is not None:
                dec_list.append(model._dec_vsn_weights.float().cpu())
            if model._attn_weights is not None:
                attn_list.append(model._attn_weights.float().cpu())

    if not enc_list:
        return {}

    enc_wts  = torch.cat(enc_list,  dim=0).mean(dim=(0, 1)).numpy()   # (186,)
    dec_wts  = torch.cat(dec_list,  dim=0).mean(dim=(0, 1)).numpy()   # (27,)
    attn_wts = torch.cat(attn_list, dim=0).mean(dim=0).numpy()        # (35, 35)
    return {'enc_vsn': enc_wts, 'dec_vsn': dec_wts, 'attn': attn_wts}


def main(config: Optional[dict] = None):
    """
    Main training and evaluation pipeline.

    Args:
        config: Configuration dictionary. If None, uses defaults with CLI args.
    """
    # Parse arguments and create config
    args = parse_args()

    if config is None:
        overrides = dict(
            max_stocks=args.max_stocks if args.max_stocks > 0 else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
        )
        # Only override loss_type if explicitly passed on the command line
        if args.loss_type is not None:
            overrides['loss_type'] = args.loss_type
        overrides['model_type'] = args.model_type
        config = get_config(**overrides)

    print("=" * 70)
    print("Multi-Class Stock Price Change Classification using Transformer")
    print(f"Predicting {NUM_CLASSES} percentage change buckets")
    print("=" * 70)

    # Warm up numba JIT compilation (avoids compilation overhead during processing)
    print("\nWarming up numba JIT compilation...")
    numba_warmup()
    print("Numba JIT ready.")

    # Set random seed
    set_seed(config['random_seed'])

    device = config['device']
    print(f"\nUsing device: {device}")

    # Determine cache directory
    cache_dir = getattr(args, 'data_cache_dir', None)
    if cache_dir is None:
        cache_dir = os.path.join(config['data_dir'], 'cache')

    # Check if using cached data
    use_cache = getattr(args, 'use_cache', False)
    memory_efficient = getattr(args, 'memory_efficient', False)

    # Full dataset always requires memory-efficient mode — loading 2M+ sequences into RAM
    # is not feasible. Auto-enable to prevent OOM errors.
    if config.get('max_stocks') is None and not use_cache:
        if not memory_efficient:
            print("INFO: Full dataset detected (--max_stocks 0), auto-enabling --memory_efficient")
            memory_efficient = True

    # interleaved_val split only works with the memmap pipeline (MemmapDataWriter.finalize).
    # The standard in-memory path uses split_data() which only supports random splits.
    if SPLIT_MODE == 'interleaved_val' and not use_cache and not memory_efficient:
        print("INFO: SPLIT_MODE='interleaved_val' requires memory_efficient mode — auto-enabling")
        memory_efficient = True

    class_names = get_class_names()

    if use_cache:
        # =====================================================================
        # USE CACHED DATA: Skip all data processing
        # =====================================================================
        print("\n" + "-" * 40)
        print("Loading from CACHE (skipping data processing)")
        print("-" * 40)

        if not cache_exists(cache_dir):
            print(f"ERROR: No valid cache found at {cache_dir}")
            print("Run with --memory_efficient first to create cache.")
            return

        cache_info = get_cache_info(cache_dir)
        # TFT mode requires future_inputs in the cache; rebuild if missing
        model_type = config.get('model_type', 'transformer')
        if model_type == 'tft' and not cache_info.get('has_future_inputs', False):
            print("WARNING: cache missing future_inputs (TFT mode requires n_future_features in metadata).")
            print("Rebuilding cache for TFT mode (--use_cache flag overridden).")
            use_cache = False
        if use_cache:
            print(f"Cache directory: {cache_dir}")
            print(f"Total samples: {cache_info['total_samples']}")
            print(f"Sequence shape: ({cache_info['splits']['train']['n_samples']}, {cache_info['seq_length']}, {cache_info['n_features']})")

        # Pre-shuffle before opening memmaps — on Windows, renaming an open memmap
        # file causes PermissionError even after del, so this must run first.
        if not cache_info.get('train_preshuffled', False):
            print("\nPre-shuffling train split for sequential I/O (one-time)...")
            preshuffle_cache(cache_dir, random_seed=config.get('random_seed', 42))

        # Load datasets via memmap — data stays on disk, only one batch in RAM at a time
        print("\nOpening memmap datasets (no full RAM load)...")
        train_dataset, val_dataset, test_dataset, scaler = load_memmap_datasets(cache_dir)
        train_labels = train_dataset.get_labels()

        input_dim = cache_info['n_features']

        # Load sector data for model (needed for num_sectors / num_industries)
        sector_data    = load_sector_data(config['data_dir'])
        industry_to_id = (
            {ind: i for i, ind in enumerate(sector_data['industry'].dropna().unique())}
            if len(sector_data) > 0 and 'industry' in sector_data.columns
            else {}
        )
        industry_to_id['Unknown'] = len(industry_to_id)

        print(f"\nNumber of features: {cache_info['n_features']}")
        print(f"Number of classes: {NUM_CLASSES}")

        # Print class distribution
        unique, counts = np.unique(train_labels, return_counts=True)
        print("\nClass distribution (train):")
        for i, name in enumerate(class_names):
            count = counts[list(unique).index(i)] if i in unique else 0
            pct = 100 * count / train_labels.size
            print(f"  {name}: {count} ({pct:.1f}%)")

        print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    else:
        # =====================================================================
        # PROCESS DATA: Load and process stock data
        # =====================================================================

        # Track total data processing time
        data_processing_start = time.perf_counter()

        # Load sector data
        print("\n" + "-" * 40)
        print("Loading Sector Data")
        print("-" * 40)
        sector_data    = load_sector_data(config['data_dir'])
        industry_to_id = (
            {ind: i for i, ind in enumerate(sector_data['industry'].dropna().unique())}
            if len(sector_data) > 0 and 'industry' in sector_data.columns
            else {}
        )
        industry_to_id['Unknown'] = len(industry_to_id)

        # Load daily basic data
        print("\n" + "-" * 40)
        print("Loading Daily Basic Data")
        print("-" * 40)
        daily_basic = load_daily_basic_data(config['data_dir'])

        # Load market context (index valuation + global returns)
        print("\n" + "-" * 40)
        print("Loading Market Context (index + global)")
        print("-" * 40)
        market_context = load_market_context_data(config['data_dir'])

        # Load index membership (CSI300 / CSI500 / SSE50 weights)
        print("\n" + "-" * 40)
        print("Loading Index Membership")
        print("-" * 40)
        index_membership = load_index_membership_data(config['data_dir'])

        # Load stk_limit data (limit up/down prices — zeros if not downloaded yet)
        print("\n" + "-" * 40)
        print("Loading Stk Limit Data")
        print("-" * 40)
        stk_limit = load_stk_limit_data(config['data_dir'])

        # Load moneyflow data (institutional flow — zeros if not downloaded yet)
        print("\n" + "-" * 40)
        print("Loading Moneyflow Data")
        print("-" * 40)
        moneyflow = load_moneyflow_data(config['data_dir'])

        max_per_market = config['max_stocks'] // 2 if config['max_stocks'] else None

        if memory_efficient:
            # =================================================================
            # MEMORY-EFFICIENT MODE: stream stocks from disk one at a time —
            # never build the all_stocks dict in RAM
            # =================================================================
            print("\n" + "-" * 40)
            print("Collecting Stock File Paths (MEMORY-EFFICIENT MODE)")
            print("-" * 40)

            stock_files = []
            for market in ['sh', 'sz']:
                files = get_stock_files(config['data_dir'], market, max_per_market)
                stock_files.extend(files)

            if not stock_files:
                print("No stock files found. Please ensure data exists in stock_data/sh and stock_data/sz directories.")
                return

            print(f"Total stock files found: {len(stock_files)}")
            print(f"Data will be cached to: {cache_dir}")

            feature_start = time.perf_counter()

            print("\nComputing cross-sectional technical stats (first pass)...")
            cs_tech_stats = compute_cross_section_tech_stats(
                stock_files, config['min_data_points']
            )

            result = prepare_dataset_to_disk(
                stock_files,
                sector_data,
                daily_basic,
                output_dir=cache_dir,
                sequence_length=config['sequence_length'],
                forward_window=config['forward_window'],
                min_data_points=config['min_data_points'],
                max_sequences_per_stock=config['max_sequences_per_stock'],
                train_ratio=config['train_ratio'],
                val_ratio=config['val_ratio'],
                random_seed=config['random_seed'],
                market_context=market_context,
                index_membership=index_membership,
                stk_limit=stk_limit if len(stk_limit) > 0 else None,
                moneyflow=moneyflow if len(moneyflow) > 0 else None,
                split_mode=SPLIT_MODE,
                data_dir=config['data_dir'],
                cs_tech_stats=cs_tech_stats,
            )

            feature_time    = time.perf_counter() - feature_start
            scaler          = result['scaler']
            metadata        = result['metadata']
            industry_to_id  = result.get('industry_to_id', {})

            # daily_basic freed inside prepare_dataset_to_disk; clear local ref
            del daily_basic
            gc.collect()

            data_processing_time = time.perf_counter() - data_processing_start
            print(f"\nData processing completed in {data_processing_time:.2f}s")
            print(f"  Feature engineering: {feature_time:.2f}s (numba-optimized)")

            # Load datasets via memmap — data stays on disk, only one batch in RAM at a time
            print("\nOpening memmap datasets (no full RAM load)...")
            train_dataset, val_dataset, test_dataset, scaler = load_memmap_datasets(cache_dir)
            train_labels = train_dataset.get_labels()

            input_dim = metadata['n_features']

            print(f"\nTotal samples: {metadata['total_samples']}")
            print(f"Sequence shape: ({metadata['splits']['train']['n_samples']}, {metadata['seq_length']}, {metadata['n_features']})")
            print(f"Number of features: {metadata['n_features']}")
            print(f"Number of classes: {NUM_CLASSES}")

            unique, counts = np.unique(train_labels, return_counts=True)
            print("\nClass distribution (train):")
            for i, name in enumerate(class_names):
                count = counts[list(unique).index(i)] if i in unique else 0
                pct = 100 * count / train_labels.size
                print(f"  {name}: {count} ({pct:.1f}%)")

            print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        else:
            # =================================================================
            # STANDARD MODE: Load all data into memory
            # =================================================================
            print("\n" + "-" * 40)
            print("Loading Stock Data")
            print("-" * 40)

            all_stocks = {}
            for market in ['sh', 'sz']:
                market_stocks = load_stock_data(
                    config['data_dir'],
                    market,
                    max_stocks=max_per_market,
                    min_data_points=config['min_data_points'],
                    num_workers=config['num_workers']
                )
                all_stocks.update(market_stocks)

            if not all_stocks:
                print("No stock data found. Please ensure data exists in stock_data/sh and stock_data/sz directories.")
                return

            print(f"Total stocks loaded: {len(all_stocks)}")

            print("\n" + "-" * 40)
            print("Preparing Dataset (numba-optimized feature engineering)")
            print("-" * 40)

            feature_start = time.perf_counter()
            sequences, labels, sectors, dates = prepare_dataset(
                all_stocks,
                sector_data,
                daily_basic=daily_basic,
                sequence_length=config['sequence_length'],
                forward_window=config['forward_window'],
                max_sequences_per_stock=config['max_sequences_per_stock'],
                num_workers=config['num_workers'],
                stk_limit=stk_limit if len(stk_limit) > 0 else None,
                moneyflow=moneyflow if len(moneyflow) > 0 else None,
            )
            feature_time = time.perf_counter() - feature_start

            # Free memory
            del all_stocks, daily_basic
            gc.collect()

            # Report data processing time
            data_processing_time = time.perf_counter() - data_processing_start
            print(f"\nData processing completed in {data_processing_time:.2f}s")
            print(f"  Feature engineering: {feature_time:.2f}s (numba-optimized)")

            print(f"\nTotal samples: {len(sequences)}")
            print(f"Sequence shape: {sequences.shape}")
            print(f"Number of features: {sequences.shape[2]}")
            print(f"Number of classes: {NUM_CLASSES}")

            # Print class distribution
            unique, counts = np.unique(labels, return_counts=True)
            print("\nClass distribution:")
            for i, name in enumerate(class_names):
                count = counts[list(unique).index(i)] if i in unique else 0
                pct = 100 * count / len(labels)
                print(f"  {name}: {count} ({pct:.1f}%)")

            # Save input dimension
            input_dim = sequences.shape[2]

            # Split data (random permutation across all time periods)
            (train_sequences, train_labels, train_sectors,
             val_sequences, val_labels, val_sectors,
             test_sequences, test_labels, test_sectors) = split_data(
                sequences, labels, sectors,
                config['train_ratio'], config['val_ratio'],
            )

            # Free memory
            del sequences, labels, sectors, dates
            gc.collect()

            print(f"\nTrain: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")

            # Normalize data
            print("Normalizing data...")
            train_sequences, val_sequences, test_sequences, scaler = normalize_data(
                train_sequences, val_sequences, test_sequences
            )

            # Create datasets
            train_dataset = StockDataset(train_sequences, train_labels, train_sectors)
            val_dataset = StockDataset(val_sequences, val_labels, val_sectors)
            test_dataset = StockDataset(test_sequences, test_labels, test_sectors)

    # Create dataloaders
    use_pin_memory = str(device).startswith('cuda')
    batch_size     = config['batch_size']

    if memory_efficient or use_cache:
        # Training: ChunkedMemmapLoader — background thread prefetches next chunk
        # while GPU trains on current chunk. No multiprocessing spawn overhead.
        # Reads large sequential slices (fast SSD throughput) then shuffles in RAM.
        train_loader = ChunkedMemmapLoader(
            cache_dir     = cache_dir,
            split         = 'train',
            batch_size    = batch_size,
            chunk_samples = config.get('chunk_samples', 200_000),
            seed          = config.get('random_seed', 42),
        )
    else:
        # Standard mode: data is already in RAM — use a regular DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=use_pin_memory)

    # Val/test: standard DataLoader with num_workers=0 (small splits, sequential)
    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=use_pin_memory)

    # Create model
    print("\n" + "-" * 40)
    print("Model Architecture")
    print("-" * 40)

    num_sectors    = len(sector_data['sector'].unique()) if len(sector_data) > 0 else 0
    # industry_to_id includes the 'Unknown' sentinel; subtract 1 for num_industries
    num_industries = max(0, len(industry_to_id) - 1)

    model_type = config.get('model_type', 'transformer')
    if model_type == 'tft':
        from .models import create_tft_model
        model = create_tft_model(config, num_sectors, num_industries).to(device)
        print("Model: TemporalFusionTransformer (TFT)")
    else:
        model = create_model(config, input_dim, num_sectors, num_industries).to(device)
        print("Model: TransformerClassifier")

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create loss function
    print("\n" + "-" * 40)
    print("Loss Function Configuration")
    print("-" * 40)

    # class_counts: (H, C) — per-horizon counts used for class-weight computation
    if train_labels.ndim == 2:                   # (N, H) multi-horizon
        class_counts = np.vstack([
            np.bincount(train_labels[:, h], minlength=NUM_CLASSES)
            for h in range(NUM_HORIZONS)
        ])                                        # shape (H, C)
    else:                                         # (N,) single-horizon fallback
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)

    criterion = create_loss_function(
        loss_type=config.get('loss_type', 'focal'),
        num_classes=NUM_CLASSES,
        class_counts=class_counts,
        device=device,
        gamma=config.get('focal_gamma', 2.0),
        beta=config.get('cb_beta', 0.9999),
        label_smoothing=config.get('label_smoothing', 0.0),
        use_class_weights=config.get('use_class_weights', True),
        horizon_weights=HORIZON_WEIGHTS,
        num_horizons=NUM_HORIZONS,
        use_relative_head=config.get('use_relative_head', False),
        num_relative_classes=NUM_RELATIVE_CLASSES,
        relative_head_weight=config.get('relative_head_weight', 0.3),
    )

    cc_flat = class_counts.sum(axis=0) if class_counts.ndim == 2 else class_counts
    print(f"Class counts (summed): min={cc_flat.min()}, max={cc_flat.max()}, ratio={cc_flat.max()/cc_flat.min():.1f}x")

    # Train model
    model, history = train_model(model, train_loader, val_loader, criterion, config, device)

    # ── Probability calibration (temperature scaling) ─────────────────────────
    print("\n" + "-" * 40)
    print("Probability Calibration (Temperature Scaling)")
    print("-" * 40)
    temperature_scaler = fit_temperature(model, val_loader, device, NUM_HORIZONS)

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n" + "-" * 40)
    print("Final Evaluation on Test Set")
    print("-" * 40)

    test_loss, test_predictions, test_labels_true, test_probs = evaluate(
        model, test_loader, criterion, device,
        temperature_scaler=temperature_scaler,
    )

    metrics = compute_metrics(test_labels_true, test_predictions)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Accuracy  (mean horizons): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1 Score  (mean horizons): {metrics['f1']:.4f}")

    # Per-horizon metrics when available
    if test_labels_true.ndim == 2:
        from .config import get_horizon_name
        for h in range(NUM_HORIZONS):
            hname = get_horizon_name(h)
            acc_h = metrics.get(f'accuracy_h{h}', float('nan'))
            f1_h  = metrics.get(f'f1_h{h}',       float('nan'))
            print(f"  {hname}: accuracy={acc_h:.4f}  f1={f1_h:.4f}")

    # Classification report and confusion matrix — use horizon-averaged predictions
    if test_labels_true.ndim == 2:
        # Flatten to horizon 1 (day 4) for display purposes
        tl_flat = test_labels_true[:, 1]
        tp_flat = test_predictions[:, 1]
    else:
        tl_flat = test_labels_true
        tp_flat = test_predictions

    print("\nClassification Report (day-4 horizon):")
    labels_in_data = sorted(list(set(tl_flat) | set(tp_flat)))
    target_names   = [class_names[i] for i in labels_in_data]
    print(classification_report(tl_flat, tp_flat,
                                labels=labels_in_data,
                                target_names=target_names,
                                zero_division=0))

    cm = confusion_matrix(tl_flat, tp_flat)
    print("Per-Class Accuracy (day-4):")
    for i, name in enumerate(class_names):
        if i < cm.shape[0]:
            class_total = cm[i].sum()
            if class_total > 0:
                class_acc = cm[i, i] / class_total if i < cm.shape[1] else 0
                print(f"  {name}: {class_acc:.2%} ({cm[i, i]}/{class_total})")

    random_baseline    = 1.0 / NUM_CLASSES
    unique, counts     = np.unique(tl_flat, return_counts=True)
    majority_class_pct = counts.max() / len(tl_flat)
    print(f"\nRandom baseline ({NUM_CLASSES} classes): {random_baseline:.4f}")
    print(f"Majority class baseline: {majority_class_pct:.4f}")
    print(f"Model improvement over random:   {(metrics['accuracy'] - random_baseline)*100:.2f}%")
    print(f"Model improvement over majority: {(metrics['accuracy'] - majority_class_pct)*100:.2f}%")

    # Visualisations — pass day-4 slice for plots (single-horizon API)
    plots_dir = os.path.join(os.path.dirname(config['data_dir']), 'plots', 'dl_results')
    probs_for_plot = test_probs[:, 1, :] if test_probs.ndim == 3 else test_probs

    # Collect TFT interpretability weights (VSN + attention) for visualization
    tft_interp = None
    if model_type == 'tft':
        tft_interp = collect_tft_interpretability(model, test_loader, device)

    plot_all_results(history, tl_flat, tp_flat, probs_for_plot, plots_dir,
                     tft_interp=tft_interp, model_type=model_type)

    # Save model (includes temperatures)
    # Store num_horizons in config so _load_checkpoint can reconstruct correctly
    config['num_horizons'] = NUM_HORIZONS
    model_type_str = config.get('model_type', 'transformer')
    model_path = os.path.join(config['data_dir'],
                               'tft_classifier.pth' if model_type_str == 'tft' else 'transformer_classifier.pth')
    save_model(model, config, scaler, history, metrics, model_path,
               temperature_scaler=temperature_scaler)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Predict for specific stocks
    if hasattr(args, 'predict_stocks') and args.predict_stocks:
        predictions = predict_specific_stocks(
            stock_codes=args.predict_stocks,
            model_path=model_path,
            data_dir=config['data_dir'],
            sector_data=sector_data,
            device=device
        )

        # Save predictions
        predictions_path = os.path.join(config['data_dir'], 'stock_predictions.json')
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"\nPredictions saved to: {predictions_path}")

    return model, history, metrics


if __name__ == '__main__':
    main()
