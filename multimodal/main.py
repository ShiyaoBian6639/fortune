"""
CLI entry point for the multimodal stock prediction pipeline.

Usage examples:

  # Step 1 — build MacBERT news embedding cache (one-time, ~10 min on GPU)
  python -m multimodal.main --mode preprocess

  # Step 2 — align price + news data, split, normalise (one-time per config)
  python -m multimodal.main --mode preprocess

  # Step 3 — Phase 1 training (frozen BERT)
  python -m multimodal.main --mode train --phase 1

  # Step 4 — Phase 2 training (unfrozen BERT, after rebuilding cache)
  python -m multimodal.main --mode train --phase 2

  # Evaluate best checkpoint on test set
  python -m multimodal.main --mode evaluate --phase 1

  # Full pipeline (preprocess + both training phases)
  python -m multimodal.main --mode all

  # Quick smoke-test with 5 stocks and 3 news days
  python -m multimodal.main --mode all --max_stocks 5 --max_days 3
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from dl.training import set_seed
from dl.losses import create_loss_function

from .config import (
    get_multimodal_config,
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    MM_NUM_CLASSES,
    MM_CLASS_NAMES,
)
from .text_encoder import (
    MacBERTEncoder,
    build_daily_news_cache,
    load_daily_news_cache,
    build_daily_token_cache,
    load_daily_token_cache,
)
from .data_pipeline import (
    build_aligned_dataset,
    split_and_normalize,
    build_predict_sequences,
    load_news_window,
)
from .dataset import (
    MultimodalStockDataset,
    MultimodalChunkedLoader,
    Phase2Dataset,
    create_val_test_dataloaders,
    create_phase2_dataloaders,
)
from .models import create_multimodal_model
from .training import (
    train_phase1,
    train_phase2,
    save_checkpoint,
    load_checkpoint,
    evaluate_multimodal,
    print_eval_report,
)


def _get_stock_files(config: dict, max_stocks: int = None) -> list:
    data_dir = config['data_dir']
    files = []
    for market in ('sh', 'sz'):
        market_dir = os.path.join(data_dir, market)
        if os.path.isdir(market_dir):
            files.extend(glob.glob(os.path.join(market_dir, '*.csv')))
    if max_stocks and len(files) > max_stocks:
        rng = np.random.default_rng(config['random_seed'])
        idx = rng.choice(len(files), max_stocks, replace=False)
        files = [files[i] for i in idx]
    return files


def run_preprocess(config: dict, max_days: int = None, rebuild_cache: bool = False):
    """Build news embedding cache, token cache, and aligned dataset."""
    cache_path       = config['news_cache_path']
    token_cache_path = config['token_cache_path']
    cache_dir        = config['multimodal_cache_dir']
    news_dir         = config['news_dir']
    # load_daily_basic_data internally appends 'daily_basic/' to this path
    daily_basic = config['data_dir']
    device      = config['device']

    # ── Step 1a: build or load BERT news embedding cache (Phase 1) ─────────
    if rebuild_cache or not os.path.exists(cache_path):
        print("\n[preprocess] Building MacBERT news embedding cache ...")
        build_daily_news_cache(
            news_dir=news_dir,
            cache_path=cache_path,
            model_name=config['bert_model_name'],
            device=device,
            max_articles_per_day=config['max_articles_per_day'],
            max_length=config['bert_max_length'],
            max_days=max_days,
        )
    else:
        print(f"[preprocess] News cache found at {cache_path}  (use --rebuild_cache to regenerate)")

    # ── Step 1b: build or load token cache (Phase 2 inline BERT) ───────────
    if rebuild_cache or not os.path.exists(token_cache_path):
        print("\n[preprocess] Building token cache for Phase 2 ...")
        build_daily_token_cache(
            news_dir=news_dir,
            cache_path=token_cache_path,
            model_name=config['bert_model_name'],
            max_articles_per_day=config['max_articles_per_day'],
            max_length=config['bert_max_length'],
            max_days=max_days,
        )
    else:
        print(f"[preprocess] Token cache found at {token_cache_path}  (use --rebuild_cache to regenerate)")

    # ── Step 2: load cache ──────────────────────────────────────────────────
    print("[preprocess] Loading news cache ...")
    daily_news = load_daily_news_cache(cache_path)
    if not daily_news:
        print("[preprocess] ERROR: news cache is empty. Run with --rebuild_cache.")
        return

    trading_calendar = sorted(daily_news.keys())
    print(f"[preprocess] {len(daily_news)} days in cache  "
          f"({trading_calendar[0]} → {trading_calendar[-1]})")

    # ── Step 3: check if aligned dataset already exists ────────────────────
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    if os.path.exists(metadata_path) and not rebuild_cache:
        print(f"[preprocess] Aligned dataset already exists at {cache_dir}  (use --rebuild_cache to regenerate)")
        return

    # ── Step 4: build aligned dataset ──────────────────────────────────────
    stock_files = _get_stock_files(config, max_stocks=config.get('max_stocks'))
    if not stock_files:
        print("[preprocess] ERROR: no stock CSV files found.")
        return

    print(f"[preprocess] Processing {len(stock_files)} stock files ...")
    build_aligned_dataset(
        stock_files=stock_files,
        daily_basic_dir=daily_basic,
        daily_news_cache=daily_news,
        config=config,
        output_cache_dir=cache_dir,
        trading_calendar=trading_calendar,
    )

    # ── Step 5: split + normalise ───────────────────────────────────────────
    print("[preprocess] Splitting and normalising ...")
    split_and_normalize(
        cache_dir=cache_dir,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
    )
    print("[preprocess] Done.")


def _inject_n_features(config: dict) -> dict:
    """Read n_features from cache metadata and add it to config.

    This ensures create_multimodal_model uses the same feature count as
    the data on disk, regardless of how many entries are in FEATURE_COLUMNS
    (some columns may not be computed during preprocessing and are absent).
    """
    import json as _json
    meta_path = os.path.join(config['multimodal_cache_dir'], 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = _json.load(f)
        n = meta.get('n_features')
        if n:
            config = {**config, 'n_features': n}
            print(f"[train] n_features={n}  (from cache metadata)")
    return config


def _read_class_counts(cache_dir: str, split: str = 'train') -> np.ndarray:
    """Read per-class sample counts from metadata.json.

    Replaces dataset.get_labels() + np.bincount() — avoids loading all labels
    into RAM since the counts are already stored in the metadata by the
    preprocessing step.
    """
    import json as _json
    meta_path = os.path.join(cache_dir, 'metadata.json')
    with open(meta_path) as f:
        meta = _json.load(f)
    counts = meta['splits'][split].get('class_counts')
    if counts is None:
        raise KeyError(
            f"'class_counts' missing from metadata.json[splits][{split}]. "
            "Re-run preprocessing with --rebuild_cache to regenerate it."
        )
    return np.array(counts, dtype=np.float64)


def run_train(config: dict, phase: int = 1, fresh: bool = False):
    """Train phase 1 or phase 2."""
    cache_dir      = config['multimodal_cache_dir']
    checkpoint_dir = config['checkpoint_dir']
    device         = config['device']

    set_seed(config['random_seed'])

    config = _inject_n_features(config)
    model = create_multimodal_model(config).to(device)

    if phase == 1:
        # BERT is not used during Phase 1 — embeddings are pre-computed in the
        # cache.  Defer loading until Phase 2 to save ~400 MB VRAM and avoid
        # the "Loading weights" startup cost.
        bert = None

        # MultimodalChunkedLoader for train: bulk sequential reads into RAM,
        # background-thread depth-2 prefetch — eliminates 100% disk from random seeks.
        train_loader = MultimodalChunkedLoader(
            cache_dir,
            split         = 'train',
            batch_size    = config.get('batch_size', 1024),
            chunk_samples = config.get('chunk_samples', 50_000),
            seed          = config.get('random_seed', 42),
        )
        val_loader, _ = create_val_test_dataloaders(cache_dir, config, splits=('val',))
        print(f"[dataset] Train: {train_loader.n_samples:,}  Val: {len(val_loader.dataset):,}"
              f"  batch_size: {config.get('batch_size', 1024)}")

        # Read class counts from metadata — already computed by preprocessing,
        # avoids loading 4M labels into RAM just to bincount them.
        class_counts = _read_class_counts(cache_dir, split='train')
        criterion = create_loss_function(
            loss_type='focal', num_classes=MM_NUM_CLASSES,
            class_counts=class_counts, device=device,
            gamma=config.get('focal_gamma', 2.0),
        )

        # Resume if a Phase 1 checkpoint exists (skip when --fresh is set)
        p1_model = os.path.join(checkpoint_dir, 'phase1_model.pth')
        if os.path.exists(p1_model) and not fresh:
            print(f"[train] Resuming from {p1_model}")
            model, _ = load_checkpoint(model, None, checkpoint_dir, phase=1, device=device)
        elif fresh and os.path.exists(p1_model):
            print(f"[train] --fresh: ignoring existing checkpoint, training from scratch")

        model, history = train_phase1(model, train_loader, val_loader, criterion, config, device)
        save_checkpoint(model, bert, config, history, {}, checkpoint_dir, phase=1)

    elif phase == 2:
        # Phase 2 uses the token cache so BERT can be called inline per-batch
        token_cache_path = config['token_cache_path']
        if not os.path.exists(token_cache_path):
            print(
                "[train] ERROR: token cache not found at:\n"
                f"  {token_cache_path}\n"
                "Run  --mode preprocess  first."
            )
            return

        print("[train/p2] Loading token cache ...")
        token_cache = load_daily_token_cache(token_cache_path)
        if not token_cache:
            print("[train] ERROR: token cache is empty.")
            return

        trading_calendar = sorted(token_cache.keys())
        print(f"[train/p2] {len(token_cache)} days in token cache  "
              f"({trading_calendar[0]} → {trading_calendar[-1]})")

        # Phase 2 uses a smaller batch to fit BERT + model on the same GPU
        p2_config = {**config, 'batch_size': config.get('phase2_batch_size', 32)}
        train_loader, val_loader, _ = create_phase2_dataloaders(
            cache_dir, token_cache, trading_calendar, p2_config
        )
        # Phase 2 uses a subsampled dataset (phase2_max_samples); get counts from
        # that subset via get_labels() since metadata only has full-split counts.
        train_labels = train_loader.dataset.get_labels()
        class_counts = np.bincount(train_labels, minlength=MM_NUM_CLASSES).astype(np.float64)
        criterion = create_loss_function(
            loss_type='focal', num_classes=MM_NUM_CLASSES,
            class_counts=class_counts, device=device,
            gamma=config.get('focal_gamma', 2.0),
        )

        # Load BERT only when actually needed (Phase 2 fine-tuning)
        bert = MacBERTEncoder(
            model_name=config['bert_model_name'],
            max_length=config['bert_max_length'],
            freeze=True,
            use_lora=config.get('use_lora', False),
            lora_r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_target_modules=config.get('lora_target_modules'),
            lora_dropout=config.get('lora_dropout', 0.1),
        ).to(device)

        # Must load Phase 1 weights before fine-tuning
        model, bert = load_checkpoint(model, bert, checkpoint_dir, phase=1, device=device)
        model, bert, history = train_phase2(
            model, bert, train_loader, val_loader, criterion, config, device
        )
        save_checkpoint(model, bert, config, history, {}, checkpoint_dir, phase=2)


def run_evaluate(config: dict, phase: int = 1):
    """Evaluate a saved checkpoint on the test set."""
    cache_dir      = config['multimodal_cache_dir']
    checkpoint_dir = config['checkpoint_dir']
    device         = config['device']

    config = _inject_n_features(config)
    val_loader_ev, test_loader = create_val_test_dataloaders(
        cache_dir, config, splits=('val', 'test')
    )

    # Use validation set class counts from metadata
    class_counts = _read_class_counts(cache_dir, split='val')
    criterion = create_loss_function(
        'focal', MM_NUM_CLASSES, class_counts, device, gamma=config.get('focal_gamma', 2.0)
    )

    model = create_multimodal_model(config).to(device)
    bert  = MacBERTEncoder(
        model_name=config['bert_model_name'],
        max_length=config['bert_max_length'],
        freeze=True,
        use_lora=config.get('use_lora', False),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_target_modules=config.get('lora_target_modules'),
        lora_dropout=config.get('lora_dropout', 0.1),
    ).to(device)
    model, bert = load_checkpoint(model, bert, checkpoint_dir, phase=phase, device=device)

    val_loss, preds, labels, _ = evaluate_multimodal(model, test_loader, criterion, device)
    print_eval_report(val_loss, preds, labels, split_name='test')


def run_predict(config: dict, output_path: str, phase: int = 2) -> None:
    """
    Run inference on the most recent data for every stock and write a CSV.

    Uses the pre-computed news embedding cache (news_embeddings.npz) — the same
    embeddings used in Phase 1 training.  For Phase 2 checkpoints the LoRA
    adapters add <0.3% of BERT parameters, so the pre-computed embeddings are a
    close approximation that avoids re-running BERT at prediction time.

    Output CSV columns:
        ts_code      stock code (e.g. '600000.SH')
        pred_date    last trading day in the 30-day input window (YYYYMMDD)
        pred_class   predicted bucket index  (0–9)
        pred_name    human-readable bucket label  (e.g. '>10%')
        confidence   probability of the predicted class
        prob_0…9     per-class softmax probabilities
    """
    import pandas as pd

    cache_dir      = config['multimodal_cache_dir']
    checkpoint_dir = config['checkpoint_dir']
    device         = config['device']

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler_path = os.path.join(cache_dir, 'price_scaler.npz')
    if not os.path.exists(scaler_path):
        print(f"[predict] ERROR: scaler not found at {scaler_path}. "
              "Run --mode preprocess first.")
        return
    scaler_data  = np.load(scaler_path)
    scaler_mean  = scaler_data['mean']
    scaler_scale = scaler_data['scale']

    # ── News embedding cache ──────────────────────────────────────────────────
    cache_path = config['news_cache_path']
    if not os.path.exists(cache_path):
        print(f"[predict] ERROR: news cache not found at {cache_path}. "
              "Run --mode preprocess first.")
        return
    print("[predict] Loading news cache ...")
    daily_news       = load_daily_news_cache(cache_path)
    trading_calendar = sorted(daily_news.keys())
    cal_index        = {d: i for i, d in enumerate(trading_calendar)}
    news_window      = config.get('news_window', 3)

    # ── Model ─────────────────────────────────────────────────────────────────
    config = _inject_n_features(config)
    model  = create_multimodal_model(config).to(device)
    # BERT not needed: we use pre-computed embeddings from the news cache
    model, _ = load_checkpoint(model, None, checkpoint_dir, phase=phase, device=device)
    model.eval()

    # ── Daily basic (fundamental metrics) ────────────────────────────────────
    from dl.data_processing import load_daily_basic_data

    daily_basic_by_ts: dict = {}
    data_dir = config['data_dir']
    if os.path.isdir(data_dir):
        try:
            db = load_daily_basic_data(data_dir)
            if db is not None and not db.empty:
                print("[predict] Pre-grouping daily_basic by ts_code ...")
                daily_basic_by_ts = {
                    code: grp.reset_index(drop=True)
                    for code, grp in db.groupby('ts_code', sort=False)
                }
        except Exception as e:
            print(f"[predict] Warning: could not load daily_basic: {e}")

    # ── Build sequences and run batched inference ─────────────────────────────
    stock_files = _get_stock_files(config)
    if not stock_files:
        print("[predict] ERROR: no stock CSV files found.")
        return

    print(f"[predict] Building sequences for {len(stock_files)} stocks ...")

    batch_price: list = []
    batch_news:  list = []
    batch_meta:  list = []   # (ts_code, date_str)
    results:     list = []
    batch_size   = min(config.get('batch_size', 1024), 1024)
    device_type  = device.split(':')[0]
    use_amp      = config.get('use_amp', True)

    def _flush():
        if not batch_price:
            return
        price_t = torch.tensor(np.stack(batch_price), dtype=torch.float32).to(device)
        news_t  = torch.tensor(np.stack(batch_news),  dtype=torch.float32).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=use_amp
        ):
            logits = model(price_t, news_t)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for (ts_code, date_str), pred, prob in zip(batch_meta, preds, probs):
            row = {
                'ts_code':    ts_code,
                'pred_date':  date_str,
                'pred_class': int(pred),
                'pred_name':  MM_CLASS_NAMES[pred],
                'confidence': float(prob[pred]),
            }
            for c in range(MM_NUM_CLASSES):
                row[f'prob_{c}'] = float(prob[c])
            results.append(row)
        batch_price.clear()
        batch_news.clear()
        batch_meta.clear()

    from tqdm import tqdm
    for sf in tqdm(stock_files, desc='  stocks', unit='stock'):
        ts_code   = os.path.basename(sf).replace('.csv', '')
        stk_basic = daily_basic_by_ts.get(ts_code)

        result = build_predict_sequences(
            sf, config, scaler_mean, scaler_scale,
            daily_basic=stk_basic,
            _prefiltered=(stk_basic is not None),
        )
        if result is None:
            continue

        price_seq, ts_code, date_str = result
        news_vec = load_news_window(
            date_str, daily_news, news_window, trading_calendar, cal_index
        )

        batch_price.append(price_seq)
        batch_news.append(news_vec)
        batch_meta.append((ts_code, date_str))

        if len(batch_price) >= batch_size:
            _flush()

    _flush()

    if not results:
        print("[predict] No predictions generated — check stock files and date range.")
        return

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.sort_values(['pred_date', 'ts_code']).reset_index(drop=True)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\n[predict] {len(df):,} predictions → {output_path}")
    print(f"  Date range : {df['pred_date'].min()} → {df['pred_date'].max()}")
    print(f"  Class distribution:")
    for i, name in enumerate(MM_CLASS_NAMES):
        n = int((df['pred_class'] == i).sum())
        print(f"    [{i}] {name:6s}: {n:6,}  ({100 * n / len(df):.1f}%)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multimodal Transformer for Chinese stock movement prediction.'
    )
    parser.add_argument(
        '--mode', choices=['preprocess', 'train', 'evaluate', 'predict', 'all'],
        default='all',
        help='Pipeline stage to run (default: all)',
    )
    parser.add_argument(
        '--phase', type=int, choices=[1, 2], default=None,
        help='Training phase (1=frozen BERT, 2=full fine-tune).  '
             'Required for train/evaluate; predict defaults to 2.',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path for --mode predict '
             '(default: stock_data/predictions/predictions_YYYYMMDD.csv)',
    )
    parser.add_argument(
        '--max_stocks', type=int, default=None,
        help='Limit number of stocks processed (for quick testing)',
    )
    parser.add_argument(
        '--max_days', type=int, default=None,
        help='Limit number of news days encoded (for quick testing)',
    )
    parser.add_argument(
        '--rebuild_cache', action='store_true',
        help='Force rebuild of news embedding cache and aligned dataset',
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override batch size from config',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Override device (cuda / cpu)',
    )
    parser.add_argument(
        '--fresh', action='store_true',
        help='Ignore existing checkpoint and train from scratch',
    )
    args = parser.parse_args()

    overrides = {}
    if args.max_stocks is not None:
        overrides['max_stocks'] = args.max_stocks
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.device is not None:
        overrides['device'] = args.device

    config = get_multimodal_config(**overrides)
    os.makedirs(config['multimodal_cache_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Multimodal Stock Transformer  [device: {config['device']}]")
    print(f"{'='*60}")

    if args.mode in ('preprocess', 'all'):
        run_preprocess(config, max_days=args.max_days, rebuild_cache=args.rebuild_cache)

    if args.mode in ('train', 'all'):
        phases = [1, 2] if args.phase is None else [args.phase]
        for ph in phases:
            run_train(config, phase=ph, fresh=args.fresh)

    if args.mode == 'evaluate':
        ph = args.phase if args.phase else 1
        run_evaluate(config, phase=ph)

    if args.mode == 'predict':
        from datetime import datetime as _dt
        ph = args.phase if args.phase else 2
        if args.output:
            out_path = args.output
        else:
            pred_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'stock_data', 'predictions',
            )
            out_path = os.path.join(
                pred_dir, f"predictions_{_dt.today().strftime('%Y%m%d')}.csv"
            )
        run_predict(config, out_path, phase=ph)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
