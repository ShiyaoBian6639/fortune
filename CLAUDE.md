# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment

Always use `./venv/Scripts/python` (never `python` or `python3`). All commands must be run from `D:\didi\stock\tushare`.

## Common Commands

```bash
# Quick smoke test (100 stocks, no cache required)
./venv/Scripts/python -m dl.main

# Full dataset — first run (builds cache, ~30-60 min)
./venv/Scripts/python -m dl.main --max_stocks 0 --memory_efficient --epochs 15

# Subsequent training runs (skip reprocessing, load from cache)
./venv/Scripts/python -m dl.main --use_cache --epochs 50

# Predict all stocks after training
./venv/Scripts/python -m dl.predict_all

# Predict specific stocks
./venv/Scripts/python -m dl.main --use_cache --predict_stocks 600000 000001 300750

# Invalidate stale cache (required after feature changes)
rm -rf stock_data/cache/

# Multimodal pipeline (price + news)
./venv/Scripts/python -m multimodal.main --mode preprocess   # build BERT embedding cache
./venv/Scripts/python -m multimodal.main --mode train --phase 1  # frozen BERT
./venv/Scripts/python -m multimodal.main --mode train --phase 2  # LoRA fine-tune
./venv/Scripts/python -m multimodal.main --mode evaluate --phase 2
./venv/Scripts/python -m multimodal.main --mode all          # full pipeline
./venv/Scripts/python -m multimodal.main --mode predict      # generate predictions CSV

# Data download
./venv/Scripts/python main.py       # download stock price + news data
./venv/Scripts/python get_original_data.py

# deeptime regression pipeline (pct_chg prediction, days 1-5)
./venv/Scripts/python -m deeptime.main --max_stocks 100 --epochs 15     # smoke test
./venv/Scripts/python -m deeptime.main --max_stocks 0 --epochs 50       # full dataset
./venv/Scripts/python -m deeptime.main --use_cache --epochs 100         # use cache
./venv/Scripts/python -m deeptime.main --use_cache --predict_only       # eval only
./venv/Scripts/python -m deeptime.main --sanity_only                    # data checks
rm -rf stock_data/deeptime_cache/                                       # invalidate cache
```

## Architecture Overview

### `deeptime/` — Regression Forecasting Pipeline (NEW)

Predicts **raw pct_chg** (days t+1 through t+5) as a regression problem. Entry point: `deeptime/main.py`.

**Key differences from `dl/`:**
- **Regression** (not classification): Huber loss, IC-based early stopping
- **5 horizons** [1,2,3,4,5] vs [3,4,5] in dl/
- **seq_len=60** (vs 30)
- **Excess return target**: `pct_chg − csi300_pct_chg` (regime-invariant)
- **Inter-stock**: `SectorCrossAttention` in model.py — O(N×K) pooling over K=31 sectors
- **Extended features**: fina_indicator (12 quarterly fundamentals), block_trade (4 features), extended moneyflow SM/MD (2 features), price limit ratios (2, also used as known-future)
- **Model**: Enhanced TFT (`deeptime/model.py`) reusing `dl/tft_model.py` building blocks
- **Cache**: `stock_data/deeptime_cache/` with float32 targets, separate obs/future arrays, anchor_dates

**Architecture:** `DeepTimeModel` = Variable Selection Networks → LSTM encoder → `SectorCrossAttention` → LSTM decoder → Interpretable Multi-Head Attention → 5 regression heads

**Plots saved to** `plots/deeptime_results/`: training_history, pred_vs_actual (5 panels), rolling_ic_heatmap, vsn_feature_importance, sector_ic_analysis, temporal_attention_heatmap, error_distribution_regime, sector_cross_attention.

**Cache invalidation**: Delete `stock_data/deeptime_cache/` whenever any feature engineering changes.

---

### `dl/` — Primary Deep Learning Pipeline

The core module. Entry point: `dl/main.py`.

**Data flow:**
1. Raw CSVs in `stock_data/sh/` and `stock_data/sz/` (one CSV per stock)
2. `data_processing.py` — feature engineering → 30-step × ~213-feature sequences
3. Labels: **cross-sectional relative returns** (stock return − CSI300 return) bucketed into 7 classes. This is intentionally relative so the model cannot game it by predicting bear in all regimes.
4. Multi-horizon: one label per horizon (day 3, 4, 5 ahead), one classifier head per horizon sharing a common Transformer backbone.

**Two data loading modes:**
- **Standard** (`--max_stocks 100`): loads everything into RAM; uses `StockDataset` + regular `DataLoader`.
- **Memory-efficient** (`--memory_efficient` or `--max_stocks 0`): streams stocks from disk, writes to `stock_data/cache/` as memory-mapped `.npy` files; uses `ChunkedMemmapLoader` (background thread prefetch, no multiprocessing spawn). Required for full dataset.

**Key config** (`dl/config.py`):
- `SPLIT_MODE = 'rolling_window'` — rolling walk-forward split (see section below).
- `CHANGE_BUCKETS` — 7-class symmetric boundaries for relative return classification.
- `FEATURE_COLUMNS` — the ordered list of ~213 features fed to the model.
- `CS_NORMALIZE_TECH_FEATURES` — features normalized cross-sectionally per trading day (rank within market peers), not globally.

**Model** (`dl/models.py`): `TransformerClassifier` — pre-norm Transformer encoder (GPT-2 style), Flash Attention 2 via `F.scaled_dot_product_attention`, fused QKV, sinusoidal positional encoding, mean-pool → per-horizon MLP heads. Default: d_model=192, 6 heads, 3 layers.

**Training** (`dl/training.py`): AdamW + cosine LR with warmup, FP16 AMP (`use_amp=True`), early stopping, temperature scaling for calibration after training.

**Loss** (`dl/losses.py`): default `ce` + `label_smoothing=0.1`. Class weights are disabled — they caused temperature explosion (T=127) on the full dataset due to 7.6× imbalance ratio amplification.

**Plots** saved to `plots/dl_results/` by `dl/plotting.py`.

### `multimodal/` — Price + News Fusion

Two-phase training:
- Phase 1: frozen MacBERT encoder, trains fusion head only
- Phase 2: LoRA fine-tunes BERT on top of Phase 1 checkpoint

Caches at `stock_data/cache/news_embeddings.npz` (Phase 1) and `stock_data/cache/news_tokens.npz` (Phase 2). Resume checkpoints auto-saved after each epoch to `stock_data/checkpoints/multimodal/phase2_resume*.pth`.

### `api/` — Tushare Data Download

Wrappers around Tushare Pro API. Token is in `dl/config.py` (`TUSHARE_TOKEN`). Downloads to `stock_data/`.

### `features/` — Supplemental Feature Data

`news.py` and `events.py` download news and macro events for all stocks.

### `backtest/` and `quant/`

Backtesting strategy and quantitative trading utilities. Separate from the DL pipeline.

## Rolling Walk-Forward Data Split

`SPLIT_MODE = 'rolling_window'` is the default and recommended mode. It implements a proper walk-forward split with no data leakage.

### How it works

Each fold has three sequential, non-overlapping windows:

```
Fold 0: [──── 12m TRAIN ────][purge][── 2m VAL ──][── 2m TEST ──]
Fold 1:                              [──── 12m TRAIN ────][purge][── 2m VAL ──][── 2m TEST ──]
Fold 2:                                                           [──── 12m TRAIN ────][purge]...
...
```

- **Train pool**: all dates falling in any fold's train window, minus purge gaps
- **Val pool**: all fold val windows pooled together
- **Test pool**: all fold test windows pooled + global holdout (2025-07-01+)
- **Purge gap** (35 trading days = seq_len 30 + max_fw 5): applied only at the train→val boundary of each fold. Prevents sequences in val from looking back into training data, removing feature contamination at the split edge.

### Leakage guarantee

Once a date is labeled val or test in any fold, the "only overwrite 'train'" rule prevents it from being re-labeled train in a later fold. No label leakage: labels are future returns (day+3/4/5), not values from other split windows.

### Tuning parameters (in `dl/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `ROLLING_TRAIN_MONTHS` | 12 | Training window length per fold |
| `ROLLING_VAL_MONTHS` | 2 | Validation window length per fold |
| `ROLLING_TEST_MONTHS` | 2 | Test window length per fold |
| `ROLLING_STEP_MONTHS` | 16 | Cursor advance per fold |
| `INTERLEAVED_TEST_START` | 20250701 | Global holdout start (always test) |

**Critical**: set `ROLLING_STEP_MONTHS = TRAIN + VAL + TEST` (default 16) for non-overlapping folds where each date appears in exactly one split role. A smaller step creates more folds but the accumulated val/test windows progressively consume training data — if step << TRAIN, training pool shrinks severely.

### Choosing step size

| Goal | Recommended step |
|---|---|
| Maximum training data, clean splits | `STEP = TRAIN + VAL + TEST` (default) |
| More evaluation coverage, some overlap | `STEP = VAL + TEST` (minimal, val/test never overlap) |
| Dense walk-forward CV (many folds) | `STEP = 1–3m` (training pool shrinks; use only with small TRAIN) |

### Typical result with defaults (50 stocks, 2017–2025)

```
Rolling walk-forward: 12m train / 2m val / 2m test, step=16m, 5 folds, purge=35 days
→ train: ~51%  val: ~9%  test: ~25%  gap: ~15%
```

### After changing split parameters

Always delete the cache before retraining — split labels are baked into the cache:
```bash
rm -rf stock_data/cache/
```

## Important Invariants

- **OBV features are non-stationary**: raw cumulative OBV causes 3–7σ test-set drift after StandardScaler. `csi300_obv` and all `*_obv` index features are intentionally excluded from `MARKET_CONTEXT_FEATURES`. Do not add them back without first-differencing or ratio-transforming.
- **Cache invalidation**: if `FEATURE_COLUMNS` or any feature engineering logic changes, delete `stock_data/cache/` before the next run or you'll silently train on stale features with wrong dimensions.
- **`interleaved_val` split requires `--memory_efficient`**: the in-memory path (`split_data()`) only supports random splits.
- All commands run from repo root (`D:\didi\stock\tushare`), not from inside `dl/`.
