# xgbmodel — useful commands

All commands run from the repo root (`D:\didi\stock\tushare`), using the venv Python.

## Training

### Walk-forward CV (recommended — used for the canonical model)

```bash
# Full walk-forward, GPU, default 12w/2w/2w folds, demean target
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --device cuda

# CPU fallback
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward

# Tight schedule — user's original ask "3w train / 1w val / 1w test"
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward \
  --fold_train_weeks 3 --fold_val_weeks 1 --fold_test_weeks 1 --fold_step_weeks 1

# Expanding train window (keeps every day of history in train)
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --expanding_train

# Limit folds for a smoke test (runs first 5 folds only)
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --max_folds 5
```

### Fixed single train/val/test split

```bash
# Default cutoffs: train<20240101 ≤ val < 20250101 ≤ test
./venv/Scripts/python -m xgbmodel.main --split_mode fixed --device cuda

# Override cutoffs
./venv/Scripts/python -m xgbmodel.main --split_mode fixed \
  --train_start 20180101 --val_start 20240101 --test_start 20250101
```

### Smoke test (small subset)

```bash
# Quick run on 100 stocks, CPU, fixed split — finishes in a few minutes
./venv/Scripts/python -m xgbmodel.main --split_mode fixed --max_stocks 100

# 500 stocks, walk-forward, first 10 folds
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward \
  --max_stocks 500 --max_folds 10 --device cuda
```

### Target configuration

```bash
# Default: excess vs CSI300 + per-day cross-sectional demean
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward

# A/B: turn OFF cross-sectional demean (old behavior)
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --cs_target_norm none

# Per-day z-score (demean + scale)
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --cs_target_norm zscore

# Raw next-day pct_chg (no market subtraction) + demean
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --target raw

# Predict 5 days ahead instead of next day
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --forward_window 5
```

### XGBoost hyperparameter overrides

```bash
# Quick LR/depth sweep on the fixed split
./venv/Scripts/python -m xgbmodel.main --split_mode fixed --device cuda \
  --learning_rate 0.02 --max_depth 6 --n_estimators 3000 \
  --subsample 0.75 --colsample_bytree 0.65
```

## Prediction

All predict modes reload `stock_data/models/xgb_pct_chg.json` saved by training.

```bash
# Score the most recent trade_date in the panel — live signal, sorted desc by pred.
# Writes stock_predictions_xgb.csv at repo root with probability columns attached.
./venv/Scripts/python -m xgbmodel.main --mode predict

# Score the held-out test window and re-render plots from saved preds
./venv/Scripts/python -m xgbmodel.main --mode predict_test

# Re-render plots only (reads from stock_data/models/xgb_preds/)
./venv/Scripts/python -m xgbmodel.main --mode plot

# Predict with a different test_start (picks up everything from that date onward)
./venv/Scripts/python -m xgbmodel.main --mode predict_test --test_start 20250601
```

## End-to-end (train + plot, default)

```bash
# Full walk-forward then plots — this is what --mode all does
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --device cuda
```

## Artifacts

| Path                                           | Produced by          | Contents                              |
| ---------------------------------------------- | -------------------- | ------------------------------------- |
| `stock_data/models/xgb_pct_chg.json`           | train / walk-forward | Canonical XGBoost booster             |
| `stock_data/models/xgb_pct_chg.features.json`  | train                | Ordered feature list                  |
| `stock_data/models/xgb_pct_chg.meta.json`      | train                | Per-fold metrics + feature importance |
| `stock_data/models/xgb_preds/val.csv`          | train                | OOF / val predictions                 |
| `stock_data/models/xgb_preds/test.csv`         | train                | Test-window predictions               |
| `stock_data/models/xgb_preds/test_full.csv`    | `--mode predict_test`| Full test-window scored panel         |
| `stock_predictions_xgb.csv` (repo root)        | `--mode predict`     | Live next-day predictions + probs     |
| `plots/xgb_results/*.png`                      | `--mode plot`/`all`  | Diagnostic plots                      |

## Common flags (reference)

| Flag                                                  | Default                       | Notes                                                              |
| ----------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------ |
| `--mode {train,predict,predict_test,plot,all}`        | `all`                         | `all` = train + plot                                               |
| `--split_mode {fixed,walk_forward}`                   | `fixed`                       | Prefer `walk_forward` for reporting                                |
| `--device {cpu,cuda}`                                 | `cpu`                         | `cuda` uses GPU hist                                               |
| `--max_stocks N`                                      | 0 (all)                       | Evenly samples SH+SZ                                               |
| `--target {raw,excess}`                               | `excess`                      | Subsumed by `--cs_target_norm` for rank-IC purposes                |
| `--cs_target_norm {none,demean,zscore}`               | `demean`                      | Per-day cross-section normalization; kills macro-feature dominance |
| `--forward_window N`                                  | 1                             | Horizon in trading days                                            |
| `--fold_{train,val,test,step}_weeks`                  | 12 / 2 / 2 / 2                | Walk-forward window sizes                                          |
| `--purge_days / --embargo_days`                       | 5 / 2                         | de Prado §7.4 leakage guards                                       |
| `--expanding_train`                                   | off                           | Walk-forward with growing train window                             |
| `--max_folds N`                                       | 0 (all)                       | Cap fold count for smoke runs                                      |
| `--learning_rate / --max_depth / --n_estimators ...`  | see `config.py` `XGB_PARAMS`  | Overrides applied on top of defaults                               |
