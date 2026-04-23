# deeptime — metrics reference & useful commands

All commands run from the repo root (`D:\didi\stock\tushare`), using the venv Python.

## What the metrics mean

Defined in `deeptime/training.py:28-70`.

```python
def compute_ic(pred, target):        # Spearman rank correlation
    return float(spearmanr(pred, target)[0])

def compute_hit_rate(pred, target):  # Fraction of correctly-signed preds
    return float(np.mean(np.sign(pred) == np.sign(target)))
```

### IC — Information Coefficient

Spearman rank correlation between prediction and actual excess return, computed across all (stock, day) pairs in a split (or cross-sectionally per day for rolling / sector IC plots). Range ≈ [−1, 1].

- IC = 0: no ranking ability.
- IC = 1: perfect stock-by-stock ranking.
- **Ignores magnitude** — scaling preds by any positive constant leaves IC unchanged.

### HR — Hit Rate

Fraction of predictions whose sign matches the actual sign. Range [0, 1], baseline ≈ 0.5.

- HR = 0.5: coin flip.
- HR = 0.55: directionally correct 55% of the time.
- **Ignores rank and magnitude** — only the sign matters.

### How IC, HR, RMSE/MAE relate to prediction accuracy

| Metric         | What it rewards       | What it ignores               |
| -------------- | --------------------- | ----------------------------- |
| **IC**         | Correct **ranking**   | Magnitude, per-day bias        |
| **HR**         | Correct **direction** | Ranking within sign, magnitude |
| **RMSE / MAE** | Correct **magnitude** | Ranking, sign                  |

Rough analytic bridge (jointly-normal approximation): `HR ≈ 0.5 + arcsin(IC)/π ≈ 0.5 + IC/π`.

| IC   | ~HR   |
| ---- | ----- |
| 0.02 | 0.506 |
| 0.05 | 0.516 |
| 0.10 | 0.532 |
| 0.20 | 0.564 |

That's why HR moves in tiny increments — a 1–2 pp HR change can correspond to a large IC change.

**Which metric to trust for "accuracy":**

- **Ranking / long-short portfolios** → IC is the right target. Grinold's fundamental law: `Sharpe ≈ IC · √breadth`, so IC 0.05 over 200 independent bets ≈ Sharpe 0.7.
- **Long-only top-N picks** → HR of the top quantile matters more than overall HR, and IC still dominates.
- **Position sizing / calibrated probabilities** → RMSE/MAE matter because you need the magnitude right, not just the rank.
- **Backtest return** → combination: IC sets alpha per trade, HR sets win-rate, RMSE sets confidence-interval width for sizing.

### A-share practical thresholds (daily / multi-day excess)

| IC          | Interpretation                           | Expected HR     |
| ----------- | ---------------------------------------- | --------------- |
| < 0.02      | Noise                                    | ~0.50           |
| 0.02 – 0.05 | Weak, tradeable with diversification     | 0.506 – 0.516   |
| 0.05 – 0.10 | Solid                                    | 0.516 – 0.532   |
| > 0.10      | Strong — check for leakage               | > 0.53          |

### Failure modes — read IC and HR together

`compute_regression_metrics()` emits `ic_{h}`, `hr_{h}`, `mae_{h}`, `rmse_{h}` per horizon plus `ic_mean`. Reading them together catches patterns a single metric hides:

- **High IC + HR ≈ 0.5** → model ranks correctly but gets the market-direction offset wrong (often fixable by target demeaning).
- **High HR + IC ≈ 0** → model predicts the market drift for everyone on a given day (constant per-day output, zero cross-sectional signal).
- **Good IC/HR + high RMSE** → ranks and signs fine but magnitudes are off; fine for ranking, bad for sizing.
- **Low IC + good RMSE** → model predicts the mean well but can't discriminate; useless for a trading signal.

---

## Training

Entry point: `deeptime/main.py`. Trains the Enhanced TFT regression model on the 5-horizon (days 1–5) excess-return target.

### Smoke test (cache build + short run)

```bash
# 100 stocks, 15 epochs — finishes in minutes, builds the cache
./venv/Scripts/python -m deeptime.main --max_stocks 100 --epochs 15
```

### Full training

```bash
# First full run — builds stock_data/deeptime_cache/ (~30-60 min preprocess)
./venv/Scripts/python -m deeptime.main --max_stocks 0 --epochs 50

# Subsequent runs — skip preprocessing, read from cache
./venv/Scripts/python -m deeptime.main --use_cache --epochs 100

# RTX 5090 preset (batch=512, hidden=256, heads=8)
./venv/Scripts/python -m deeptime.main --use_cache --preset rtx5090 --epochs 100

# Aggressive preset (batch=768, hidden=384) — more capacity, more VRAM
./venv/Scripts/python -m deeptime.main --use_cache --preset rtx5090_aggressive --epochs 100
```

### Sanity checks only

```bash
# Run data-quality checks and exit (no training)
./venv/Scripts/python -m deeptime.main --sanity_only
```

### Evaluate / predict only (skip training, reload checkpoint)

```bash
./venv/Scripts/python -m deeptime.main --use_cache --predict_only
```

### Hyperparameter overrides

```bash
# Target / loss
./venv/Scripts/python -m deeptime.main --use_cache --target_mode excess --loss_type huber+ic

# Optimizer / schedule
./venv/Scripts/python -m deeptime.main --use_cache \
  --lr 3e-5 --weight_decay 0.05 --warmup_epochs 5 --lr_schedule cosine \
  --max_grad_norm 0.5 --dropout 0.15 --patience 15

# Architecture
./venv/Scripts/python -m deeptime.main --use_cache \
  --hidden 256 --heads 8 --lstm_layers 2 --seq_len 60

# Disable features
./venv/Scripts/python -m deeptime.main --use_cache --no_swa --no_amp

# Loader / memory tuning
./venv/Scripts/python -m deeptime.main --use_cache \
  --batch_size 512 --num_workers 0 --prefetch 3 --chunk_samples 8192
./venv/Scripts/python -m deeptime.main --use_cache --preload   # full train set → RAM
```

### Cache invalidation

Cache is baked from feature engineering; delete after changing any feature code:

```bash
rm -rf stock_data/deeptime_cache/
```

---

## Prediction

### Score train / val / test splits from cache

`deeptime/predict_all.py` loads the saved checkpoint, scores the chosen split, prints per-horizon IC / MAE / HR, and dumps a CSV to `plots/deeptime_results/predictions_{split}.csv`.

```bash
# Default: score the test split
./venv/Scripts/python -m deeptime.predict_all

# Score a specific split
./venv/Scripts/python -m deeptime.predict_all --split val
./venv/Scripts/python -m deeptime.predict_all --split train
./venv/Scripts/python -m deeptime.predict_all --split all

# Top/bottom-N stocks by mean prediction
./venv/Scripts/python -m deeptime.predict_all --split test --top 20

# Larger batch for faster scoring on big GPUs
./venv/Scripts/python -m deeptime.predict_all --split test --batch_size 1024
```

### Live inference — tomorrow's prediction from the latest bar

`deeptime/predict_live.py` reads fresh data from `stock_data/`, builds a `seq_len`-long window ending at the most recent bar, runs the checkpoint, and prints top/bottom-N stocks by predicted multi-horizon return.

```bash
# Default: all stocks, top 20 each way
./venv/Scripts/python -m deeptime.predict_live

# Limit universe
./venv/Scripts/python -m deeptime.predict_live --max_stocks 500 --top 30

# Architecture overrides must match the training checkpoint
./venv/Scripts/python -m deeptime.predict_live --hidden 256 --heads 8
```

---

## Artifacts

| Path                                          | Produced by                          | Contents                                         |
| --------------------------------------------- | ------------------------------------ | ------------------------------------------------ |
| `stock_data/deeptime_cache/`                  | training (preprocess)                | Memmapped obs / future / targets + anchor_dates  |
| `stock_data/deeptime_model.pth`               | training                             | Best-val-IC checkpoint (model_state + meta)      |
| `plots/deeptime_results/predictions_{split}.csv` | `predict_all`                     | Per-(stock, date, horizon) pred vs target        |
| `plots/deeptime_results/*.png`                | training + `predict_all`             | Diagnostic plots (see below)                     |

Plots written to `plots/deeptime_results/`:

- `training_history.png` — loss + val IC per epoch
- `pred_vs_actual_*.png` — 5-panel scatter per horizon
- `rolling_ic_heatmap.png` — IC over time × horizon
- `vsn_feature_importance.png` — Variable Selection Network weights
- `sector_ic_analysis.png` — IC and HR broken down by SW sector
- `temporal_attention_heatmap.png` — attention weight over lookback
- `error_distribution_regime.png` — residuals under different market regimes
- `sector_cross_attention.png` — SectorCrossAttention weights K=31 sectors

---

## Common flags (reference)

| Flag                                       | Default            | Notes                                                    |
| ------------------------------------------ | ------------------ | -------------------------------------------------------- |
| `--max_stocks N`                           | 100                | 0 = all stocks (stratified by SW L1 sector)              |
| `--epochs N`                               | 50                 | Early stopping via `--patience`                          |
| `--use_cache`                              | off                | Skip preprocessing, load from cache                      |
| `--predict_only`                           | off                | Skip training, just evaluate + plot                      |
| `--sanity_only`                            | off                | Run data checks and exit                                 |
| `--preset {rtx5090,rtx5090_aggressive}`    | none               | Hardware preset sets batch/hidden/heads together         |
| `--target_mode {excess,raw}`               | `excess`           | Excess = stock − CSI300; raw = stock pct_chg             |
| `--loss_type {huber,huber+ic}`             | `huber`            | `huber+ic` adds soft-IC regularizer                      |
| `--seq_len N`                              | 30 (60 in preset)  | Lookback length                                          |
| `--batch_size N`                           | 128                | Scaled per preset                                        |
| `--lr FLOAT`                               | 2e-5               | Peak LR (auto-scaled with batch unless `--no_lr_scale`)  |
| `--weight_decay FLOAT`                     | 0.05               | AdamW weight decay                                       |
| `--max_grad_norm FLOAT`                    | 0.5                | Gradient clip                                            |
| `--dropout FLOAT`                          | 0.15               | Model-wide dropout                                       |
| `--warmup_epochs N`                        | 2                  | LR warmup; use 5–8 for large batches                     |
| `--lr_schedule {cosine,flat}`              | `cosine`           | Cosine decay vs flat hold                                |
| `--no_swa / --swa_start / --swa_eval_every`| SWA on             | Stochastic Weight Averaging controls                     |
| `--patience N`                             | 15                 | Early-stopping patience                                  |
| `--no_amp`                                 | AMP on             | Disable mixed precision                                  |
| `--preload`                                | off                | Load entire train set to RAM                             |
| `--num_workers / --prefetch / --chunk_samples / --max_chunk_gb` | auto | Data-loader tuning                                |
