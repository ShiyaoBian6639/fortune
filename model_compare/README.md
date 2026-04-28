# `model_compare/` — Apples-to-apples regression model comparison

Six model families on the **same feature panel** (174 features), the **same
walk-forward CV** (212 folds, 12w/2w/2w with purge=5d / embargo=2d), and
**the same prediction CSV schema**. Output of every engine plugs straight into
`backtest.xgb_markowitz` and the dashboard's "model selection" section.

## Engines

| Engine | Type | Source | Notes |
|---|---|---|---|
| `xgb_default` | gradient boosting | `engines_gbm.py:XGBEngine` | mirrors current production model |
| `xgb_shallow` | GB variant | `XGBShallow` | `max_depth=3`, `min_child_weight=100` |
| `xgb_deep` | GB variant | `XGBDeep` | `max_depth=8`, `n_estimators=800` |
| `xgb_strong_reg` | GB variant | `XGBStrongReg` | `reg_lambda=10`, lower subsample |
| `lightgbm` | gradient boosting | `LightGBMEngine` | huber objective, leaf-wise |
| `catboost` | gradient boosting | `CatBoostEngine` | symmetric trees, GPU optional |
| `transformer_reg` | encoder-only Transformer | `transformer_reg.py` | **Pure** Vaswani-2017 architecture, regression head, T=30, d=192, 3 layers, sinusoidal PE, causal mask, mean-pool. No reuse of `dl/` (classifier). |
| `tft` | Temporal Fusion Transformer | `tft.py` | **Pure** Lim et al. 2019: VSN → LSTM → static enrichment GRN → temporal multi-head attention → output GRN → mean-pool → linear. No reuse of `deeptime/` (which has sector cross-attention). |

Each writes to `stock_data/models_<engine>/`:

```
stock_data/
├── models_lightgbm/
│   ├── meta.json                    fold-level metrics + summary
│   └── xgb_preds/
│       ├── val.csv                  OOF val predictions
│       └── test.csv                 OOF test predictions ← consumed by backtest
├── models_catboost/...
├── models_transformer_reg/...
├── models_tft/...
└── models_ensemble_{mean,rankavg}/  ← built by `model_compare.ensemble`
```

Existing `backtest/xgb_markowitz.py` doesn't need any changes — just point it
at a different `xgb_preds/test.csv`.

## CLI

```bash
# Quick smoke test (5 folds × 200 stocks)
./venv/Scripts/python -m model_compare.main --engine lightgbm --max_folds 5 --max_stocks 200

# Full walk-forward (~45 min for LightGBM, ~3-5 hr for TFT)
./venv/Scripts/python -m model_compare.main --engine lightgbm

# Multiple engines sequentially (panel built once, reused)
./venv/Scripts/python -m model_compare.main --engines xgb_shallow xgb_deep xgb_strong_reg

# GPU on for NN models
./venv/Scripts/python -m model_compare.main --engine transformer_reg --device cuda
./venv/Scripts/python -m model_compare.main --engine tft              --device cuda
```

## Ensemble + comparison

After at least two engines have completed walk-forward:

```bash
./venv/Scripts/python -m model_compare.ensemble \
    --models xgb_default lightgbm catboost transformer_reg tft
```

Produces:
- `stock_data/models_ensemble_mean/xgb_preds/test.csv` — uniform per-row average
- `stock_data/models_ensemble_rankavg/xgb_preds/test.csv` — per-day rank averaging
- `stock_data/models_ensemble_comparison.json` — per-model and per-ensemble metrics + correlation matrix
- Console table: rank IC, ICIR, hit rate, RMSE per model + correlation matrix

## Strict alignment guarantees

| Aspect | Spec |
|---|---|
| Target | `pct_chg(t+1) − csi300_pct_chg(t+1)`, cross-sectionally demeaned, ±11% clipped (same as `xgbmodel`) |
| Forward window | 1 trading day |
| Feature set | 174 features from `xgbmodel.data_loader.build_panel` |
| Folds | 212 walk-forward folds, 12w / 5d purge / 2w val / 2d embargo / 2w test |
| Sequence length (NN) | T = 30 days |
| Loss | pseudo-Huber (matches XGBoost's huber objective) |
| Output schema | `ts_code, trade_date, pred, target, fold` |
| Metrics | rank IC, ICIR, hit rate, Pearson, RMSE — same `xgbmodel.train.compute_metrics` |

Any deviation from this list breaks the apples-to-apples comparison.

## Wall time estimates (full 212-fold walk-forward)

| Engine | Hardware | Wall time |
|---|---|---|
| `xgb_default` (already trained) | GPU | ~52 min |
| `xgb_shallow` / `xgb_deep` / `xgb_strong_reg` | GPU | ~50–80 min each |
| `lightgbm` | CPU | ~45 min |
| `catboost` | CPU/GPU | ~70 min |
| `transformer_reg` | GPU | ~3–4 hr |
| `tft` | GPU | ~4–6 hr |
| `ensemble` (post-train) | CPU | <5 min |

Total to evaluate all six: **~12-15 hours**.

## Reading the results

Once everything has run, the predictions dashboard's **模型比较** section
shows:

- A sortable per-model metrics table (rank IC / ICIR / IC>0% / hit rate / Pearson / RMSE), highlighted winner with 🏆
- A horizontal bar chart of rank IC mean
- A correlation heatmap of model predictions — *low correlation (<0.7) means
  the models are seeing different signals and ensembling will help; high
  correlation (>0.9) means they're redundant.*

The two ensemble rows (`ensemble_mean`, `ensemble_rankavg`) typically beat
the best single model on rank IC if individual models are decorrelated.

## Architecture notes

### Pure Transformer (`transformer_reg.py`)
- Linear projection (F=174 → d_model=192)
- Sinusoidal positional encoding
- 3 × `TransformerEncoderLayer` (heads=6, d_ff=512, GeLU, pre-norm, causal mask)
- Mean-pool over T → Linear(d_model, 1) regression head
- AdamW + cosine LR + AMP FP16

### Pure TFT (`tft.py`)
Pure Lim 2019 — *no* sector cross-attention, *no* extended-feature merge:
- Variable Selection Network on each timestep (per-variable GRN + softmax weights)
- 1-layer LSTM encoder (d_model=128)
- Gated Linear Unit + LayerNorm residual
- Static enrichment GRN
- Multi-head temporal self-attention (heads=4, causal)
- Position-wise GRN
- Mean-pool → Linear(d_model, 1) regression head
- AdamW + cosine LR + AMP FP16

Both use pseudo-Huber loss to match XGBoost's `reg:pseudohubererror` objective.
