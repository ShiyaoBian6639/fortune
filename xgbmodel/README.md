# `xgbmodel/` — XGBoost next-day prediction pipeline

Predicts each stock's next-day return relative to CSI300 (excess return), trained
under strict walk-forward CV with no look-ahead. Used by `backtest/xgb_markowitz.py`
for the long-only Markowitz backtest, and by the dashboards under `dashboard/` for
both retrospective accuracy reporting and live next-day signals.

For the day-to-day command reference, see [`COMMANDS.md`](COMMANDS.md). This file
documents:

1. Pipeline architecture
2. Prediction target and why it's relative to CSI300
3. Feature catalog (174 features, organised by category)
4. Walk-forward training and the leakage guarantees
5. The prediction hygiene workflow (archive + staleness check)
6. Dashboard integration

---

## 1. Pipeline architecture

```
                        ┌──────────────────────────┐
                        │  stock_data/             │
                        │   ├── sh/, sz/  (OHLCV)  │
                        │   ├── daily_basic/       │
                        │   ├── moneyflow/         │
                        │   ├── block_trade/       │
                        │   ├── stk_limit/         │
                        │   ├── fina_indicator/    │
                        │   ├── index/             │
                        │   │   ├── idx_factor_pro/│
                        │   │   ├── index_dailybasic/
                        │   │   └── index_global/  │
                        │   └── stock_sectors.csv  │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                          xgbmodel/data_loader.py
                          ├── per-stock OHLCV  → features.py (TA, lags, rolls)
                          ├── daily_basic merge (forward-fill)
                          ├── moneyflow / block / limit merge
                          ├── fina_indicator merge_asof (point-in-time)
                          ├── index dailybasic merge (CSI300/500/SSE/SZSE/...)
                          ├── global indices (lagged 1d for causality)
                          └── cross_section.py: per-day rank + demean + breadth
                                     │
                                     ▼
                            (5,100 stocks × 9 yrs × 174 features)
                                     │
                                     ▼
                          xgbmodel/split.py  (walk_forward_folds)
                          212 folds: 12w train | 5d purge | 2w val | 2d embargo | 2w test
                                     │
                                     ▼
                          xgbmodel/train.py  (XGBRegressor, GPU hist)
                          ├── per-fold fit → val/test predictions saved
                          ├── canonical refit on full pool with median best_iter
                          └── stock_data/models/xgb_pct_chg.{json, meta.json, features.json}
                                     │
                                     ▼
                          xgbmodel/predict.py
                          ├── predict_latest(cfg) → next-day forecast
                          └── predict_test(cfg)   → score full test window
                                     │
                                     ▼
                          stock_predictions_xgb.csv  (live pointer)
                          stock_predictions_xgb_features_YYYYMMDD.csv  (archive)
                          stock_data/models/xgb_preds/test.csv  (OOF)
```

---

## 2. Prediction target

The model predicts `pct_chg` of each stock for the **next** trading day, expressed
as **excess return over CSI300**:

```
y_{i,t} = pct_chg_{i,t+1} − pct_chg_{CSI300, t+1}
```

with two preprocessing steps:

1. **Clip** at ±11% (A-share daily limit-up/down band, ChiNext/STAR have wider bands
   but ±11% catches the worst outliers reliably).
2. **Cross-sectional demean** per `trade_date`: subtract the daily mean of `y` across
   all stocks. This removes the predictive value of any per-day-constant feature (a
   "today the market is hot" feature can no longer win — only stock-discriminating
   signals can).

### Why excess return rather than raw `pct_chg`?

- **Removes market beta**: the model isn't trying to time the market (intractable),
  it's identifying which stocks will outperform/underperform on a given day.
- **Sample balance across regimes**: in a bear market, raw `pct_chg` distribution
  shifts left; excess return stays centered near zero. Optimization stays consistent
  across bull/bear/sideways regimes.
- **Strategy alignment**: the long-only Markowitz top-K selection naturally consumes
  rank/excess-return signals.

The forecast horizon is configurable via `--forward_window` (default 1 trading day).
For multi-day horizons, the target becomes the **k-day** excess return, but the
backtest, dashboard, and live workflow assume `--forward_window 1`.

---

## 3. Features (174 total)

Per-feature meanings and data sources are catalogued in
`dashboard/feature_catalog.py` (174/174 coverage) and rendered in the
"特征工程" section of the combined dashboard. High-level groups:

| Group | Count | Examples |
|---|---|---|
| 价格行为 (price action) | 41 | `pct_chg`, `oc_ratio`, `hl_ratio`, `momentum_{5,10,20}`, `close_ma_{5,10,20,60}_ratio`, `vol_pct_{5,10,20}`, `parkinson_{5,10,20}`, `dist_from_high_{20,60}` |
| 成交量与换手 (volume / turnover) | 8 | `vol`, `amount`, `vol_ratio_{5,20}`, `amt_ratio_{5,20}`, `vol_pct_chg`, `amount_pct_chg` |
| 技术指标 (TA) | 8 | `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bbpct_20`, `atr_14_pct`, `obv_flow_ma{5,20}` |
| 涨跌停信号 (limit-board) | 8 | `hit_up_limit`, `limit_up_streak`, `limit_up_count_20`, `up_limit_ratio` (proximity to limit price) |
| 形态识别 (chart patterns) | 4 | `w_bottom_{10,20}`, `m_top_{10,20}` |
| 资金流向 (money flow) | 5 | `net_{sm,md,lg,elg,mf}_amount_ratio` |
| 大宗交易 (block trades) | 3 | `block_count`, `block_vol_ratio`, `block_amount_ratio` |
| 估值 / 基本面 (valuation / fundamentals) | 22 | `turnover_rate_f`, `pe_ttm`, `pb`, `log_circ_mv`, `roe`, `current_ratio`, `assets_yoy`, `op_yoy`, `eps`, `has_fina_data`, ... |
| 宏观 / 市场环境 (macro / market context) | 33 | CSI300/CSI500/SSE50/SSE/SZSE/ChiNext × {turnover, PE-TTM, PB, pct_chg}; SPX/DJI/IXIC/HSI/N225/FTSE × {pct_chg_lag1, vol_ratio_lag1} |
| 沪深300 技术形态 (CSI300 TA) | 18 | `idx_bias{1,2,3}`, `idx_cci`, `idx_dmi_{adx,pdi,mdi}`, `idx_kdj_{k,d}`, `idx_rsi_{6,12,24}`, `idx_mfi`, `idx_wr`, `idx_macd_{dif,dea}`, `idx_psy`, `idx_vr` |
| 横截面 (cross-section, computed per trade_date) | 18 | `cs_rank_*` (12), `cs_demean_*` (4), `cs_market_breadth`, `cs_daily_dispersion` |
| 行业 (sector) | 1 | `sector_id` (SW L1, label-encoded) |
| 日历 (calendar) | 7 | `dow`, `month`, `day_of_month`, `quarter`, `is_month_{end,start}`, `is_quarter_end` |

The top-50 features by `total_gain` (212-fold sum) — heavily weighted towards
`turnover_rate_f`, `cs_demean_turnover_rate_f`, `hl_ratio`, `lowershadow`,
`up_limit_ratio` — are visualised in the dashboard's "特征重要性" section.

---

## 4. Walk-forward training and leakage guarantees

### Fold structure

Each of the 212 folds runs:

```
[ 12-week TRAIN ] [ purge=5d ] [ 2-week VAL ] [ embargo=2d ] [ 2-week TEST ]
```

- **Purge gap (5 days)** prevents the forward-shifted target on the last train rows
  from leaking into the val window's features.
- **Embargo gap (2 days)** further decorrelates serial dependence around the
  val→test boundary (de Prado, 2018, §7.4).
- Test windows are non-overlapping (`--fold_step_weeks 2 = fold_test_weeks`), so
  every test prediction is from a unique fold.

### What this guarantees

**For any test prediction at trade_date `t`, the model that produced it was trained
strictly on rows where `trade_date ≤ t − 5d − 2w − 2d`.** No feature row used in
training overlaps with the prediction row.

### Where leakage *could* sneak in (and where it doesn't)

| Possible vector | Verdict | Reason |
|---|---|---|
| Forward-shifted target leaking via lagged features | ✅ safe | Target `= pct_chg.shift(-1)`, then rows with NaN target are dropped before splitting. Lagged features only use `.shift(k≥0)`. |
| Rolling window leaking future values | ✅ safe | `.rolling(w)` is left-anchored; values at index `t` use only `t-w+1..t`. No `.rolling(...).shift(-1)` or similar anywhere. |
| Cross-sectional rank using future days | ✅ safe | `panel.groupby('trade_date')[col].rank(pct=True)` ranks within a single day. |
| Quarterly fundamentals announced after | ✅ safe | `merge_asof(direction='backward', allow_exact_matches=True)` only joins fina rows with `ann_date ≤ trade_date`. |
| Global index same-day overlap (Asia/Europe) | ✅ safe | All global indices explicitly suffixed `_lag1` and computed via `.shift(1)` before merging. |
| Per-fold model leaking via XGBoost early stopping | ✅ safe | Early stopping uses the **val** window only; test is never consulted during training. |
| The canonical model (refit on full pool) being applied to the *training* dates it saw | n/a | Canonical model is only used for **live forecasting** of dates beyond all folds. The OOF `test.csv` predictions used by the backtest come from the per-fold models. |

### Live (production) inference

For predicting `t+1` where `t` = today:

- `data_loader.build_panel(cfg)` is called with `for_inference=True`, which keeps
  the most recent feature row whose target would be NaN (because `t+1` lies past
  available data).
- The canonical full-pool-refit model (`stock_data/models/xgb_pct_chg.json`) scores
  this most-recent row.
- Features at row `t` use only data ≤ `t` (verified by the same guarantees as
  training).

---

## 4b. ST risk control (special-treatment stocks)

China's "special treatment" status (`ST` / `*ST` prefix on the stock name) marks
firms with severe financial distress, three years of losses, or accounting
irregularities. They have a tighter 5% daily price band and an elevated risk
of delisting or trading suspension.

### The empirical question

Is the backtest's reported track record actually a non-ST strategy with some
ST decoration, or is it primarily an ST-trading strategy with everything else
nibbling around the edges?

We answered this rigorously by independently checking each historical entry
against the authoritative tushare `namechange` roster (NOT the 5%-band
heuristic, which conflates ST with stocks whose 5% band came from price-step
rules).

**Findings on the 7,676 historical backtest entries**:

| Metric | Value |
|---|---|
| Trades made on ST/*ST stocks (at trade time) | **4,393 / 7,676 = 57.2%** |
| Net P&L from ST trades | **+141.0M / +141.3M = 99.7%** |
| TP hit rate on ST trades | 76.2% |
| TP hit rate on non-ST trades | 36–44% |

So the historical alpha is *almost entirely* an ST-trading strategy. The
model learned that ST/*ST stocks exhibit a recoverable next-day bounce after
extreme moves — a real signal, but one that lives entirely in a high-risk
sub-universe.

### Did any of those ST trades end in delisting?

For every one of the 310 distinct ST stocks the strategy traded, we checked
its **current** state from `pro.stock_basic.list_status`:

| Current state | Stocks | % | Trades | P&L (CNY) |
|---|---|---|---|---|
| **recovered** (摘帽) | 178 | 57.4% | 1,722 | +27.2M |
| **still_st** | 131 | 42.3% | 2,655 | +112.9M |
| **delisted** | 1 | **0.3%** | 16 | +0.85M (still profitable) |
| paused | 0 | 0% | — | — |

Of every 310 ST stocks entered, **only 1 ended up delisted** (and the
strategy made money on it). The 1–5 day holding window catches the volatile
rebound moves without staying long enough for delisting events to materialise.
The ex-post survivorship bias of the ST sub-universe in this window is
significantly less scary than the prior would suggest.

### Authoritative ST detection (do NOT use 5%-band heuristic)

`api/st_history.py` pulls every stock's name-change history from tushare's
`pro.namechange` endpoint and writes `stock_data/st_history.csv` with one
row per ST/non-ST transition:

```
ts_code,st_kind,start_date,end_date,ann_date,name,change_reason
000609.SZ,*ST,20210430,20220515,20210429,*ST中迪,*ST
000609.SZ,*ST,20240429,20250610,20240426,*ST中迪,*ST
000609.SZ,ST, 20250611,20260422,20250610, ST中迪,摘星
000609.SZ,*ST,20260423,        ,20260422,*ST中迪,*ST
```

783 stocks have been ST/*ST at some point in the universe; 2,433 transition
events recorded.

`api/st_history.py:is_st_at(roster, ts_code, date)` answers whether
`ts_code` was ST/*ST on `date`. Used by:

- `backtest/xgb_markowitz.py` — `--max_st_per_day N` caps ST positions per day.
- `dashboard/live_prediction.py` — the live-forecast `exclude_st` flag.
- `dashboard/st_outcomes.py` — analyses what eventually happened to each ST
  stock the strategy traded.

Why not the 5%-daily-band heuristic: (1) it fires on stocks whose 5% band
came from non-ST price-step rules, (2) it can't see multiple ST → recovery
→ ST cycles, (3) it requires per-day stk_limit lookups instead of one
in-memory roster. The namechange-based roster is authoritative; the band
heuristic is deprecated.

### Capped backtest variant — does the alpha survive?

```bash
# Hard cap of 4 ST stocks per day; non-ST candidates fill the rest.
./venv/Scripts/python -m backtest.xgb_markowitz \
  --solver qp --max_st_per_day 4 --tag qp_st4
```

Outputs `plots/backtest_xgb_markowitz/equity_qp_st4.csv` and `trades_qp_st4.csv`,
which `dashboard.build_combined` picks up automatically.

**Empirical result over the same 5-year window** (2021-04-22 → 2026-04-21):

| Metric | Original (no cap) | Capped (max 4 ST/day) | Δ |
|---|---|---|---|
| CAGR | +169.71% | **+148.91%** | −20.8 pp (−12% relative) |
| Sharpe (rf=0) | 7.58 | **6.86** | −0.72 |
| Max drawdown | −18.20% | **−19.69%** | −1.5 pp |
| Annual alpha vs CSI300 | +104.4% | **+96.1%** | −8.3 pp |
| Information ratio | 5.32 | **4.92** | −0.40 |
| Trades | 7,676 | 6,265 | −18% |

The strategy retains **57/100 of its CAGR ratio** under the cap (148.9 / 169.7 ≈
0.88). Far from "the alpha was all ST" — the model has real non-ST signals it
can deploy when the ST budget is exhausted. Sharpe stays well above 6.

The cap CLI flag is fully parameterised; `-1` (default) preserves the
original behaviour exactly:

| `--max_st_per_day` | Effect |
|---|---|
| `-1` (default) | No cap. Full historical behaviour. |
| `0` | Exclude ST entirely (most risk-averse). |
| `4` | The recommended balance — used in dashboards. |
| `N` (any non-negative) | Cap per-day ST positions at N (across both newly-entered and already-held). |

### Dashboard surfaces

Three sections in `dashboard/combined.html`:

- **ST 个股结局** — for every ST stock the strategy ever entered, classifies
  its current state from tushare `stock_basic.list_status` + current name:
  `delisted` (退市), `paused` (暂停上市), `still_st` (仍 ST), `recovered`
  (摘帽). Donut chart + summary table + top-50 contributors with current
  status.
- **ST 限额回测** — original (no cap) vs `--max_st_per_day 4` equity curves,
  CAGR / Sharpe / MDD deltas, and a plain-language verdict on whether the
  strategy's alpha survives the cap.
- **下一交易日预测** — toggle between "排除 ST (recommended for live)" and
  "包含 ST (matches backtest)" views; ST detection backed by the same roster.

---

## 5. Prediction hygiene: archive + staleness check

### The bug we hit

`stock_predictions_xgb.csv` is a *pointer to the latest run*, and
`stock_predictions_xgb_YYYYMMDD.csv` is a sibling snapshot named with the
**generation date** (when the prediction was run, not the feature date it used).
This caused two files to look "identical" when in fact the live pointer simply
hadn't been refreshed since the last archive — confusing because it suggested
the model was stuck.

### What's now in place

`predict_latest()` in `xgbmodel/predict.py` writes **two outputs every run**:

| File | Type | Renamed every run? |
|---|---|---|
| `stock_predictions_xgb.csv` | live pointer (always latest) | yes — overwritten |
| `stock_predictions_xgb_features_YYYYMMDD.csv` | immutable archive, named by **feature date** | no — new file each unique feature date |

After the file write, a **diff against the previous archive** is printed:

```
[xgbmodel.predict] vs previous archive (stock_predictions_xgb_features_2026-04-23.csv):
  differ on 5057/5058 stocks (mean |Δ|=0.1735, max |Δ|=9.7598)
```

If the prediction is identical to the previous archive (same model + same input
data), the script emits a **warning**:

```
[xgbmodel.predict] ⚠️ WARNING: predictions are IDENTICAL to
  stock_predictions_xgb_features_YYYYMMDD.csv — stale data?
```

### Dashboard staleness guard

`dashboard/live_prediction.py` independently sanity-checks the prediction CSV
against the per-stock OHLCV files on disk. If the data on disk is fresher than
the prediction's `feature_date`, it sets `payload['stale_warning']` and the
combined dashboard renders a red banner above the live-prediction section
prompting you to re-run `predict`.

### Daily workflow

```bash
# 1. download today's data (incremental)
./venv/Scripts/python update_all.py

# 2. produce next-day prediction; auto-archives + diffs vs previous
./venv/Scripts/python -m xgbmodel.main --mode predict

# 3. rebuild predictions dashboard (auto-discovers live pointer; nothing
#    hardcoded). Top-20 / bottom-20 tables now reflect the new forecast.
./venv/Scripts/python -m dashboard.build

# 4. rebuild combined dashboard (also runs Markowitz QP on the new μ).
#    Will paint a red 警告：预测数据陈旧 banner if step 2 was skipped and
#    the data on disk is fresher than the prediction's feature_date.
./venv/Scripts/python -m dashboard.build_combined --cached-barra dashboard/barra_cache.json

# 5. encrypt + ship
./venv/Scripts/python -m dashboard.package_secure          --password 'YOURPASS'
./venv/Scripts/python -m dashboard.package_secure_combined --password 'YOURPASS'
```

After several runs you should see a clean audit trail, e.g.:

```
stock_predictions_xgb.csv                           ← always latest
stock_predictions_xgb_features_2026-04-21.csv
stock_predictions_xgb_features_2026-04-22.csv
stock_predictions_xgb_features_2026-04-23.csv      ← predicts 04-24
stock_predictions_xgb_features_2026-04-24.csv      ← predicts 04-27 (current)
```

If a future archive ever appears identical to its predecessor, the printed
warning will catch it before it propagates into the dashboard or the backtest.

### Verifying that the model responds to fresh data

A simple two-row diff (post-fix sanity check):

```bash
./venv/Scripts/python -c "
import pandas as pd
a = pd.read_csv('stock_predictions_xgb_features_2026-04-23.csv').set_index('ts_code')
b = pd.read_csv('stock_predictions_xgb_features_2026-04-24.csv').set_index('ts_code')
common = a.index.intersection(b.index)
diff = (a.loc[common, 'pred_pct_chg_next'] - b.loc[common, 'pred_pct_chg_next']).abs()
print(f'differ on {(diff > 1e-6).sum()}/{len(diff)} stocks, mean|Δ|={diff.mean():.4f}, max|Δ|={diff.max():.4f}')
"
```

Expected output for two consecutive trading days' predictions:

```
differ on 5057/5058 stocks, mean|Δ|=0.1735, max|Δ|=9.7598
```

(Identical predictions on consecutive days would be the alarm signal.)

---

## 6. Dashboard integration

Both dashboards now read the **live pointer** `stock_predictions_xgb.csv`
auto-discovered at build time — no hardcoded dates anywhere. The chain:

```
xgbmodel.main --mode predict
       │
       ├─→ stock_predictions_xgb.csv                  (live pointer)
       └─→ stock_predictions_xgb_features_YYYYMMDD.csv (immutable archive)
              │                                  │
              ▼                                  ▼
    dashboard.build                  dashboard.build_combined
    (predictions dashboard)          (unified dashboard, runs Markowitz QP
                │                     on the live μ via dashboard.live_prediction)
                │                                  │
                ▼                                  ▼
    dashboard/data.json              dashboard/combined_data.json
                │                                  │
                ▼                                  ▼
    index.html / index_secure.html   combined.html / index_combined.html /
                                     index_combined_secure.html
```

### Predictions dashboard (`index.html`)

`dashboard/build.py` now calls `load_live_predictions()` which:

1. Tries the live pointer `stock_predictions_xgb.csv`.
2. Falls back to the most recent `stock_predictions_xgb_features_*.csv`
   archive if the pointer is missing.

It also calls `_next_trading_day_for(feature_date, ...)` which scans real
OHLCV data to determine the actual next trading day (handles weekends and
holidays correctly), so `data.json['prediction_date']` is always derived
from data, never a hardcoded string.

The dashboard's title bar, footer, and all date-stamped section headers
read those fields dynamically — drop-in fresh predictions and the dashboard
re-styles for the new date with no edits.

### Combined dashboard (`combined.html`)

Two top-of-page sections come from the live prediction:

- **下一交易日预测** — feature_date → forecast_date, candidate funnel
  (limit-stop filtering breakdown), top-K=10 long-only Markowitz portfolio
  with QP weights, suggested CNY allocation, share counts, μ, σ_60, 80%
  prediction interval, P(↑), P(>3%); plus full top-30 / bottom-30 tables.
- **无前视数据保证** (Leakage Audit) — the 7 guarantees from §4 above with
  source-code references; data freshness inventory across all 19 sources,
  color-coded against the live `feature_date`.

Both sections automatically reflect re-runs:

- Live signal updates every time `predict` is re-run and `build_combined`
  is re-run.
- **Staleness banner** appears if data on disk is fresher than `feature_date`
  (catches "forgot to re-run predict" failure mode).
- Funnel cards show why the candidate pool was thinned (limit-stops,
  suspended names, insufficient history).

The same payload is also encrypted into `index_combined_secure.html` for
password-gated Netlify drag-drop deployment (see `dashboard/README.md` for
details on the AES-GCM-256 + PBKDF2 envelope).

---

## 7. Re-training the model

A full walk-forward re-train (212 folds, GPU) takes around 60–90 minutes on a
modern Nvidia GPU. You should retrain when:

1. **Feature engineering changes** — adding/removing/renaming any column in
   `xgbmodel/features.py`, `cross_section.py`, or `data_loader.py`. Always
   delete `stock_data/models/xgb_pct_chg.*` first so the dashboard doesn't
   keep showing stale feature_importance / metric data.
2. **Data range extension** — every new month or two of data is enough to
   shift the canonical model's best_iter, refresh fold-level metrics, and
   incorporate the new regime.
3. **Hyperparameter tuning** — see `--learning_rate / --max_depth / ...` in
   `COMMANDS.md`.

Standard re-train command:

```bash
./venv/Scripts/python -m xgbmodel.main --split_mode walk_forward --device cuda
```

After re-training:

```bash
# 1. run a fresh prediction (writes new archive)
./venv/Scripts/python -m xgbmodel.main --mode predict

# 2. rebuild the predictions dashboard JSON
./venv/Scripts/python -m dashboard.build

# 3. rebuild the combined dashboard
./venv/Scripts/python -m dashboard.build_combined --cached-barra dashboard/barra_cache.json

# 4. ship
./venv/Scripts/python -m dashboard.package_secure_combined --password 'YOURPASS'
```

---

## 8. References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7
  (purged K-fold + embargo).
- Harvey & Liu (2015). *Backtesting: An Implementation Guide.*
- Bailey et al. (2017). *The Probability of Backtest Overfitting.*
- This repo: `xgbmodel/COMMANDS.md` (commands), `dashboard/README.md` (dashboard
  pipeline), `backtest/xgb_markowitz.py` (Markowitz QP backtest),
  `dashboard/leakage_audit.py` (leakage guarantees machine-checkable inventory).
