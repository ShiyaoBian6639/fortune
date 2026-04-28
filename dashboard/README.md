# Dashboards

Three static-HTML dashboards generated from this repo, all using Plotly via CDN
and rendering correctly when dragged into Netlify (or served from any static
host). They share styling but each focuses on a different artifact:

| File entrypoint | Purpose | Builder | Encrypted variant |
|---|---|---|---|
| `index.html`              | XGBoost prediction dashboard (live signals + OOS accuracy) | `dashboard.build`           | `dashboard.package_secure`           → `index_secure.html` |
| `backtest.html`           | XGB Markowitz long-only backtest (5-year, QP solver)      | `dashboard.build_backtest`  | `dashboard.package_secure_backtest`  → `index_backtest_secure.html` |
| `combined.html`           | Unified: predictions + features + backtest + Barra        | `dashboard.build_combined`  | `dashboard.package_secure_combined`  → `index_combined_secure.html` |

> **Drop on Netlify**: drag any single `index_*.html` file (the encrypted ones
> include a Chinese login overlay; the unencrypted `index_*.html` files embed
> the JSON for offline use). For folder drops, dropping the whole `dashboard/`
> directory works too.

---

## 1. `index.html` — XGBoost prediction dashboard

What it shows:

1. **Model overview** — target, feature count, walk-forward config, train time.
2. **Prediction accuracy (out-of-sample)** — MAE, MSE, RMSE, Pearson / Spearman
   IC, directional hit rate, confident hit rate, daily rank IC series with
   20-day rolling mean, predicted-vs-actual scatter.
3. **Decile analysis** — per-decile mean realized return + cumulative
   long-short spread + Sharpe.
4. **Performance by group** — province, board (Main / ChiNext / STAR),
   exchange. Each row: Rank IC, Pearson IC, hit rate, MAE, visual heatmap bar.
5. **Global feature importance** — top 30 XGBoost features by total gain.
6. **Live signal: Top 20 predicted outperformers** (and Bottom 20) for the
   next trading day. Each stock card shows:
   - Point estimate, 80% prediction interval, probability decomposition.
   - Recent price sparkline (60 trading days).
   - Factor snapshot (5-day return, 20-day vol, close vs MA20, volume ratio,
     H-L range, open gap, up-day count).
   - Human-readable reason tags explaining *why* the model scored it high.
7. **Prediction distribution** — histogram + count of up/down/strong signals.
8. **Full searchable table** — all live predictions with sort + filter.

### Data sources

- `stock_predictions_xgb.csv` (repo root) — **live pointer**, always updated
  to the most recent `xgbmodel.main --mode predict` run. The dashboard now
  reads this dynamically; `feature_date` and `prediction_date` are derived
  from the data, never hardcoded.
- `stock_predictions_xgb_features_YYYYMMDD.csv` — immutable archives written
  on every `predict` run (named by feature date). Used as a fallback if the
  live pointer is missing.
- `stock_data/models/xgb_preds/test.csv` — 6.25M OOS walk-forward predictions
  from 212 non-overlapping folds. Used for all accuracy / group metrics.
- `stock_data/models/xgb_pct_chg.meta.json` — model metadata + global feature
  importance.
- `stock_sectors.csv` — province / board / exchange metadata for grouping.
- `stock_data/{sh,sz}/*.csv` — raw OHLCV, used to derive the actual next
  trading day after `feature_date` (handles weekends/holidays correctly), and
  for the 60-day sparkline + factor snapshot on each top/bottom stock card.

### Rebuild

```bash
# 1. produce a fresh prediction (writes new archive + auto-diff vs previous)
./venv/Scripts/python -m xgbmodel.main --mode predict

# 2. rebuild dashboard JSON
./venv/Scripts/python -m dashboard.build
```

Rewrites `dashboard/data.json` from the sources above (~30 s).

### Notes

- Accuracy metrics live on the **OOS walk-forward test** set because the most
  recent forecast date is by definition not yet realised; OOS test.csv gives
  9+ years of clean (pred, target) pairs covering the full backtest window.
- "Reasoning" is not full SHAP — it's a rule-based factor snapshot chosen to
  mirror the XGBoost top-10 global features (`turnover_rate_f`, `hl_ratio`,
  `overnight_gap`, `close_ma_20_ratio`, etc.).
- The page title and footer no longer hardcode any specific trading date —
  the badge under the page title shows `预测日期 {prediction_date}` populated
  from the JSON.
- All Chinese text (stock names, provinces) is UTF-8 in the JSON. Windows
  terminals may render garbled but browsers display correctly.

---

## 2. `backtest.html` — XGB Markowitz long-only backtest

A Chinese-language dashboard focused on the 5-year live backtest of the
top-K=10 long-only Markowitz strategy with full-Σ QP solver.

Sections:

- **策略概览** — final NAV, CAGR, Sharpe, MDD, Calmar, IR, alpha, β.
- **净值曲线与回撤** — strategy NAV (log) vs CSI300 buy-and-hold + drawdown +
  position-count / cash time series.
- **风险特征 (按时间轴)** — per-trade return scatter and per-trade hold-days
  scatter, both keyed on the **exit date** (datetime x-axis), color-coded by
  exit reason (止盈 / 止损 / 强平).
- **卖出原因** — donut chart + table explaining the trigger condition for each
  reason (TP, SL, Horizon).
- **原因 × 信号强度分组** — stacked bars of TP/SL/Horizon by μ-decile with
  mean realised-return overlay (model-confidence calibration diagnostic).
- **个股贡献** — searchable, scrollable table of all 2,170 stocks ever traded,
  ranked by total realised P&L. Includes Chinese name.
- **个股买卖时点 (2017 → 至今)** — for each of the **top-20 stocks by P&L**,
  a long-horizon (2017 onward) close-price chart with **buy/sell markers** at
  every entry/exit. Markers are color-coded by reason; hovering shows price,
  return, P&L, hold-days, and matched entry/exit date. Switch stocks via the
  dropdown.
- **完整交易明细** — paginated, sortable, searchable table of all 7,676 trades.

### Data sources

- `plots/backtest_xgb_markowitz/equity_qp.csv` — daily NAV, cash, invested.
- `plots/backtest_xgb_markowitz/trades_qp.csv` — per-trade buy/sell records.
- `plots/backtest_xgb_markowitz/metrics_qp.txt` — verbatim metrics dump.
- `stock_data/index/idx_factor_pro/000300_SH.csv` — CSI300 benchmark.
- `stock_sectors.csv` — Chinese names.
- `stock_data/{sh,sz}/*.csv` — raw OHLCV for 2017+ price charts on the top-20.

### Rebuild

```bash
# 1. produce or refresh the backtest run
./venv/Scripts/python -m backtest.xgb_markowitz --solver qp --tag qp

# 2. assemble the dashboard JSON + single-file Netlify HTML
./venv/Scripts/python -m dashboard.build_backtest --tag qp
```

Outputs:
- `dashboard/backtest_data.json` (~4 MB) — fetched by the dev `backtest.html`.
- `dashboard/index_backtest.html` (~4 MB) — JSON inlined as
  `window.BACKTEST_DATA = {…}`; works as a stand-alone single-file drop.

### Notes — QP solver

The backtest module `backtest/xgb_markowitz.py` accepts `--solver {diag,qp}`:

- **`diag`** (default, closed-form): Σ assumed diagonal, weights ∝ μᵢ / σᵢ².
- **`qp`**: Σ estimated as **Ledoit-Wolf-shrunk** sample covariance over a
  60-day rolling panel of realised pct_chg; long-only mean-variance QP solved
  with **SciPy SLSQP**:
  ```
  minimize    0.5·λ·wᵀΣw − μᵀw
  subject to  Σ wᵢ = 1,  0 ≤ wᵢ ≤ 1
  ```
  No external solver required (CPLEX/IPOPT not needed). Falls back to the
  diagonal closed form on solver failure or all-non-positive μ.

---

## 3. `combined.html` — Unified prediction + backtest + Barra dashboard

Single Chinese-language page that documents the **full pipeline**: feature
engineering → model → predictions → backtest → risk attribution. Useful for
sharing one URL that covers everything end-to-end.

14 sections:

| # | Anchor | Content |
|---|---|---|
| 1 | **下一交易日预测** | Live forecast for the next trading day: feature_date → forecast_date, candidate-funnel cards (limit-stop / no-data / insufficient-history breakdown), top-K=10 long-only Markowitz portfolio with QP weights, μ, σ_60, 80% PI, P(↑), P(>3%), CNY allocation, share counts (100-lot rounded). Also top-30 / bottom-30 prediction tables and a stale-data banner that fires if `feature_date` lags the data on disk. **ST detection now uses the authoritative `stock_data/st_history.csv` roster** (downloaded via tushare `namechange`), not the deprecated 5%-band heuristic. |
| 2 | **无前视数据保证** | 7 leakage guarantees with source-code references + 19-row data-source freshness inventory color-coded against the live `feature_date`. |
| 2b | **ST 个股结局分析** | For every ST stock the strategy ever entered, classifies current status as delisted / paused / still_st / recovered (摘帽). Donut chart + table + top-50 contributors. Built by `dashboard/st_outcomes.py`. |
| 2c | **ST 限额回测对照** | Original (no cap) vs `--max_st_per_day 4` capped backtest, side-by-side equity curves with CAGR/Sharpe/MDD deltas. Built when `equity_qp_st4.csv` and `trades_qp_st4.csv` exist under `plots/backtest_xgb_markowitz/`. |
| 3 | 模型概览 | 174 features / 212 folds / training cost / XGBoost hyperparameters dump |
| 4 | 预测目标 | y = pct_chg − csi300_pct_chg formula and the cross-sectional demeaning rationale |
| 5 | 特征工程 | All 174 features by category; click-to-filter chips + fuzzy search; each row shows English name, group, **Chinese meaning**, **data source** |
| 6 | 特征重要性 | Top 50 features by total_gain (horizontal bar, colored by category, hover shows Chinese meaning) |
| 7 | 预测精度 | Pearson IC / Spearman IC / daily Rank IC / ICIR / hit rate / decile L-S Sharpe + daily-IC time series + decile bar |
| 8 | 回测概览 | Final NAV / CAGR / Sharpe / MDD / Alpha / Beta / IR + side-by-side strategy-vs-CSI300 + trade quality cards |
| 9 | 净值曲线 | Equity (log) + drawdown + position count / cash |
| 10 | **Barra 风险归因** | α + 5 factor cumulative contribution cards + factor exposures over time + cumulative attribution decomposition |
| 11 | 卖出原因 | Donut + table |
| 12 | 个股买卖时点 | Top-20 by P&L; long-horizon price chart + buy/sell markers |
| 13 | 个股贡献 | All 2,170 traded stocks |
| 14 | 完整交易明细 | All 7,676 trades, paginated/sortable/searchable |

### Live next-day forecast (sections 1–2)

Built by `dashboard/live_prediction.py`:

1. Reads the **live pointer** `stock_predictions_xgb.csv` (set by
   `xgbmodel.main --mode predict`). Falls back to the most recent
   `stock_predictions_xgb_features_*.csv` archive if the pointer is missing.
2. Pulls the top-200 candidates by μ, then applies the same filters as the
   backtest: limit-band check (±10% main-board / ±20% ChiNext-STAR), data
   availability, σ_60 history requirement.
3. For the top-K=10 survivors, builds a 60-day rolling pct_chg panel
   **strictly before** `feature_date`, estimates Σ via Ledoit-Wolf shrinkage,
   and runs the long-only Markowitz QP via SciPy SLSQP. Same code path as
   the backtest — see §1.4 of `xgbmodel/README.md` for the leakage proof.
4. Records the **filter funnel** (pool size → limit-stops → no-data →
   insufficient-history → pass) so the dashboard can show users why the
   candidate count was thinned.
5. Independently sanity-checks the prediction CSV against per-stock OHLCV
   files on disk; sets `payload['stale_warning']` (rendered as a red banner)
   if data on disk is fresher than `feature_date`. This catches the
   "predictions never refreshed" failure mode end-to-end.

### Barra-style risk attribution

5 style factors built from `stock_data/daily_basic` + per-stock pct_chg:

| Factor | Formula | 含义 |
|---|---|---|
| `SIZE` | log(1+circ_mv) | 流通市值 |
| `VALUE` | 1/pb | 价值 |
| `MOMENTUM` | 63-day cumulative return (lagged 1d) | 中期动量 |
| `VOLATILITY` | 60-day std(pct_chg) | 波动率 |
| `LIQUIDITY` | 60-day mean turnover_rate_f | 流动性 |

Pipeline:
1. **Per-day cross-sectional z-score** with 1%/99% winsorisation.
2. **Daily factor return** = top-quintile mean pct_chg − bottom-quintile mean
   (long-short factor portfolio).
3. **Strategy daily exposure** = mean z-score of held stocks
   (top-K=10 ≈ equal-weight after the QP).
4. **Daily attribution**:
   ```
   contrib_k,t  =  exposure_k,t-1  ·  factor_return_k,t  ·  (invested_t / NAV_t)
   ```
   The `invested/NAV` factor accounts for cash drag.
5. **Residual α** = strategy daily return − Σ_k contrib_k.

Result for the QP backtest:

- **Cumulative α (residual stock-selection)** = **+8033 %**  ← bulk of return
- 5 factors total cumulative contribution ≈ +24 %
- Mean exposures: small-cap-tilted (SIZE −0.20), low-liquidity-tilted
  (LIQUIDITY −0.21), momentum-tilted (+0.22), VALUE / VOLATILITY ~ neutral

Interpretation: the strategy's outperformance is overwhelmingly **stock
selection**, not a systematic style bet. The small-cap and low-liquidity
tilts are sample-driven artifacts of *ST* names dominating top-P&L holdings.

### Data sources

In addition to the prediction + backtest sources above:

- `stock_predictions_xgb.csv` (repo root) — **live pointer**, source of truth
  for the live forecast block. Read by both `dashboard.build` and
  `dashboard.live_prediction`. Refreshed by `xgbmodel.main --mode predict`.
- `stock_predictions_xgb_features_YYYYMMDD.csv` — immutable per-feature-date
  archives written on every `predict` run, used for the staleness diagnostic
  inside `xgbmodel.predict`.
- `stock_data/daily_basic/daily_basic_YYYYMMDD.csv` — daily valuation panel
  (used for SIZE, VALUE, LIQUIDITY exposures and for the live close-price
  reference of the suggested portfolio).
- `stock_data/sh/`, `stock_data/sz/` — per-stock OHLCV for MOMENTUM and
  VOLATILITY rolling computations and for next-trading-day determination.
- `stock_data/models/xgb_pct_chg.meta.json` — XGBoost hyperparameters,
  walk-forward config, full top-50 feature importance.
- `stock_data/models/xgb_pct_chg.features.json` — canonical 174-feature list.
- `dashboard/feature_catalog.py` — hand-curated mapping of every feature →
  (group, Chinese meaning, data source). 174/174 coverage.
- `dashboard/leakage_audit.py` — 7 static leakage guarantees + dynamic data
  freshness inventory, both rendered in the "无前视数据保证" section.
- `dashboard/live_prediction.py` — the live-forecast / portfolio block
  (Markowitz QP applied to today's predictions).
- `dashboard/data.json` — already-built prediction-side payload (reused).
- `dashboard/backtest_data.json` — already-built backtest-side payload (reused).

### Rebuild

```bash
# 0a. (after data refresh) regenerate the live prediction.
#     Writes stock_predictions_xgb.csv + stock_predictions_xgb_features_YYYYMMDD.csv,
#     and prints a diff against the previous archive.
./venv/Scripts/python -m xgbmodel.main --mode predict

# 0b. (occasional, e.g. weekly) refresh the authoritative ST roster.
#     ~5500 stocks × namechange API call, ~25 min total.
#     Writes stock_data/st_history.csv (used by backtest, live, st_outcomes).
./venv/Scripts/python -m api.st_history --download

# 0c. (occasional) re-run the capped backtest if you've changed the ST cap or
#     refreshed the OOF predictions in xgb_preds/test.csv.
./venv/Scripts/python -m backtest.xgb_markowitz \
  --solver qp --max_st_per_day 4 --tag qp_st4

# 1. fresh dashboard.build (predictions) and dashboard.build_backtest (backtest).
#    dashboard.build now auto-discovers the live pointer; nothing is hardcoded.
./venv/Scripts/python -m dashboard.build
./venv/Scripts/python -m dashboard.build_backtest --tag qp

# 2. unify them + run Barra attribution + run the live-forecast Markowitz QP.
#    The combined dashboard's "下一交易日预测" section is built here.
./venv/Scripts/python -m dashboard.build_combined

# (Optional) skip the ~90 s Barra recompute on a re-build
./venv/Scripts/python -m dashboard.build_combined --cached-barra dashboard/barra_cache.json

# (Optional) skip Barra entirely
./venv/Scripts/python -m dashboard.build_combined --skip-barra

# (Optional) skip the live-forecast block (e.g. when no fresh predict has been run)
./venv/Scripts/python -m dashboard.build_combined --skip-live
```

Outputs:
- `dashboard/combined_data.json` (~5 MB).
- `dashboard/index_combined.html` (~5 MB) — JSON inlined as
  `window.COMBINED_DATA = {…}`; stand-alone single-file Netlify drop.
- `dashboard/barra_cache.json` — saved Barra payload.

If you skip step 0, the dashboards will still build, but the "下一交易日预测"
section paints a red **stale-data** banner pointing at the freshness mismatch
between disk and `feature_date` (see `dashboard/live_prediction.py`).

---

## 4. Password protection (AES-GCM-256 + PBKDF2)

Each dashboard has a sibling packager that produces a single password-locked
HTML file. The encryption envelope:

- Plaintext = `gzip(level=9, dashboard_data.json)`.
- Cipher = **AES-GCM-256** with random 12-byte IV.
- Key = **PBKDF2-HMAC-SHA256(password, random-16-byte-salt, 200,000 iter)**.
- Layout = `salt || iv || ciphertext || GCM_tag`, base64-encoded into a
  `<script type="application/octet-stream">` block.
- Decryption runs **entirely in-browser** via Web Crypto API + native
  `DecompressionStream('gzip')`; no server.
- A login overlay covers the page until the correct password is entered;
  on success it resolves the deferred Promise the dashboard's `getData()` is
  already awaiting (no source-patching of the dashboard HTML required).

Build commands:

```bash
# Prediction dashboard
./venv/Scripts/python -m dashboard.package_secure          --password 'YOURPASS'
# → dashboard/index_secure.html

# Backtest dashboard
./venv/Scripts/python -m dashboard.package_secure_backtest --password 'YOURPASS'
# → dashboard/index_backtest_secure.html

# Combined dashboard
./venv/Scripts/python -m dashboard.package_secure_combined --password 'YOURPASS'
# → dashboard/index_combined_secure.html
```

Each command also accepts `DASH_PW` env var or prompts interactively if no
flag is given. ⚠️ Never commit a password to git and never bake it into a
checked-in build script.

### Deploy

1. Open <https://app.netlify.com/drop> (sign in, free tier).
2. Drag the desired `index_*.html` (or `index_*_secure.html`) file onto the
   page.
3. Netlify gives you a URL like `https://<random>.netlify.app/`.

If you accidentally drop the **non-secure** `index_*.html` to a previously
password-protected URL, the site becomes public again. Always ship the
`*_secure.html` variants in production.

---

## 5. File index (under `dashboard/`)

```
dashboard/
├── README.md                          ← this file
│
├── build.py                           ← prediction dashboard builder
├── build_backtest.py                  ← backtest dashboard builder
├── build_combined.py                  ← unified-dashboard builder (orchestrator)
├── feature_catalog.py                 ← 174 features → {group, meaning, source}
├── barra_attribution.py               ← Barra-style 5-factor risk attribution
├── leakage_audit.py                   ← 7 leakage guarantees + freshness inventory
├── live_prediction.py                 ← live next-day Markowitz QP + filter funnel
├── st_outcomes.py                     ← what eventually happened to each ST stock traded
│
├── package_secure.py                  ← AES-GCM packager (predictions)
├── package_secure_backtest.py         ← AES-GCM packager (backtest)
├── package_secure_combined.py         ← AES-GCM packager (combined)
│
├── index.html         + data.json     ← prediction dashboard (dev pair)
├── index_secure.html                  ← prediction, password-locked single file
│
├── backtest.html      + backtest_data.json   ← backtest dashboard (dev pair)
├── index_backtest.html                       ← backtest, JSON inlined
├── index_backtest_secure.html                ← backtest, password-locked
│
├── combined.html      + combined_data.json   ← unified dashboard (dev pair)
├── index_combined.html                       ← unified, JSON inlined
├── index_combined_secure.html                ← unified, password-locked
│
└── barra_cache.json                   ← cached Barra attribution payload
```

## 6. Daily refresh — end-to-end

After every market close, run this canonical sequence to keep both dashboards
in sync with the latest prediction. Each step **automatically picks up the
output of the previous step** — no hardcoded dates anywhere.

```bash
# 1. download the day's incremental data (OHLCV, daily_basic, moneyflow, ...)
./venv/Scripts/python update_all.py

# 1b. (occasional, e.g. weekly) refresh the authoritative ST roster.
#     ~25 min. Source for the live ST filter, the capped backtest, and
#     the ST 个股结局 dashboard section.
./venv/Scripts/python -m api.st_history --download

# 2. score every stock for the next trading day.
#    Writes:
#      - stock_predictions_xgb.csv                            (live pointer)
#      - stock_predictions_xgb_features_YYYYMMDD.csv          (immutable archive)
#    And prints a diff vs the previous archive — if the prediction is identical
#    to yesterday's, an explicit ⚠️ warning is emitted.
./venv/Scripts/python -m xgbmodel.main --mode predict

# 2b. (occasional) re-run the capped backtest if you've changed the ST cap
#     or refreshed the OOF predictions. Surfaced in the dashboard's
#     "ST 限额回测对照" section.
./venv/Scripts/python -m backtest.xgb_markowitz \
  --solver qp --max_st_per_day 4 --tag qp_st4

# 3. rebuild the predictions dashboard
./venv/Scripts/python -m dashboard.build

# 4. rebuild the backtest dashboard (only when you re-ran the backtest)
./venv/Scripts/python -m dashboard.build_backtest --tag qp

# 5. rebuild the combined dashboard (re-uses cached Barra → ~5 s)
./venv/Scripts/python -m dashboard.build_combined --cached-barra dashboard/barra_cache.json

# 6. encrypt + ship
./venv/Scripts/python -m dashboard.package_secure          --password 'YOURPASS'
./venv/Scripts/python -m dashboard.package_secure_combined --password 'YOURPASS'
```

### Source-of-truth flow

```
api.st_history --download                 backtest.xgb_markowitz
     │                                       --max_st_per_day 4 --tag qp_st4
     │                                            │
     ▼                                            ▼
stock_data/st_history.csv          plots/.../equity_qp_st4.csv
   (783 stocks, 2,433 events)        plots/.../trades_qp_st4.csv
                       │             │
xgbmodel.main          │             │
   --mode predict      │             │
     │                 │             │
     ├─→ stock_predictions_xgb.csv   │
     │   (live pointer; both         │
     │   dashboards read this)       │
     │                 │             │
     └─→ stock_predictions_xgb_features_YYYYMMDD.csv  (audit archive)
                       │             │
                       ▼             ▼
                 dashboard.build_combined
                       │
                       ▼
              dashboard/combined_data.json
                       │
                       ▼
              combined.html / index_combined.html / index_combined_secure.html
```

The combined dashboard auto-discovers any of the input files and degrades
gracefully if one is missing (the corresponding section just shows a "请运行
…" placeholder instead of crashing).

If you skip step 2, the `dashboard/live_prediction.py` staleness check fires
during step 5 and the combined dashboard renders a red "⚠️ 警告：预测数据陈旧"
banner above the live-forecast section pointing at the freshness mismatch.

### Verifying the refresh actually moved the prediction

```bash
./venv/Scripts/python -c "
import pandas as pd, glob
archives = sorted(glob.glob('stock_predictions_xgb_features_*.csv'))
if len(archives) < 2:
    print('not enough archives to diff')
else:
    a = pd.read_csv(archives[-2]).set_index('ts_code')
    b = pd.read_csv(archives[-1]).set_index('ts_code')
    common = a.index.intersection(b.index)
    diff = (a.loc[common, 'pred_pct_chg_next'] - b.loc[common, 'pred_pct_chg_next']).abs()
    print(f'{archives[-2]} vs {archives[-1]}:')
    print(f'  differ on {(diff > 1e-6).sum()}/{len(diff)} stocks, mean|Δ|={diff.mean():.4f}, max|Δ|={diff.max():.4f}')
"
```

A clean refresh between two consecutive trading days produces something like
`differ on 5057/5058 stocks, mean|Δ|=0.1735, max|Δ|=9.7598`. Identical
predictions on consecutive days would mean `predict` ran on stale data —
investigate before publishing.

---

## 7. ST findings — what the dashboard reveals

The combined dashboard's **ST 个股结局** + **ST 限额回测** + the live ST toggle
together answer four practical questions about how dependent the strategy is
on Special-Treatment (ST/*ST) stocks. Below is the empirical snapshot at the
time of writing — all numbers are reproduced inside the dashboard (live cards
and tables).

### Q1. How much of the historical alpha comes from ST trading?

Almost all of it. Of the 7,676 historical backtest trades:

- **57.2% (4,393)** were on stocks ST/*ST at the time of trade
- They produced **99.7% of net P&L** (+141.0M out of +141.3M CNY)
- TP hit rate on ST trades was **76.2%** vs 36–44% on non-ST

ST detection here is the authoritative tushare `namechange` roster
(`stock_data/st_history.csv`), not the deprecated 5%-band heuristic.

### Q2. What happens to those ST stocks afterwards — do they get delisted?

Of the 310 distinct ST stocks the strategy ever entered:

| Current state | Stocks | Outcome |
|---|---|---|
| **recovered** (摘帽) | 178 (57.4%) | Got the ST/*ST tag removed |
| **still_st** | 131 (42.3%) | Still under special treatment |
| **delisted** | 1 (0.3%) | Actually delisted (P&L still +850K CNY) |
| paused | 0 | — |

The 1–5 day holding window catches the volatile rebound moves without
staying long enough for delisting events to materialise. Delisting risk is
real but extremely rare in this sub-strategy.

### Q3. Does the alpha survive a hard cap on ST exposure?

`backtest/xgb_markowitz.py --max_st_per_day 4` reruns the same QP backtest
with at most 4 ST positions in any daily portfolio. Side-by-side over the
same 2021-04-22 → 2026-04-21 window:

| Metric | Original (no cap) | Capped (max 4 ST) | Δ |
|---|---|---|---|
| CAGR | +169.71% | **+148.91%** | −20.8 pp |
| Sharpe (rf=0) | 7.58 | **6.86** | −0.72 |
| Max drawdown | −18.20% | **−19.69%** | −1.5 pp |
| Annual alpha | +104.4% | **+96.1%** | −8.3 pp |
| Information ratio | 5.32 | **4.92** | −0.40 |
| Trades | 7,676 | 6,265 | −18% |

The strategy retains 88% of CAGR and 91% of Sharpe under the cap. The
non-ST signal exists.

### Q4. What should I show in the live recommendation today?

The live "下一交易日预测" section provides two views via a radio toggle:

- **排除 ST (实盘推荐)** ⭐ default — applies the same authoritative roster
  to filter ST/*ST candidates out of the top-200 pool. If the QP returns
  zero weights (no positive-edge non-ST candidates), a green
  "建议持币观望 (cash recommendation)" banner replaces the portfolio table.
- **包含 ST (与回测一致)** — keeps everything, mirrors the historical
  backtest's filter.

`--max_st_per_day 4` (the dashboard's recommended balance) is not yet a
live-time CLI flag for the live forecast — only `exclude_st: bool`. If you
want the live signal to mirror the capped backtest exactly (4 ST allowed),
that's a future enhancement to `dashboard/live_prediction.py`.

---

## 8. Serve locally

```bash
./venv/Scripts/python -m http.server -d dashboard 8000
```

Then open one of:

- <http://localhost:8000/>                   — prediction dashboard
- <http://localhost:8000/backtest.html>      — backtest dashboard
- <http://localhost:8000/combined.html>      — unified dashboard

The `index_*.html` and `index_*_secure.html` variants work the same way (they
ignore the JSON sidecars because their data is embedded).
