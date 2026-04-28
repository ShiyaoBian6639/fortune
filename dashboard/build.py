"""
Build the xgbmodel prediction dashboard.

Reads:
  - stock_predictions_xgb_20260424.csv   : live predictions for t+1 (5066 stocks)
  - stock_data/models/xgb_preds/test.csv : 6.25M OOS walk-forward (pred, target) pairs
  - stock_data/models/xgb_pct_chg.meta.json : model metadata + global feature importance
  - stock_sectors.csv                    : per-stock area / market / board / name
  - stock_data/{sh,sz}/*.csv             : raw OHLCV for factor snapshots

Writes:
  - dashboard/data.json : all computed metrics, group stats, top stock data
  - dashboard/index.html : static HTML dashboard (uses Plotly CDN)

Run:
  ./venv/Scripts/python -m dashboard.build
Then:
  ./venv/Scripts/python -m http.server -d dashboard 8000
and open http://localhost:8000
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(f"[dashboard.build] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# 1. Load inputs
# ────────────────────────────────────────────────────────────────────────────

def load_live_predictions() -> pd.DataFrame:
    """Load the most recent live prediction.

    Prefers the live pointer `stock_predictions_xgb.csv` (always points at the
    latest run of `xgbmodel.main --mode predict`). Falls back to the most
    recent dated archive `stock_predictions_xgb_features_*.csv` if the pointer
    is missing.
    """
    pointer = ROOT / 'stock_predictions_xgb.csv'
    if pointer.exists():
        p = pointer
    else:
        archives = sorted(ROOT.glob('stock_predictions_xgb_features_*.csv'))
        if not archives:
            raise FileNotFoundError(
                "No live predictions found. Run `./venv/Scripts/python -m xgbmodel.main --mode predict` first.")
        p = archives[-1]
        _log(f"live pointer missing — using archive {p.name}")
    df = pd.read_csv(p)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    _log(f"live predictions: {len(df):,} rows from {p.name}, feature_date={df['trade_date'].max().date()}")
    return df


def _next_trading_day_for(feature_date: pd.Timestamp,
                           ts_codes: list) -> str:
    """Find the next trading day after feature_date by scanning a few stock CSVs."""
    DATA = ROOT / 'stock_data'
    for ts in ts_codes[:20]:
        code, suffix = ts.split('.')
        sub = 'sh' if suffix.upper() == 'SH' else 'sz'
        fp = DATA / sub / f'{code}.csv'
        if not fp.exists():
            continue
        try:
            d = pd.read_csv(fp, usecols=['trade_date'])
            d['trade_date'] = pd.to_datetime(d['trade_date'].astype(str))
            after = d[d['trade_date'] > feature_date]['trade_date']
            if not after.empty:
                return after.min().strftime('%Y-%m-%d')
        except Exception:
            continue
    # No realised data — calendar-based estimate
    return (feature_date + pd.tseries.offsets.BDay(1)).strftime('%Y-%m-%d')


def load_oos_test() -> pd.DataFrame:
    p = ROOT / 'stock_data' / 'models' / 'xgb_preds' / 'test.csv'
    df = pd.read_csv(p, parse_dates=['trade_date'])
    _log(f"OOS test: {len(df):,} rows, {df['trade_date'].min().date()} → {df['trade_date'].max().date()}")
    return df


def load_model_meta() -> dict:
    p = ROOT / 'stock_data' / 'models' / 'xgb_pct_chg.meta.json'
    with open(p) as f:
        return json.load(f)


def load_sectors() -> pd.DataFrame:
    p = ROOT / 'stock_sectors.csv'
    with open(p, 'rb') as f:
        raw = f.read().decode('utf-8', errors='replace')
    df = pd.read_csv(io.StringIO(raw))
    # Standardize names for the dashboard
    df = df.rename(columns={'market': 'board', 'area': 'province'})
    # Board → English for consistent display
    board_map = {'主板': 'Main Board', '创业板': 'ChiNext', '科创板': 'STAR Market'}
    df['board_en'] = df['board'].map(board_map).fillna(df['board'])
    # ts_code prefix → exchange
    df['exchange'] = df['ts_code'].str.endswith('.SH').map({True: 'SSE', False: 'SZSE'})
    _log(f"sectors: {len(df)} stocks, {df['province'].nunique()} provinces, {df['board'].nunique()} boards")
    return df[['ts_code', 'name', 'province', 'board', 'board_en', 'exchange', 'list_date']]


# ────────────────────────────────────────────────────────────────────────────
# 2. Overall accuracy metrics (on OOS test.csv)
# ────────────────────────────────────────────────────────────────────────────

def compute_overall_metrics(test: pd.DataFrame) -> dict:
    _log("computing overall metrics ...")
    pred, tgt = test['pred'].values, test['target'].values
    n = len(test)

    mae  = float(np.mean(np.abs(pred - tgt)))
    mse  = float(np.mean((pred - tgt) ** 2))
    rmse = float(np.sqrt(mse))

    # Pearson IC (full), Spearman IC on sample (full is O(n log n) sort → slow on 6M)
    pearson = float(np.corrcoef(pred, tgt)[0, 1])
    rng = np.random.default_rng(0)
    samp = rng.choice(n, min(500_000, n), replace=False)
    spearman = float(spearmanr(pred[samp], tgt[samp]).correlation)

    # Directional hit rate
    hr = float(np.mean((pred > 0) == (tgt > 0)))

    # Magnitude-confident HR (top-decile |pred|)
    q90 = np.quantile(np.abs(pred), 0.9)
    mask = np.abs(pred) >= q90
    hr_conf = float(np.mean((pred[mask] > 0) == (tgt[mask] > 0))) if mask.sum() > 0 else 0.0

    # R² (won't be impressive — 6% IC → R² ~0.005). Still report.
    ss_res = np.sum((tgt - pred) ** 2)
    ss_tot = np.sum((tgt - tgt.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    out = {
        'rows': int(n),
        'stocks': int(test['ts_code'].nunique()),
        'date_start': test['trade_date'].min().strftime('%Y-%m-%d'),
        'date_end':   test['trade_date'].max().strftime('%Y-%m-%d'),
        'folds': int(test['fold'].nunique()),
        'mae': mae, 'mse': mse, 'rmse': rmse,
        'pearson_ic': pearson, 'spearman_ic': spearman,
        'hit_rate': hr, 'hit_rate_top10pct': hr_conf,
        'r2': r2,
    }
    _log(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  PearsonIC={pearson:.4f}  "
         f"SpearmanIC={spearman:.4f}  HR={hr:.4f}")
    return out


def compute_daily_rank_ic(test: pd.DataFrame) -> dict:
    """Daily cross-sectional rank IC series — the metric that actually matters."""
    _log("computing daily rank IC ...")
    # Group by date, spearman(pred, target) per day
    def ic(g):
        if len(g) < 10:
            return np.nan
        return spearmanr(g['pred'].values, g['target'].values).correlation

    daily = test.groupby('trade_date').apply(ic).dropna()
    daily.index = pd.to_datetime(daily.index)

    # Rolling 20-day mean for plotting
    rolling = daily.rolling(20, min_periods=5).mean()

    out = {
        'dates': [d.strftime('%Y-%m-%d') for d in daily.index],
        'ic_daily': [float(x) for x in daily.values],
        'ic_roll20': [float(x) if not np.isnan(x) else None for x in rolling.values],
        'mean': float(daily.mean()),
        'std':  float(daily.std()),
        'pct_positive': float((daily > 0).mean()),
        'icir': float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0,
    }
    _log(f"  daily rank IC mean={out['mean']:.4f}  ICIR={out['icir']:.4f}  "
         f"pos%={out['pct_positive']:.2%}")
    return out


def compute_decile_analysis(test: pd.DataFrame) -> dict:
    """Long-short decile spread — the classic alpha diagnostic.
    For each day: bucket stocks into 10 deciles by pred, then measure the mean
    target per decile. Top-minus-bottom averaged across days = the strategy's
    daily long-short return in excess-return units.
    """
    _log("computing decile analysis ...")
    df = test[['trade_date', 'pred', 'target']].copy()
    # Bucket by pred within each day
    df['decile'] = df.groupby('trade_date')['pred'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') if len(x) >= 10 else np.nan
    )
    df = df.dropna(subset=['decile'])
    df['decile'] = df['decile'].astype(int) + 1

    # Mean target per decile, across all (stock, day) observations
    decile_stats = df.groupby('decile')['target'].agg(['mean', 'std', 'count']).reset_index()

    # Long-short: daily (top-decile mean) - (bottom-decile mean), average over days
    def ls(g):
        top = g.loc[g['decile'] == 10, 'target'].mean()
        bot = g.loc[g['decile'] == 1,  'target'].mean()
        return top - bot
    daily_ls = df.groupby('trade_date').apply(ls).dropna()
    cum_ls = daily_ls.cumsum()

    out = {
        'decile': decile_stats['decile'].tolist(),
        'mean_target': [float(x) for x in decile_stats['mean']],
        'std_target':  [float(x) for x in decile_stats['std']],
        'count':       [int(x) for x in decile_stats['count']],
        'long_short_daily_mean': float(daily_ls.mean()),
        'long_short_daily_std':  float(daily_ls.std()),
        'long_short_sharpe': float(daily_ls.mean() / daily_ls.std() * np.sqrt(252))
            if daily_ls.std() > 0 else 0.0,
        'cum_dates': [d.strftime('%Y-%m-%d') for d in cum_ls.index],
        'cum_long_short': [float(x) for x in cum_ls.values],
    }
    _log(f"  L/S top-minus-bottom daily mean = {out['long_short_daily_mean']:.4f}%  "
         f"Sharpe = {out['long_short_sharpe']:.2f}")
    return out


def compute_scatter_sample(test: pd.DataFrame, n: int = 20_000) -> dict:
    """Random sample of (pred, target) for visualization — full 6M is too heavy."""
    samp = test.sample(n, random_state=0)
    return {
        'pred':   [float(x) for x in samp['pred'].values],
        'target': [float(x) for x in samp['target'].values],
    }


# ────────────────────────────────────────────────────────────────────────────
# 3. Group metrics by sector / province / board
# ────────────────────────────────────────────────────────────────────────────

def compute_group_metrics(test: pd.DataFrame, sectors: pd.DataFrame,
                          group_col: str, min_rows: int = 5000) -> list:
    """Per-group MAE, RMSE, Pearson IC, daily-avg rank IC, HR."""
    merged = test.merge(sectors[['ts_code', group_col]], on='ts_code', how='left')
    merged = merged.dropna(subset=[group_col])

    rows = []
    for g, sub in merged.groupby(group_col):
        if len(sub) < min_rows:
            continue
        pred, tgt = sub['pred'].values, sub['target'].values
        mae  = float(np.mean(np.abs(pred - tgt)))
        rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
        pic  = float(np.corrcoef(pred, tgt)[0, 1]) if sub['pred'].std() > 0 else 0.0
        # Daily rank IC averaged over dates
        def dayic(d):
            if len(d) < 5: return np.nan
            return spearmanr(d['pred'], d['target']).correlation
        daily = sub.groupby('trade_date').apply(dayic).dropna()
        rank_ic = float(daily.mean()) if len(daily) else 0.0
        hr   = float(np.mean((pred > 0) == (tgt > 0)))
        rows.append({
            'group': str(g),
            'rows': int(len(sub)),
            'stocks': int(sub['ts_code'].nunique()),
            'mae': mae, 'rmse': rmse,
            'pearson_ic': pic, 'rank_ic': rank_ic, 'hit_rate': hr,
        })
    rows.sort(key=lambda r: r['rank_ic'], reverse=True)
    return rows


# ────────────────────────────────────────────────────────────────────────────
# 4. Live top-N / bottom-N with factor snapshot + reasoning
# ────────────────────────────────────────────────────────────────────────────

def _load_recent_stock_data(ts_code: str) -> pd.DataFrame | None:
    """Load last ~60 trading days for a specific stock."""
    code, ex = ts_code.split('.')
    sub = 'sh' if ex == 'SH' else 'sz'
    p = ROOT / 'stock_data' / sub / f'{code}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, nrows=60)  # files are newest-first
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
    return df.sort_values('trade_date').reset_index(drop=True)


def compute_factor_snapshot(ts_code: str) -> dict:
    """
    Rule-based 'reasoning' features that are interpretable and correlate with
    the top-20 XGB feature importances (turnover_rate_f, hl_ratio, pct_chg,
    overnight_gap, close_ma_20_ratio, net_mf_amount_ratio …).

    Not true SHAP — but tells a consistent story with what the model learned.
    """
    df = _load_recent_stock_data(ts_code)
    if df is None or len(df) < 21:
        return {}
    last = df.iloc[-1]
    mean5 = df.tail(5)['pct_chg'].mean()
    mean20 = df.tail(20)['pct_chg'].mean()
    vol20 = df.tail(20)['pct_chg'].std()
    close20 = df.tail(20)['close'].mean()
    amt_mean20 = df.tail(20)['amount'].mean()
    amt_last = last['amount']
    hl_last = (last['high'] - last['low']) / max(last['pre_close'], 1e-9) * 100
    gap_last = (last['open'] - last['pre_close']) / max(last['pre_close'], 1e-9) * 100
    close_ma20 = last['close'] / max(close20, 1e-9) - 1.0
    # recent 5-day window: up-day count
    up5 = int((df.tail(5)['pct_chg'] > 0).sum())

    # "Why" narrative tokens — human-readable reason flags
    reasons = []
    if mean5 > 2.5: reasons.append(f"strong 5-day momentum (+{mean5:.2f}%/day)")
    if mean5 < -2.5: reasons.append(f"5-day drawdown ({mean5:.2f}%/day)")
    if close_ma20 > 0.05: reasons.append(f"trading {close_ma20*100:+.1f}% above 20-day MA")
    if close_ma20 < -0.05: reasons.append(f"trading {close_ma20*100:+.1f}% below 20-day MA")
    if amt_mean20 > 0 and amt_last / amt_mean20 > 2.0:
        reasons.append(f"volume spike: {amt_last/amt_mean20:.1f}× 20-day avg")
    if hl_last > 6:
        reasons.append(f"wide range ({hl_last:.1f}% H-L)")
    if abs(gap_last) > 1.5:
        reasons.append(f"open gap {gap_last:+.1f}%")
    if up5 >= 4: reasons.append("4+ up days of the last 5")
    if up5 <= 1: reasons.append("1 or fewer up days of last 5")
    if vol20 > 5: reasons.append(f"high realized vol ({vol20:.1f}%)")

    return {
        'last_close':      float(last['close']),
        'last_pct_chg':    float(last['pct_chg']),
        'ret_5d_mean':     float(mean5),
        'ret_20d_mean':    float(mean20),
        'vol_20d':         float(vol20),
        'close_vs_ma20':   float(close_ma20 * 100),  # %
        'amount_ratio_20': float(amt_last / amt_mean20) if amt_mean20 > 0 else 0.0,
        'hl_range_pct':    float(hl_last),
        'open_gap_pct':    float(gap_last),
        'up_days_last_5':  up5,
        'reasons':         reasons,
        'history': {
            'dates':  [d.strftime('%Y-%m-%d') for d in df['trade_date']],
            'closes': [float(x) for x in df['close'].values],
            'pct':    [float(x) for x in df['pct_chg'].values],
        }
    }


def build_live_topn(live: pd.DataFrame, sectors: pd.DataFrame, n: int = 20) -> dict:
    """Top-N predicted outperformers and bottom-N predicted underperformers, enriched."""
    _log(f"building live top/bottom {n} with factor snapshots ...")
    merged = live.merge(sectors, on='ts_code', how='left')
    merged = merged.sort_values('pred_pct_chg_next', ascending=False).reset_index(drop=True)

    def _opt(row, key, default=float('nan')):
        try:
            v = row.get(key, default)
            return float(v) if v is not None else default
        except (KeyError, TypeError, ValueError):
            return default

    def enrich(row: pd.Series) -> dict:
        rec = {
            'ts_code': row['ts_code'],
            'name': row.get('name', ''),
            'province': row.get('province', ''),
            'board_en': row.get('board_en', ''),
            'exchange': row.get('exchange', ''),
            'pred_pct_chg': float(row['pred_pct_chg_next']),
            'prob_up':      _opt(row, 'prob_up'),
            'prob_gt_1pct': _opt(row, 'prob_gt_1pct'),
            'prob_gt_3pct': _opt(row, 'prob_gt_3pct'),
            'prob_gt_5pct': _opt(row, 'prob_gt_5pct'),
            'prob_lt_1pct': _opt(row, 'prob_lt_1pct'),
            'prob_lt_3pct': _opt(row, 'prob_lt_3pct'),
            'prob_lt_5pct': _opt(row, 'prob_lt_5pct'),
            'pi_lo_80':     _opt(row, 'pi_lo_80'),
            'pi_hi_80':     _opt(row, 'pi_hi_80'),
        }
        rec.update(compute_factor_snapshot(row['ts_code']))
        return rec

    top = [enrich(merged.iloc[i]) for i in range(n)]
    bot = [enrich(merged.iloc[-(i+1)]) for i in range(n)]
    return {'top': top, 'bottom': bot[::-1]}


def build_live_table(live: pd.DataFrame, sectors: pd.DataFrame) -> list:
    """Full list of live predictions, slim for the searchable table.

    Probability/PI columns are optional — if the prediction CSV doesn't have
    them (e.g. produced by model_compare.refit_canonical instead of the
    canonical xgbmodel.predict_latest path), they're filled with NaN and
    rounded gracefully.
    """
    merged = live.merge(sectors, on='ts_code', how='left')
    base_cols = ['ts_code', 'name', 'province', 'board_en', 'exchange',
                 'pred_pct_chg_next']
    opt_cols  = ['prob_up', 'prob_gt_3pct', 'prob_lt_3pct', 'pi_lo_80', 'pi_hi_80']
    for c in opt_cols:
        if c not in merged.columns:
            merged[c] = float('nan')
    out = merged[base_cols + opt_cols].copy()
    out = out.sort_values('pred_pct_chg_next', ascending=False)
    out['pred_pct_chg_next'] = out['pred_pct_chg_next'].astype(float).round(3)
    for c in opt_cols:
        out[c] = pd.to_numeric(out[c], errors='coerce').round(3)
    out = out.fillna('')
    return out.to_dict(orient='records')


# ────────────────────────────────────────────────────────────────────────────
# 5. Distribution of live predictions
# ────────────────────────────────────────────────────────────────────────────

def compute_live_distribution(live: pd.DataFrame) -> dict:
    pred = live['pred_pct_chg_next'].values
    hist, edges = np.histogram(pred, bins=40)
    return {
        'bin_edges':  [float(x) for x in edges],
        'bin_counts': [int(x)   for x in hist],
        'mean': float(pred.mean()),
        'std':  float(pred.std()),
        'p05':  float(np.percentile(pred, 5)),
        'p50':  float(np.percentile(pred, 50)),
        'p95':  float(np.percentile(pred, 95)),
        'n_up_signals':   int((pred > 0).sum()),
        'n_down_signals': int((pred < 0).sum()),
        'n_strong_up':    int((pred > 3).sum()),
        'n_strong_down':  int((pred < -3).sum()),
    }


# ────────────────────────────────────────────────────────────────────────────
# 5b. Backtest (xgb_markowitz) — load equity curves + compute summary
# ────────────────────────────────────────────────────────────────────────────

BACKTEST_DIR = ROOT / 'plots' / 'backtest_xgb_markowitz'
CSI300_CSV   = ROOT / 'stock_data' / 'index' / 'idx_factor_pro' / '000300_SH.csv'


def _load_csi300_for_backtest() -> pd.DataFrame:
    df = pd.read_csv(CSI300_CSV, encoding='utf-8-sig', usecols=['trade_date', 'close'])
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    return df.sort_values('trade_date').set_index('trade_date')


def _compute_backtest_scenario(equity_csv: Path, trades_csv: Path,
                                initial: float, bench: pd.DataFrame,
                                key: str, label_cn: str) -> dict | None:
    if not equity_csv.exists():
        return None
    eq = pd.read_csv(equity_csv, parse_dates=['trade_date']).set_index('trade_date')
    nav = eq['nav'].astype(float)
    if nav.empty:
        return None

    b = bench.reindex(nav.index, method='ffill')
    bclose = b['close'].astype(float)

    ret  = nav.pct_change().fillna(0.0)
    bret = bclose.pct_change().fillna(0.0)

    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-6)
    total_return = float(nav.iloc[-1] / initial - 1.0)
    cagr  = float((nav.iloc[-1] / initial) ** (1.0 / years) - 1.0)
    vol   = float(ret.std(ddof=1) * np.sqrt(252))
    sharpe = float((ret.mean() * 252) / (vol + 1e-12))
    roll_max = nav.cummax()
    dd  = (nav / roll_max - 1.0)
    mdd = float(dd.min())
    mdd_trough = dd.idxmin()
    mdd_peak   = roll_max.loc[:mdd_trough].idxmax()
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float('nan')

    b_total  = float(bclose.iloc[-1] / bclose.iloc[0] - 1.0)
    b_cagr   = float((bclose.iloc[-1] / bclose.iloc[0]) ** (1.0 / years) - 1.0)
    b_roll   = bclose.cummax()
    b_mdd    = float((bclose / b_roll - 1.0).min())

    cov = np.cov(ret.values, bret.values, ddof=1)
    beta  = float(cov[0, 1] / (cov[1, 1] + 1e-12))
    alpha = float((ret.mean() - beta * bret.mean()) * 252)
    active = ret - bret
    ir = float((active.mean() * 252) / (active.std(ddof=1) * np.sqrt(252) + 1e-12))

    # Curve — downsample to weekly for JSON size
    step = max(1, len(nav) // 300)
    nav_ds  = nav.iloc[::step]
    bnorm   = bclose / bclose.iloc[0] * initial
    bnorm_ds = bnorm.iloc[::step]
    dd_pct_ds = (dd * 100).iloc[::step]

    # Trades summary
    trades_stats = {}
    if trades_csv.exists():
        td = pd.read_csv(trades_csv)
        if len(td) > 0:
            wins = td[td['pnl'] > 0]
            losses = td[td['pnl'] <= 0]
            trades_stats = {
                'n_trades':     int(len(td)),
                'hit_rate':     float(len(wins) / len(td)),
                'avg_win_pct':  float(wins['ret'].mean() * 100) if len(wins) else 0.0,
                'avg_loss_pct': float(losses['ret'].mean() * 100) if len(losses) else 0.0,
                'avg_ret_pct':  float(td['ret'].mean() * 100),
                'median_hold_days': float(td['held_days'].median()),
                'pct_tp':       float((td['reason'] == 'take_profit').mean()),
                'pct_sl':       float((td['reason'] == 'stop_loss').mean()),
                'pct_horizon':  float((td['reason'] == 'horizon').mean()),
            }

    return {
        'key': key,
        'label_cn': label_cn,
        'initial': float(initial),
        'date_start': nav.index[0].strftime('%Y-%m-%d'),
        'date_end':   nav.index[-1].strftime('%Y-%m-%d'),
        'years': float(years),
        'final_nav': float(nav.iloc[-1]),
        'total_return': total_return,
        'cagr': cagr,
        'vol_ann': vol,
        'sharpe': sharpe,
        'max_drawdown': mdd,
        'mdd_peak_date':  str(mdd_peak.date()),
        'mdd_trough_date': str(mdd_trough.date()),
        'calmar': calmar,
        'bench_total': b_total,
        'bench_cagr':  b_cagr,
        'bench_mdd':   b_mdd,
        'alpha_ann': alpha,
        'beta': beta,
        'info_ratio': ir,
        'curve': {
            'dates': [d.strftime('%Y-%m-%d') for d in nav_ds.index],
            'nav':   [float(x) for x in nav_ds.values],
            'bench': [float(x) for x in bnorm_ds.values],
            'dd_pct': [float(x) for x in dd_pct_ds.values],
        },
        'trades': trades_stats,
    }


def load_backtest_data() -> dict:
    """Load all three backtest scenarios (unconstrained + two liquidity-capped)."""
    if not BACKTEST_DIR.exists():
        _log("backtest dir missing — skipping backtest panel")
        return {'scenarios': []}
    _log("loading backtest equity curves ...")
    bench = _load_csi300_for_backtest()

    cfgs = [
        ('unconstrained',  'equity.csv',                 'trades.csv',
         1_000_000,   '理想组合 · 100万起始 · 无流动性约束'),
        ('realistic_10m',  'equity_realistic_10m.csv',   'trades_realistic_10m.csv',
         10_000_000,  '真实约束 · 1000万起始 · 5% ADV上限'),
        ('realistic_100m', 'equity_realistic_100m.csv',  'trades_realistic_100m.csv',
         100_000_000, '真实约束 · 1亿起始 · 5% ADV上限'),
    ]
    scenarios = []
    for key, eq_name, tr_name, initial, label in cfgs:
        s = _compute_backtest_scenario(
            BACKTEST_DIR / eq_name, BACKTEST_DIR / tr_name,
            initial, bench, key, label,
        )
        if s is not None:
            scenarios.append(s)
            _log(f"  {key}: CAGR={s['cagr']*100:+.2f}%  Sharpe={s['sharpe']:.2f}  "
                 f"MDD={s['max_drawdown']*100:+.2f}%")
    return {'scenarios': scenarios}


# ────────────────────────────────────────────────────────────────────────────
# 6. Orchestration
# ────────────────────────────────────────────────────────────────────────────

def build() -> None:
    t0 = time.time()
    live    = load_live_predictions()
    test    = load_oos_test()
    meta    = load_model_meta()
    sectors = load_sectors()

    feature_date = live['trade_date'].max()
    prediction_date = _next_trading_day_for(feature_date,
                                             live['ts_code'].head(20).tolist())
    data = {
        'generated_at':     pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction_date':  prediction_date,
        'feature_date':     feature_date.strftime('%Y-%m-%d'),
        'n_live_stocks':    int(len(live)),
        'model_meta': {
            'target_mode':    meta['target_mode'],
            'forward_window': meta['forward_window'],
            'n_features':     meta['n_features'],
            'n_folds':        len(meta['per_fold']),
            'canonical_n_estimators': meta['canonical_n_estimators'],
            'xgb_params':     meta['xgb_params'],
            'metric_summary': meta['metric_summary'],
            'total_train_seconds': meta['total_seconds'],
        },
        'feature_importance_top30': meta['feature_importance_top50'][:30],
        'overall_metrics':   compute_overall_metrics(test),
        'daily_rank_ic':     compute_daily_rank_ic(test),
        'decile_analysis':   compute_decile_analysis(test),
        'scatter_sample':    compute_scatter_sample(test, 15_000),
        'group_by_province': compute_group_metrics(test, sectors, 'province', min_rows=20000),
        'group_by_board':    compute_group_metrics(test, sectors, 'board_en', min_rows=50000),
        'group_by_exchange': compute_group_metrics(test, sectors, 'exchange', min_rows=50000),
        'live_distribution': compute_live_distribution(live),
        'live_topn':         build_live_topn(live, sectors, n=30),
        'live_table':        build_live_table(live, sectors),
        'backtest':          load_backtest_data(),
    }

    out_json = OUT / 'data.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    _log(f"wrote {out_json}  ({out_json.stat().st_size / 1024:.1f} KB)")
    _log(f"total build time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    build()
