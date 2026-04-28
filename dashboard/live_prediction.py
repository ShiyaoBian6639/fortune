"""
Build a "next trading day" payload from the latest XGB live predictions.

Steps:
1. Read the most recent prediction CSV produced by `xgbmodel.predict.predict_latest`
   (default location: PREDICT_OUT, written one row per stock with point estimate,
   80% prediction interval, probabilities). The feature_date in the CSV is the
   last day of available data; the forecast horizon is feature_date + 1 trading day.
2. For the top-K candidates, run the same long-only Markowitz QP solver used in
   the backtest:
       Σ from a 60-day rolling pct_chg panel ending strictly before forecast date,
       with Ledoit-Wolf shrinkage; SLSQP solves
           min 0.5·λ·wᵀΣw − μᵀw  s.t.  Σwᵢ = 1, wᵢ ≥ 0.
3. Return a JSON-serialisable dict with: feature_date, forecast_date, candidates
   (per-stock μ, σ, π_lo, π_hi, prob_*), portfolio (ts_code, name, weight, μ, σ).

This mirrors the backtest's decision logic exactly, applied to the live
features. No future data is consulted (verified by the dashboard.leakage_audit
narrative).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'

from backtest.xgb_markowitz import (
    estimate_covariance, markowitz_qp_weights, markowitz_diagonal_weights
)


def _log(msg: str) -> None:
    print(f"[live_prediction] {msg}", flush=True)


def _next_trading_day(feature_date: pd.Timestamp,
                      ts_codes: List[str]) -> Optional[pd.Timestamp]:
    """Find the next trading day after feature_date by checking when at least one
    sample stock has data. Returns None if no data points exist after feature_date."""
    candidates = []
    for ts_code in ts_codes[:10]:
        code, suffix = ts_code.split('.')
        sub = 'sh' if suffix.upper() == 'SH' else 'sz'
        fp = DATA_DIR / sub / f'{code}.csv'
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, usecols=['trade_date'])
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            after = df[df['trade_date'] > feature_date]['trade_date']
            if not after.empty:
                candidates.append(after.min())
        except Exception:
            continue
    return min(candidates) if candidates else None


def _load_pct_chg_window(ts_code: str, end_date: pd.Timestamp,
                          window: int) -> np.ndarray:
    """Load `window` most recent pct_chg values strictly before `end_date`."""
    code, suffix = ts_code.split('.')
    sub = 'sh' if suffix.upper() == 'SH' else 'sz'
    fp = DATA_DIR / sub / f'{code}.csv'
    if not fp.exists():
        return np.full(window, np.nan)
    try:
        df = pd.read_csv(fp, usecols=['trade_date', 'pct_chg'])
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df = df[df['trade_date'] <= end_date].sort_values('trade_date')
        # Drop end_date if it's in there (the QP semantics: data strictly before end)
        df = df[df['trade_date'] < end_date]
        arr = df['pct_chg'].astype(np.float64).tail(window).to_numpy()
        if arr.size < window:
            arr = np.concatenate([np.full(window - arr.size, np.nan), arr])
        return arr
    except Exception:
        return np.full(window, np.nan)


def _load_close_at(ts_code: str, date: pd.Timestamp) -> Optional[float]:
    code, suffix = ts_code.split('.')
    sub = 'sh' if suffix.upper() == 'SH' else 'sz'
    fp = DATA_DIR / sub / f'{code}.csv'
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, usecols=['trade_date', 'close', 'pct_chg'])
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        match = df[df['trade_date'] == date]
        if match.empty:
            return None
        return float(match['close'].iloc[0]), float(match['pct_chg'].iloc[0])
    except Exception:
        return None


def _band_pct_at(ts_code: str, date: pd.Timestamp,
                  cache: dict) -> Optional[float]:
    """Daily price-band % for a stock on a given date, derived from stk_limit.

    Returns the symmetric band in percent — 5 for *ST, 10 for main board,
    20 for ChiNext (300xxx) / STAR (688xxx). Pre-close is implicit in
    (up_limit + down_limit) / 2 — no dividend-adjustment issue.

    Used for the dashboard "状态" tag (主板 / 创科 / ⚠️ *ST display) — but
    not for ST detection itself, which uses the authoritative tushare roster.
    """
    ds = date.strftime('%Y%m%d')
    if ds not in cache:
        fp = DATA_DIR / 'stk_limit' / f'stk_limit_{ds}.csv'
        if not fp.exists():
            cache[ds] = None
            return None
        try:
            df = pd.read_csv(fp, usecols=['ts_code', 'up_limit', 'down_limit']).set_index('ts_code')
            cache[ds] = df
        except Exception:
            cache[ds] = None
            return None
    cached = cache[ds]
    if cached is None or ts_code not in cached.index:
        return None
    up = float(cached.loc[ts_code, 'up_limit'])
    dn = float(cached.loc[ts_code, 'down_limit'])
    if up <= 0 or dn <= 0:
        return None
    mid = (up + dn) / 2.0
    return (up - mid) / mid * 100.0


_ST_INTERVAL_INDEX = None
def _load_st_intervals() -> dict:
    """Load (and cache) the per-stock ST interval index from
    stock_data/st_history.csv. Returns {} if the roster is missing."""
    global _ST_INTERVAL_INDEX
    if _ST_INTERVAL_INDEX is not None:
        return _ST_INTERVAL_INDEX
    try:
        from api.st_history import load_roster, build_daily_index
        roster = load_roster()
        _ST_INTERVAL_INDEX = build_daily_index(roster)
        _log(f"loaded ST roster: {len(_ST_INTERVAL_INDEX):,} ts_codes ever flagged ST")
    except Exception as e:
        _log(f"ST roster unavailable ({e}); falling back to band heuristic")
        _ST_INTERVAL_INDEX = {}
    return _ST_INTERVAL_INDEX


def _is_st(ts_code: str, date: pd.Timestamp,
           band_pct: Optional[float] = None) -> bool:
    """Authoritative ST detection: uses tushare namechange roster if present.

    Falls back to the 5%-band heuristic only if the roster file is missing.
    """
    intervals = _load_st_intervals()
    if intervals:
        ds = date.strftime('%Y%m%d')
        for s, e, _kind in intervals.get(ts_code, ()):
            if s <= ds <= e:
                return True
        return False
    # Heuristic fallback (only used when st_history.csv is missing)
    return band_pct is not None and band_pct < 5.5


def _load_adv20(ts_code: str, end_date: pd.Timestamp,
                window: int = 20) -> Optional[float]:
    """20-day mean amount (CNY) over data strictly before end_date.

    Mirrors backtest/xgb_markowitz.py:compute_rolling_adv — same window,
    same `.shift(1)` semantics (data ≤ end_date−1).
    Returns CNY (tushare's `amount` is in 千元 → ×1000).
    """
    code, suffix = ts_code.split('.')
    sub = 'sh' if suffix.upper() == 'SH' else 'sz'
    fp = DATA_DIR / sub / f'{code}.csv'
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, usecols=['trade_date', 'amount'])
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df = df[df['trade_date'] < end_date].sort_values('trade_date').tail(window)
        if len(df) < 5:
            return None
        return float(df['amount'].mean()) * 1000.0
    except Exception:
        return None


def _load_names() -> dict:
    p = ROOT / 'stock_sectors.csv'
    if not p.exists():
        return {}
    try:
        with open(p, 'rb') as f:
            raw = f.read().decode('utf-8', errors='replace')
        import io
        df = pd.read_csv(io.StringIO(raw), usecols=['ts_code', 'name'])
        return dict(zip(df['ts_code'], df['name']))
    except Exception:
        return {}


def build_live_payload(preds_csv: Path,
                        top_k: int   = 10,
                        cov_window: int = 60,
                        risk_aversion: float = 1.0,
                        # Backtest compatibility: pool = top_k × 4 = 40 by default
                        # (matches backtest/xgb_markowitz.py:407 `head(top_k * 4)`).
                        # On heavy-limit-stop days, this means K can be < 10 —
                        # and the QP correctly returns 100% in the 1 surviving
                        # positive-μ name (or cash if all are negative).
                        candidate_pool: int = 40,
                        limit_pct_main: float = 9.8,
                        limit_pct_chinext: float = 19.8,
                        # Execution-model knobs — defaults match
                        # backtest/xgb_markowitz.py defaults exactly.
                        entry_bps: float = 10.0,
                        exit_bps:  float = 15.0,
                        max_pos_adv: float = 0.05,
                        impact_bps: float = 10.0,
                        initial: float = 1_000_000.0,
                        # ST risk control: cap the number of ST/*ST positions
                        # in the daily portfolio. Mirrors the backtest's
                        # `--max_st_per_day N`:
                        #   -1 → no cap (matches the original backtest +14000% run)
                        #    0 → exclude ST entirely (most risk-averse)
                        #    4 → recommended balance (used by capped backtest)
                        max_st_per_day: int = 4) -> dict:
    """Read latest predictions, select top-K, solve QP, return dashboard payload."""
    if not preds_csv.exists():
        raise SystemExit(f"Live predictions file not found: {preds_csv}\n"
                         "Run `./venv/Scripts/python -m xgbmodel.main --mode predict` first.")
    df = pd.read_csv(preds_csv)
    if 'pred_pct_chg_next' not in df.columns:
        raise RuntimeError(f"{preds_csv} has no `pred_pct_chg_next` column")
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    feature_date = df['trade_date'].max()

    # Sanity check: are the input features at feature_date actually the most recent
    # data on disk? Compare against the latest stock price file.
    stale_warning = None
    try:
        sample_codes = df['ts_code'].head(20).tolist()
        latest_disk = []
        for ts in sample_codes:
            code, suffix = ts.split('.')
            sub = 'sh' if suffix.upper() == 'SH' else 'sz'
            fp = ROOT / 'stock_data' / sub / f'{code}.csv'
            if fp.exists():
                d = pd.read_csv(fp, usecols=['trade_date'])
                latest_disk.append(int(str(d['trade_date'].astype(str).max())))
        if latest_disk:
            disk_max = pd.Timestamp(str(max(latest_disk)))
            if disk_max > feature_date:
                stale_warning = (
                    f"data on disk is fresh through {disk_max.date()} but the "
                    f"prediction file uses features from {feature_date.date()} — "
                    f"re-run `./venv/Scripts/python -m xgbmodel.main --mode predict`"
                )
                _log(f"⚠️ {stale_warning}")
    except Exception as e:
        _log(f"could not run staleness check: {e}")
    df = df[df['trade_date'] == feature_date].copy()
    df = df.rename(columns={'pred_pct_chg_next': 'pred'})
    df = df.sort_values('pred', ascending=False).reset_index(drop=True)
    df['mu_rank'] = df.index + 1   # rank 1 = highest μ in the universe
    _log(f"loaded {len(df):,} live predictions (features@{feature_date.date()})")

    forecast_date = _next_trading_day(feature_date,
                                       df['ts_code'].head(20).tolist())
    if forecast_date is None:
        # No realised data after feature_date yet — predict for "next trading day"
        # (calendar-based estimate; real next trading day will be filled in once
        # data becomes available)
        forecast_date = feature_date + pd.tseries.offsets.BDay(1)
        forecast_known = False
        _log(f"forecast_date estimated as {forecast_date.date()} "
             f"(no realised data for it yet)")
    else:
        forecast_known = True
        _log(f"forecast_date observed = {forecast_date.date()}")

    # Filter usable candidates (entry constraint mirroring backtest):
    #   - feature-day pct_chg not at limit band (10% main, 20% ChiNext/STAR)
    #   - σ_60 finite
    pool = df.head(candidate_pool).copy()
    names = _load_names()
    usable = []
    funnel = {'pool': len(pool), 'no_file': 0, 'no_close_row': 0,
              'st_capped': 0, 'at_limit_main': 0, 'at_limit_chinext': 0,
              'insufficient_history': 0, 'pass': 0,
              'st_in_pass': 0, 'nonst_in_pass': 0,
              'max_st_per_day': max_st_per_day}
    rejects = []   # keep a few examples for the dashboard
    band_cache = {}   # date → DataFrame of stk_limit (avoid re-reading)
    n_st_so_far = 0  # running tally for the cap

    for _, r in pool.iterrows():
        ts = r['ts_code']
        info = _load_close_at(ts, feature_date)
        if info is None:
            funnel['no_close_row'] += 1
            if len(rejects) < 60:
                rejects.append({'ts_code': ts, 'pred': float(r['pred']),
                                'reason': '当日无收盘数据 (停牌或新股)'})
            continue
        close, pct_chg_today = info

        # ST detection: authoritative roster from tushare namechange API
        # (stock_data/st_history.csv). Falls back to 5%-band heuristic only
        # if the roster is missing. The `band` is still computed for the
        # dashboard's 状态 tag display.
        band = _band_pct_at(ts, feature_date, band_cache)
        is_st_now = _is_st(ts, feature_date, band)

        # ST cap (mirrors backtest --max_st_per_day):
        #   max_st_per_day < 0 → no cap (admit all ST)
        #   max_st_per_day = 0 → exclude ST entirely
        #   max_st_per_day = N → admit at most N ST stocks; rest must be non-ST
        if is_st_now and max_st_per_day >= 0 and n_st_so_far >= max_st_per_day:
            funnel['st_capped'] += 1
            if len(rejects) < 60:
                cap_reason = (f'已达 ST 限额 {max_st_per_day}/日 '
                              if max_st_per_day > 0 else 'ST 名单 (限额=0,完全排除)')
                rejects.append({
                    'ts_code': ts, 'name': names.get(ts, ''), 'pred': float(r['pred']),
                    'pct_chg': float(pct_chg_today), 'band_pct': band, 'is_st': True,
                    'reason': cap_reason + '— tushare namechange 权威列表',
                })
            continue

        # Use the actual band from stk_limit if we have it, else fallback to
        # board-suffix heuristic. This way a stock that was once ChiNext but is
        # now *ST gets the right (5%) limit threshold.
        if band is not None:
            limit = band - 0.2  # leave a small buffer below the actual limit
        else:
            is_wide_band = ts.startswith(('300', '688'))
            limit = limit_pct_chinext if is_wide_band else limit_pct_main

        if abs(pct_chg_today) >= limit:
            key = 'at_limit_chinext' if (band and band >= 19) else 'at_limit_main'
            funnel[key] += 1
            if len(rejects) < 60:
                rejects.append({
                    'ts_code': ts, 'name': names.get(ts, ''), 'pred': float(r['pred']),
                    'pct_chg': float(pct_chg_today), 'band_pct': band,
                    'reason': f'已触及 ±{limit:.1f}% 涨跌停',
                })
            continue
        rets_strict = _load_pct_chg_window(ts, feature_date, cov_window)
        if np.isfinite(rets_strict).sum() < 20:
            funnel['insufficient_history'] += 1
            if len(rejects) < 60:
                rejects.append({'ts_code': ts, 'pred': float(r['pred']),
                                'reason': f'历史数据不足 (n={int(np.isfinite(rets_strict).sum())})'})
            continue
        sigma = float(np.nanstd(rets_strict, ddof=1))
        funnel['pass'] += 1
        if is_st_now:
            funnel['st_in_pass'] += 1
            n_st_so_far += 1
        else:
            funnel['nonst_in_pass'] += 1
        usable.append({
            'is_st': bool(is_st_now), 'band_pct': band,
            'ts_code': ts,
            'name':    names.get(ts, ''),
            'mu_rank': int(r['mu_rank']),
            'pred':    float(r['pred']),
            'sigma':   sigma,
            'close':   float(close),
            'pct_chg_feature_date': float(pct_chg_today),
            'pi_lo_80': float(r['pi_lo_80']) if 'pi_lo_80' in r else None,
            'pi_hi_80': float(r['pi_hi_80']) if 'pi_hi_80' in r else None,
            'prob_up':       float(r['prob_up'])       if 'prob_up'       in r else None,
            'prob_gt_3pct':  float(r['prob_gt_3pct'])  if 'prob_gt_3pct'  in r else None,
            'prob_lt_3pct':  float(r['prob_lt_3pct'])  if 'prob_lt_3pct'  in r else None,
            'rets_window':   rets_strict.tolist(),
        })
        if len(usable) >= top_k:
            break

    if not usable:
        # No survivors — common on heavy-limit-stop days. Return a payload
        # with empty portfolio so the dashboard can render a "cash" recommendation
        # rather than crashing the whole build.
        _log(f"no usable candidates after filtering — cash recommendation")
        return {
            'feature_date':   feature_date.strftime('%Y-%m-%d'),
            'forecast_date':  forecast_date.strftime('%Y-%m-%d'),
            'forecast_known': forecast_known,
            'stale_warning':  stale_warning,
            'n_candidates':   int(len(df)),
            'funnel':         funnel,
            'rejects':        rejects,
            'pred_stats':     {
                'mean': float(df['pred'].mean()),
                'std':  float(df['pred'].std(ddof=1)),
                'min':  float(df['pred'].min()),
                'max':  float(df['pred'].max()),
                'q05':  float(df['pred'].quantile(0.05)),
                'q95':  float(df['pred'].quantile(0.95)),
            },
            'top':              [],
            'bottom':           [],
            'portfolio':        [],
            'cov_diag':         [],
            'cov_offdiag_max_abs': 0.0,
            'cov_window':       cov_window,
            'risk_aversion':    risk_aversion,
            'execution_params': {
                'top_k':          top_k,
                'candidate_pool': candidate_pool,
                'limit_pct_main': limit_pct_main,
                'limit_pct_chinext': limit_pct_chinext,
                'entry_bps':      entry_bps,
                'exit_bps':       exit_bps,
                'max_pos_adv':    max_pos_adv,
                'impact_bps':     impact_bps,
                'initial':        initial,
                'cov_window':     cov_window,
                'risk_aversion':  risk_aversion,
                'max_st_per_day': max_st_per_day,
            },
            'total_alloc_cny':       0.0,
            'total_cost_cny':        0.0,
            'n_capped_by_adv':       0,
            'cash_recommendation':   True,
            'n_st_in_portfolio':     0,
        }
    _log(f"selected top-{len(usable)} candidates after filtering")

    # Build μ, Σ, solve QP
    mu = np.array([u['pred'] for u in usable], dtype=np.float64)
    R = np.column_stack([np.array(u['rets_window']) for u in usable])  # (T, K)
    cov = estimate_covariance(R, min_sigma=0.5)
    weights = markowitz_qp_weights(mu, cov,
                                    risk_aversion=risk_aversion,
                                    long_only=True)

    # Diagonal-only fallback weight for comparison
    sigmas_arr = np.array([u['sigma'] for u in usable], dtype=np.float64)
    weights_diag = markowitz_diagonal_weights(mu, sigmas_arr, long_only=True)

    # ── Position sizing — mirrors backtest/xgb_markowitz.py exactly ───────────
    #   1. Deployable budget = 0.95 × initial (cash buffer for slippage)
    #   2. Per-stock CNY alloc = budget × wᵢ
    #   3. Liquidity cap: alloc ≤ max_pos_adv × ADV₂₀  (same 5% cap as backtest)
    #   4. Gross shares = alloc / (close × (1 + entry_bps/1e4))
    #   5. Round down to 100-lot
    #   6. Market impact: extra cost = impact_bps × √(order_value / ADV)
    #   7. Final cost basis = close × shares × (1 + entry_bps/1e4 + impact)
    portfolio = []
    ec = entry_bps / 1e4
    xc = exit_bps  / 1e4
    budget = initial * 0.95
    weights_sum = float(np.sum(weights))
    cash_recommendation = (weights_sum < 1e-9)  # QP says no positive-edge candidate
    for u, w_qp, w_dg in zip(usable, weights, weights_diag):
        w = float(w_qp)
        # Even when QP weight is 0 we keep the candidate row so users can see
        # what survived the filter. shares/cny will be 0 in that case.
        alloc = budget * w
        adv20 = _load_adv20(u['ts_code'], feature_date, window=20)
        capped_by_adv = False
        if adv20 is not None and adv20 > 0:
            adv_cap = adv20 * max_pos_adv
            if alloc > adv_cap:
                alloc = adv_cap
                capped_by_adv = True
        gross_shares = alloc / (u['close'] * (1.0 + ec))
        shares = int(gross_shares // 100) * 100
        if shares <= 0:
            portfolio.append({
                **u, 'weight_qp': w, 'weight_diag': float(w_dg),
                'cny_alloc': alloc, 'cny_alloc_uncapped': budget * w,
                'shares': 0, 'cost_basis': 0.0, 'impact_bps_est': 0.0,
                'adv20_cny': adv20, 'capped_by_adv': capped_by_adv,
                'is_zero_weight': True,
            })
            continue
        order_val = u['close'] * shares
        impact = 0.0
        if impact_bps > 0 and adv20 is not None and adv20 > 0:
            impact = (impact_bps / 1e4) * np.sqrt(order_val / adv20)
        cost_basis = order_val * (1.0 + ec + impact)
        portfolio.append({
            **u,
            'weight_qp':         w,
            'weight_diag':       float(w_dg),
            'cny_alloc_uncapped': budget * w,
            'cny_alloc':         alloc,        # post-ADV cap
            'shares':            shares,
            'cost_basis':        cost_basis,   # what you actually spend
            'impact_bps_est':    impact * 1e4, # in bps
            'adv20_cny':         adv20,
            'capped_by_adv':     capped_by_adv,
        })

    # Top 30 by pred for dashboard table (richer than just the 10 selected)
    full_top_table = []
    for _, r in df.head(30).iterrows():
        full_top_table.append({
            'ts_code': r['ts_code'],
            'name':    names.get(r['ts_code'], ''),
            'pred':    float(r['pred']),
            'pi_lo_80': float(r['pi_lo_80']) if 'pi_lo_80' in r else None,
            'pi_hi_80': float(r['pi_hi_80']) if 'pi_hi_80' in r else None,
            'prob_up':      float(r['prob_up']) if 'prob_up' in r else None,
            'prob_gt_3pct': float(r['prob_gt_3pct']) if 'prob_gt_3pct' in r else None,
            'prob_lt_3pct': float(r['prob_lt_3pct']) if 'prob_lt_3pct' in r else None,
        })

    # Bottom 30 (worst predicted)
    bottom = df.tail(30).sort_values('pred', ascending=True)
    full_bottom_table = []
    for _, r in bottom.iterrows():
        full_bottom_table.append({
            'ts_code': r['ts_code'],
            'name':    names.get(r['ts_code'], ''),
            'pred':    float(r['pred']),
            'pi_lo_80': float(r['pi_lo_80']) if 'pi_lo_80' in r else None,
            'pi_hi_80': float(r['pi_hi_80']) if 'pi_hi_80' in r else None,
            'prob_up':      float(r['prob_up']) if 'prob_up' in r else None,
        })

    payload = {
        'feature_date':   feature_date.strftime('%Y-%m-%d'),
        'forecast_date':  forecast_date.strftime('%Y-%m-%d'),
        'forecast_known': forecast_known,
        'stale_warning':  stale_warning,
        'n_candidates':   int(len(df)),
        'funnel':         funnel,
        'rejects':        rejects,
        'pred_stats': {
            'mean': float(df['pred'].mean()),
            'std':  float(df['pred'].std(ddof=1)),
            'min':  float(df['pred'].min()),
            'max':  float(df['pred'].max()),
            'q05':  float(df['pred'].quantile(0.05)),
            'q95':  float(df['pred'].quantile(0.95)),
        },
        'top':       full_top_table,
        'bottom':    full_bottom_table,
        'portfolio': portfolio,
        'cov_diag':  np.diag(cov).tolist(),
        'cov_offdiag_max_abs': float(np.abs(cov - np.diag(np.diag(cov))).max()),
        'cov_window': cov_window,
        'risk_aversion': risk_aversion,
        # Execution-model parameters (mirror backtest/xgb_markowitz.py defaults)
        'execution_params': {
            'top_k':          top_k,
            'candidate_pool': candidate_pool,
            'limit_pct_main': limit_pct_main,
            'limit_pct_chinext': limit_pct_chinext,
            'entry_bps':      entry_bps,
            'exit_bps':       exit_bps,
            'max_pos_adv':    max_pos_adv,
            'impact_bps':     impact_bps,
            'initial':        initial,
            'cov_window':     cov_window,
            'risk_aversion':  risk_aversion,
            'max_st_per_day': max_st_per_day,
        },
        # Sums for the dashboard cards
        'total_alloc_cny': float(sum(p.get('cny_alloc', 0) for p in portfolio)),
        'total_cost_cny':  float(sum(p.get('cost_basis', 0) for p in portfolio)),
        'n_capped_by_adv': int(sum(1 for p in portfolio if p.get('capped_by_adv'))),
        'cash_recommendation': cash_recommendation,
        'n_st_in_portfolio':   int(sum(1 for p in portfolio if p.get('is_st'))),
    }
    _log(f"forecast={payload['forecast_date']}, "
         f"top-K weights sum to {sum(p['weight_qp'] for p in portfolio):.4f}")
    return payload


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds_csv',
                    default=str(ROOT / 'stock_predictions_xgb.csv'),
                    help='Live predictions CSV (output of xgbmodel.predict)')
    ap.add_argument('--out', default=str(ROOT / 'dashboard' / 'live_prediction.json'))
    ap.add_argument('--top_k', type=int, default=10)
    args = ap.parse_args()

    payload = build_live_payload(Path(args.preds_csv), top_k=args.top_k)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'[live_prediction] wrote {args.out}')
