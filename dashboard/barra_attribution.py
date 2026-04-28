"""
Barra-style risk attribution for the XGB Markowitz long-only backtest.

Style factors (5):
    SIZE        log(circ_mv)         市值 — 大盘 vs 小盘
    VALUE       1 / pb               价值 — 估值便宜 vs 昂贵
    MOMENTUM    63d cumulative ret   动量 — 中期趋势强度
    VOLATILITY  60d std(pct_chg)     波动率 — 高波 vs 低波
    LIQUIDITY   60d mean turnover_f  流动性 — 高频交易 vs 沉睡股

Pipeline:
1. Build (T × N) panel of standardized factor exposures (z-score within each
   trade_date, after 1%/99% winsorisation).
2. Reconstruct daily strategy holdings from trades_qp.csv (positions that are
   open and untouched on day t).
3. Strategy daily exposure E_p,k,t = mean z-score of held stocks.
4. Daily factor return f_k,t = mean(top-quintile r) − mean(bottom-quintile r),
   where quintiles are sorted by exposure on day t and r is t→t+1 return.
5. Attribution: daily factor PnL_k,t ≈ E_p,k,t · f_k,t · (effective gross
   exposure ≈ invested / NAV).
6. Residual α = strategy daily return − Σ_k factor_PnL_k.

Output: a JSON-serialisable dict embedded in the combined dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'
DAILY_BASIC_DIR = DATA_DIR / 'daily_basic'

FACTOR_NAMES = ['SIZE', 'VALUE', 'MOMENTUM', 'VOLATILITY', 'LIQUIDITY']
FACTOR_LABEL = {
    'SIZE':       '规模 (log 流通市值)',
    'VALUE':      '价值 (1 / PB)',
    'MOMENTUM':   '动量 (63日累计收益)',
    'VOLATILITY': '波动率 (60日 σ)',
    'LIQUIDITY': '流动性 (60日均换手率)',
}


def _log(msg: str) -> None:
    print(f"[barra] {msg}", flush=True)


# ─── Data loaders ─────────────────────────────────────────────────────────────
def _load_daily_basic_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Concatenate daily_basic_*.csv files in the [start, end] window."""
    files = sorted(os.listdir(DAILY_BASIC_DIR))
    rows = []
    for f in files:
        if not f.startswith('daily_basic_') or not f.endswith('.csv'):
            continue
        date_str = f.replace('daily_basic_', '').replace('.csv', '')
        try:
            d = pd.Timestamp(date_str)
        except Exception:
            continue
        if d < start or d > end:
            continue
        df = pd.read_csv(DAILY_BASIC_DIR / f,
                         usecols=['ts_code', 'trade_date',
                                  'close', 'turnover_rate_f', 'pb', 'circ_mv'])
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out['trade_date'] = pd.to_datetime(out['trade_date'].astype(str))
    _log(f"daily_basic: {len(out):,} rows over {out['trade_date'].nunique()} dates")
    return out


def _load_pct_chg_panel(ts_codes: List[str],
                         start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """For each ts_code, load (trade_date, pct_chg) within window. Returns long-form
    DataFrame [ts_code, trade_date, pct_chg]."""
    rows = []
    for ts_code in ts_codes:
        code, suffix = ts_code.split('.')
        sub = 'sh' if suffix.upper() == 'SH' else 'sz'
        fp = DATA_DIR / sub / f'{code}.csv'
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, usecols=['trade_date', 'pct_chg'])
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df[(df['trade_date'] >= start) & (df['trade_date'] <= end)]
            df['ts_code'] = ts_code
            rows.append(df[['ts_code', 'trade_date', 'pct_chg']])
        except Exception:
            continue
    out = pd.concat(rows, ignore_index=True)
    return out


# ─── Exposure construction ────────────────────────────────────────────────────
def _winsor_zscore(s: pd.Series) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    mu, sd = s.mean(), s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - mu) / sd


def _build_exposure_panel(daily_basic: pd.DataFrame,
                          rets: pd.DataFrame) -> pd.DataFrame:
    """Returns long-form panel:
       [ts_code, trade_date, SIZE, VALUE, MOMENTUM, VOLATILITY, LIQUIDITY, fwd_ret]
       where each factor is per-day cross-sectional z-score and fwd_ret is the
       same-day pct_chg (used for factor-return regression in next step).
    """
    # Wide-form panel for momentum / vol / liquidity rolling statistics
    rets = rets.sort_values(['ts_code', 'trade_date'])
    rets['mom_63']  = (rets.groupby('ts_code')['pct_chg']
                          .transform(lambda x: x.shift(1).rolling(63, min_periods=20).sum()))
    rets['vol_60']  = (rets.groupby('ts_code')['pct_chg']
                          .transform(lambda x: x.shift(1).rolling(60, min_periods=20).std(ddof=1)))

    db = daily_basic.copy()
    db['log_circ_mv'] = np.log1p(db['circ_mv'].astype(float))
    db['inv_pb']      = 1.0 / db['pb'].replace(0, np.nan).astype(float)
    db['turnover_60'] = (db.sort_values(['ts_code', 'trade_date'])
                           .groupby('ts_code')['turnover_rate_f']
                           .transform(lambda x: x.shift(1).rolling(60, min_periods=20).mean()))

    panel = db.merge(rets[['ts_code', 'trade_date', 'pct_chg', 'mom_63', 'vol_60']],
                     on=['ts_code', 'trade_date'], how='inner')
    panel = panel.rename(columns={
        'log_circ_mv': 'SIZE',
        'inv_pb':      'VALUE',
        'mom_63':      'MOMENTUM',
        'vol_60':      'VOLATILITY',
        'turnover_60': 'LIQUIDITY',
    })

    # Cross-sectional z-score per trade_date
    for fac in FACTOR_NAMES:
        panel[fac] = (panel.groupby('trade_date')[fac]
                            .transform(_winsor_zscore))

    keep = ['ts_code', 'trade_date', 'pct_chg'] + FACTOR_NAMES
    panel = panel[keep].dropna(subset=FACTOR_NAMES)
    _log(f"exposure panel: {len(panel):,} rows × {len(FACTOR_NAMES)} factors")
    return panel


# ─── Factor returns: long-short quintile spread ───────────────────────────────
def _factor_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (factor, trade_date), compute factor return = mean pct_chg of
    top quintile − mean pct_chg of bottom quintile. Returns wide DataFrame
    indexed by trade_date with one column per factor (in % units)."""
    rows = []
    for date, day in panel.groupby('trade_date'):
        row = {'trade_date': date}
        for fac in FACTOR_NAMES:
            sub = day[[fac, 'pct_chg']].dropna()
            if len(sub) < 50:
                row[fac] = np.nan
                continue
            q = sub[fac].quantile([0.2, 0.8]).values
            top = sub.loc[sub[fac] >= q[1], 'pct_chg'].mean()
            bot = sub.loc[sub[fac] <= q[0], 'pct_chg'].mean()
            row[fac] = top - bot
        rows.append(row)
    out = pd.DataFrame(rows).set_index('trade_date').sort_index()
    return out


# ─── Strategy holdings reconstruction ─────────────────────────────────────────
def _reconstruct_holdings(trades: pd.DataFrame,
                          all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each (trade_date, ts_code), 1 if held at close that day else 0.
    A trade contributes to the long position on dates [entry_date, exit_date−1].
    """
    rows = []
    for _, t in trades.iterrows():
        entry, exit_ = t['entry_date'], t['exit_date']
        # Held at close from entry day through the day BEFORE exit (sold during exit day)
        days = all_dates[(all_dates >= entry) & (all_dates < exit_)]
        for d in days:
            rows.append({'trade_date': d, 'ts_code': t['ts_code']})
    if not rows:
        return pd.DataFrame(columns=['trade_date', 'ts_code'])
    return pd.DataFrame(rows)


def _strategy_exposures(holdings: pd.DataFrame,
                         panel: pd.DataFrame) -> pd.DataFrame:
    """For each trade_date, average exposure across active holdings (equal-weight
    proxy — the QP weights are close to uniform under top-K=10)."""
    if holdings.empty:
        return pd.DataFrame(columns=['trade_date'] + FACTOR_NAMES)
    j = holdings.merge(panel[['ts_code', 'trade_date'] + FACTOR_NAMES],
                       on=['ts_code', 'trade_date'], how='left')
    out = (j.groupby('trade_date')[FACTOR_NAMES].mean()
              .reset_index().sort_values('trade_date'))
    return out


# ─── Attribution ──────────────────────────────────────────────────────────────
def _attribution(strat_exp: pd.DataFrame,
                 fac_rets: pd.DataFrame,
                 equity: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Daily PnL contribution from each factor:
        contrib_k,t = exposure_k,t-1 · factor_return_k,t · invested_share_t

    invested_share_t = invested_value_t / NAV_t accounts for cash drag.
    """
    eq = equity.copy().set_index('trade_date').sort_index()
    eq['invested_share'] = (eq['invested'] / eq['nav']).clip(0, 1)
    eq['nav_ret'] = eq['nav'].pct_change()

    exp = strat_exp.set_index('trade_date').sort_index().reindex(eq.index, method='ffill')
    rets = fac_rets.reindex(eq.index, method='ffill') / 100.0   # convert % → fraction

    # Use yesterday's exposure × today's factor return
    exp_lag = exp.shift(1)

    contribs = {}
    for fac in FACTOR_NAMES:
        contribs[fac] = (exp_lag[fac] * rets[fac] * eq['invested_share']).fillna(0.0)

    contrib_df = pd.DataFrame(contribs, index=eq.index)
    explained = contrib_df.sum(axis=1)
    alpha = (eq['nav_ret'] - explained).fillna(0.0)

    cum_factor = (1.0 + contrib_df).cumprod() - 1.0
    cum_alpha  = (1.0 + alpha).cumprod() - 1.0
    cum_total  = (1.0 + eq['nav_ret'].fillna(0.0)).cumprod() - 1.0

    summary = {
        'total_return':   float(cum_total.iloc[-1]),
        'alpha_cum':      float(cum_alpha.iloc[-1]),
        'factor_cum':     {fac: float(cum_factor[fac].iloc[-1]) for fac in FACTOR_NAMES},
        'mean_exposure':  {fac: float(exp[fac].mean()) for fac in FACTOR_NAMES},
        'std_exposure':   {fac: float(exp[fac].std(ddof=1)) for fac in FACTOR_NAMES},
        'mean_factor_ret':{fac: float(rets[fac].mean() * 252) for fac in FACTOR_NAMES},
    }
    return contrib_df.assign(alpha=alpha), summary


# ─── Public entry point ───────────────────────────────────────────────────────
def run_barra(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    start = equity['trade_date'].min()
    end   = equity['trade_date'].max()
    _log(f"running Barra attribution {start.date()} → {end.date()}")

    ts_codes = trades['ts_code'].unique().tolist()

    daily_basic = _load_daily_basic_window(
        start - pd.Timedelta(days=120), end + pd.Timedelta(days=10))
    rets = _load_pct_chg_panel(
        ts_codes, start - pd.Timedelta(days=120), end + pd.Timedelta(days=10))
    daily_basic = daily_basic[daily_basic['ts_code'].isin(set(ts_codes))]

    panel = _build_exposure_panel(daily_basic, rets)
    fac_rets = _factor_returns(panel)
    holdings = _reconstruct_holdings(trades, pd.DatetimeIndex(equity['trade_date']))
    strat_exp = _strategy_exposures(holdings, panel)
    contrib, summary = _attribution(strat_exp, fac_rets, equity)

    # Compact JSON shape
    dates_str = [d.strftime('%Y-%m-%d') for d in strat_exp['trade_date']]
    out = {
        'factors':   FACTOR_NAMES,
        'labels':    FACTOR_LABEL,
        'dates':     dates_str,
        'exposures': {fac: strat_exp[fac].fillna(0.0).tolist() for fac in FACTOR_NAMES},
        'fac_rets_dates': [d.strftime('%Y-%m-%d') for d in fac_rets.index],
        'fac_rets':       {fac: fac_rets[fac].fillna(0.0).tolist() for fac in FACTOR_NAMES},
        'contrib_dates':  [d.strftime('%Y-%m-%d') for d in contrib.index],
        'contrib_cum':    {fac: ((1.0 + contrib[fac]).cumprod() - 1.0).tolist()
                           for fac in FACTOR_NAMES},
        'alpha_cum':      ((1.0 + contrib['alpha']).cumprod() - 1.0).tolist(),
        'summary':        summary,
    }
    _log(f"final cum-alpha = {summary['alpha_cum']*100:+.1f}%, "
         f"factors = {sum(summary['factor_cum'].values())*100:+.1f}%, "
         f"total = {summary['total_return']*100:+.1f}%")
    return out
