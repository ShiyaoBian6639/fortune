"""
Per-stock technical feature engineering for xgbmodel.

Input: a DataFrame for ONE stock sorted by ascending trade_date with at least
  columns [open, high, low, close, pct_chg, vol, amount, pre_close].
Output: the same DataFrame with technical feature columns appended.

All features are strictly causal — they only use information available as of
the row's trade_date.
"""

from typing import List

import numpy as np
import pandas as pd

from .config import MA_WINDOWS, VOL_WINDOWS, MOMENTUM_WINDOWS, RETURN_LAGS


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock price/volume/volatility/TA features.

    The caller is responsible for passing a DataFrame sorted by trade_date
    ascending and having a RangeIndex.
    """
    out = {}

    close  = df['close'].astype('float32')
    high   = df['high'].astype('float32')
    low    = df['low'].astype('float32')
    openp  = df['open'].astype('float32')
    pre    = df['pre_close'].astype('float32')
    vol    = df['vol'].astype('float32')
    amount = df['amount'].astype('float32')
    pct    = df['pct_chg'].astype('float32')

    # Intraday-shape features
    out['oc_ratio']     = _safe_div(close - openp,  pre).astype('float32')
    out['hl_ratio']     = _safe_div(high  - low,    pre).astype('float32')
    out['uppershadow']  = _safe_div(high  - np.maximum(close, openp), pre).astype('float32')
    out['lowershadow'] = _safe_div(np.minimum(close, openp) - low,   pre).astype('float32')
    out['overnight_gap'] = _safe_div(openp - pre, pre).astype('float32')

    # Log-return for numerical stability
    log_ret = np.log(_safe_div(close, pre)).astype('float32')
    out['log_ret'] = log_ret

    # Lagged returns (skip lag 0 — that's the raw pct_chg already available)
    for k in RETURN_LAGS:
        out[f'ret_lag_{k}'] = pct.shift(k).astype('float32')

    # Momentum over N days = cumulative log-return
    for w in MOMENTUM_WINDOWS:
        out[f'momentum_{w}'] = log_ret.rolling(w).sum().astype('float32')

    # Moving averages and close/ma ratios
    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()
        out[f'close_ma_{w}_ratio'] = _safe_div(close, ma).astype('float32')
        out[f'ma_{w}_slope']       = _safe_div(ma - ma.shift(w), ma).astype('float32')

    # Volatility (std of daily returns)
    for w in VOL_WINDOWS:
        out[f'vol_pct_{w}'] = pct.rolling(w).std().astype('float32')

    # Rolling range normalized (Parkinson-like)
    for w in VOL_WINDOWS:
        hl = np.log(_safe_div(high, low)).astype('float32')
        out[f'parkinson_{w}'] = hl.rolling(w).mean().astype('float32')

    # Distance from rolling high/low
    for w in [20, 60]:
        hh = high.rolling(w).max()
        ll = low.rolling(w).min()
        out[f'dist_from_high_{w}'] = _safe_div(close - hh, hh).astype('float32')
        out[f'dist_from_low_{w}']  = _safe_div(close - ll, ll).astype('float32')

    # Volume / amount dynamics
    for w in [5, 20]:
        vol_ma   = vol.rolling(w).mean()
        amt_ma   = amount.rolling(w).mean()
        out[f'vol_ratio_{w}'] = _safe_div(vol,    vol_ma).astype('float32')
        out[f'amt_ratio_{w}'] = _safe_div(amount, amt_ma).astype('float32')

    out['vol_pct_chg']   = vol.pct_change().clip(-10, 10).astype('float32')
    out['amount_pct_chg'] = amount.pct_change().clip(-10, 10).astype('float32')

    # RSI(14)
    delta = close.diff()
    up    = delta.clip(lower=0.0)
    down  = -delta.clip(upper=0.0)
    roll  = 14
    avg_gain = up.ewm(alpha=1 / roll, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / roll, adjust=False).mean()
    rs = _safe_div(avg_gain, avg_loss)
    out['rsi_14'] = (100 - 100 / (1 + rs)).astype('float32')

    # MACD(12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd       = ema12 - ema26
    signal     = macd.ewm(span=9, adjust=False).mean()
    out['macd']        = macd.astype('float32')
    out['macd_signal'] = signal.astype('float32')
    out['macd_hist']   = (macd - signal).astype('float32')

    # Bollinger %B (20, 2)
    ma20  = close.rolling(20).mean()
    sd20  = close.rolling(20).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    out['bbpct_20'] = _safe_div(close - lower, upper - lower).astype('float32')

    # ATR(14) normalized
    tr = pd.concat([
        high - low,
        (high - pre).abs(),
        (low  - pre).abs(),
    ], axis=1).max(axis=1)
    out['atr_14_pct'] = _safe_div(tr.rolling(14).mean(), close).astype('float32')

    # OBV differenced (raw cumulative OBV is non-stationary — see CLAUDE.md)
    direction = np.sign(pct).astype('float32')
    obv_flow  = (vol * direction).astype('float32')
    out['obv_flow_ma5']  = obv_flow.rolling(5).mean().astype('float32')
    out['obv_flow_ma20'] = obv_flow.rolling(20).mean().astype('float32')

    return df.assign(**out)


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived from trade_date (datetime)."""
    td = df['trade_date']
    return df.assign(
        dow           = td.dt.dayofweek.astype('int8'),
        month         = td.dt.month.astype('int8'),
        day_of_month  = td.dt.day.astype('int8'),
        quarter       = td.dt.quarter.astype('int8'),
        is_month_end  = td.dt.is_month_end.astype('int8'),
        is_month_start = td.dt.is_month_start.astype('int8'),
        is_quarter_end = td.dt.is_quarter_end.astype('int8'),
    )


def price_feature_columns() -> List[str]:
    """Return the deterministic list of price-derived feature names (for reference)."""
    names = [
        'oc_ratio', 'hl_ratio', 'uppershadow', 'lowershadow', 'overnight_gap', 'log_ret',
        'vol_pct_chg', 'amount_pct_chg',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bbpct_20', 'atr_14_pct',
        'obv_flow_ma5', 'obv_flow_ma20',
    ]
    for k in RETURN_LAGS:
        names.append(f'ret_lag_{k}')
    for w in MOMENTUM_WINDOWS:
        names.append(f'momentum_{w}')
    for w in MA_WINDOWS:
        names += [f'close_ma_{w}_ratio', f'ma_{w}_slope']
    for w in VOL_WINDOWS:
        names += [f'vol_pct_{w}', f'parkinson_{w}']
    for w in [20, 60]:
        names += [f'dist_from_high_{w}', f'dist_from_low_{w}']
    for w in [5, 20]:
        names += [f'vol_ratio_{w}', f'amt_ratio_{w}']
    return names


def calendar_feature_columns() -> List[str]:
    return [
        'dow', 'month', 'day_of_month', 'quarter',
        'is_month_end', 'is_month_start', 'is_quarter_end',
    ]
