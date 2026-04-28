"""Technical and market-factor library for BTC price prediction.

Single-asset technicals: RSI, MACD, Bollinger, ATR, Stochastic, ADX/DI,
CCI, Williams %R, OBV-diff (stationary), realized volatility, multi-horizon
log returns.

Patterns: W-bottom (double bottom) and M-top (double top) detection on a
rolling window — confirmation requires neckline breakout/breakdown.

Note on OBV: raw cumulative OBV is non-stationary and has caused multi-sigma
distribution drift in this project before. Use `obv_diff` (one-step change)
instead, never the cumulative series directly.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_dn = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return macd_line, sig, hist


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, pctb, bandwidth


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def stoch(high, low, close, k_period=14, d_period=3, smooth=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def adx(high, low, close, period: int = 14):
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    atr_v = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_v = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_v, plus_di, minus_di


def cci(high, low, close, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(high, low, close, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def obv_diff(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Stationary OBV: 1-step change in cumulative on-balance volume."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv.diff()


def realized_vol(close: pd.Series, period: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(period).std() * np.sqrt(period)


def aggregate_5m_to_daily(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-minute klines to daily-level features.

    Input: 5m kline df with columns open/high/low/close/volume/taker_buy_base
    and index `open_time` (UTC tz-aware).

    Returns daily-indexed DataFrame with:
        intraday_rv         realized vol from 5m log returns
        intraday_bipower    jump-robust bipower variation
        intraday_jump       rv - bipower (jump component)
        intraday_skew       skew of 5m returns
        intraday_kurt       excess kurtosis
        intraday_range_pct  (high - low) / open
        intraday_efficiency |close - open| / (high - low)
        intraday_taker_ratio sum(taker_buy_base) / sum(volume)
        intraday_vol_imbal  (taker_buy - taker_sell) / volume
        intraday_drawup     max(close)/open - 1
        intraday_drawdown   min(close)/open - 1
        intraday_vol_conc   Herfindahl index of 5m volume distribution
    """
    df = df_5m.copy()
    if "open_time" in df.columns:
        df = df.set_index("open_time")
    log_ret = np.log(df["close"]).diff()
    df["log_ret"] = log_ret
    df["abs_ret"] = log_ret.abs()
    df["bp_term"] = df["abs_ret"] * df["abs_ret"].shift(1)

    date_key = df.index.floor("D")
    grp = df.groupby(date_key)

    rv = grp["log_ret"].apply(lambda s: float(np.sqrt((s ** 2).sum())))
    bp = grp["bp_term"].sum() * (np.pi / 2)
    bp = np.sqrt(bp)
    skew = grp["log_ret"].skew()
    kurt = grp["log_ret"].apply(lambda s: float(s.kurt()))

    pos_sq = log_ret.clip(lower=0) ** 2
    neg_sq = log_ret.clip(upper=0) ** 2
    rv_up = pos_sq.groupby(date_key).sum().pow(0.5)
    rv_dn = neg_sq.groupby(date_key).sum().pow(0.5)
    semi_skew = (rv_up.pow(2) - rv_dn.pow(2)) / (rv ** 2).replace(0, np.nan)

    abs_med = log_ret.abs().rolling(288, min_periods=50).median()
    jump_flag = (log_ret.abs() > 4 * abs_med).astype(int)
    jump_count = jump_flag.groupby(date_key).sum()

    daily_high = grp["high"].max()
    daily_low = grp["low"].min()
    daily_open = grp["open"].first()
    daily_close = grp["close"].last()
    range_pct = (daily_high - daily_low) / daily_open.replace(0, np.nan)
    efficiency = (daily_close - daily_open).abs() / (daily_high - daily_low).replace(0, np.nan)

    total_vol = grp["volume"].sum()
    taker_buy = grp["taker_buy_base"].sum()
    taker_ratio = taker_buy / total_vol.replace(0, np.nan)
    vol_imbal = (2 * taker_buy - total_vol) / total_vol.replace(0, np.nan)

    drawup = grp.apply(lambda g: float(g["close"].max() / g["open"].iloc[0] - 1)
                        if len(g) and g["open"].iloc[0] > 0 else np.nan,
                        include_groups=False)
    drawdown = grp.apply(lambda g: float(g["close"].min() / g["open"].iloc[0] - 1)
                          if len(g) and g["open"].iloc[0] > 0 else np.nan,
                          include_groups=False)

    def herf(g):
        v = g["volume"].to_numpy(dtype=float)
        s = v.sum()
        if s <= 0 or len(v) == 0:
            return np.nan
        p = v / s
        return float((p ** 2).sum() * len(v))

    vol_conc = grp.apply(herf, include_groups=False)
    jump = (rv ** 2 - bp ** 2).clip(lower=0).pow(0.5)

    out = pd.DataFrame({
        "intraday_rv": rv,
        "intraday_bipower": bp,
        "intraday_jump": jump,
        "intraday_rv_up": rv_up,
        "intraday_rv_dn": rv_dn,
        "intraday_semi_skew": semi_skew,
        "intraday_jump_count": jump_count,
        "intraday_skew": skew,
        "intraday_kurt": kurt,
        "intraday_range_pct": range_pct,
        "intraday_efficiency": efficiency,
        "intraday_taker_ratio": taker_ratio,
        "intraday_vol_imbal": vol_imbal,
        "intraday_drawup": drawup,
        "intraday_drawdown": drawdown,
        "intraday_vol_conc": vol_conc,
    })
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "open_time"
    return out


def detect_double_bottom_top(close: pd.Series, window: int = 60,
                             prominence_pct: float = 0.02,
                             tol_pct: float = 0.03,
                             min_separation: int = 5):
    """Rolling W-bottom (double bottom) and M-top (double top) detection.

    On each bar, looks back `window` bars for the two most recent prominent
    extrema. Confirmation requires the current bar to break beyond the
    intervening peak (W) or trough (M).

    Returns:
        (w_bottom, m_top) — 0/1 int8 series indexed like `close`.
    """
    arr = close.to_numpy(dtype=float)
    n = len(arr)
    w = np.zeros(n, dtype=np.int8)
    m = np.zeros(n, dtype=np.int8)
    for i in range(window, n):
        sub = arr[i - window:i + 1]
        scale = sub.mean()
        prom = prominence_pct * scale

        lows, _ = find_peaks(-sub, prominence=prom)
        if len(lows) >= 2:
            l1, l2 = lows[-2], lows[-1]
            if l2 - l1 >= min_separation:
                between = sub[l1:l2 + 1]
                peak = float(between.max())
                base = float(sub[l1])
                if (peak - base) / base > prominence_pct and \
                   abs(sub[l1] - sub[l2]) / base < tol_pct and \
                   sub[-1] > peak:
                    w[i] = 1

        highs, _ = find_peaks(sub, prominence=prom)
        if len(highs) >= 2:
            h1, h2 = highs[-2], highs[-1]
            if h2 - h1 >= min_separation:
                between = sub[h1:h2 + 1]
                trough = float(between.min())
                base = float(sub[h1])
                if (base - trough) / base > prominence_pct and \
                   abs(sub[h1] - sub[h2]) / base < tol_pct and \
                   sub[-1] < trough:
                    m[i] = 1

    return (pd.Series(w, index=close.index, name="w_bottom"),
            pd.Series(m, index=close.index, name="m_top"))
