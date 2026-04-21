"""
Data loading and feature engineering with multiprocessing support.
Optimized with numba JIT compilation for computational bottlenecks.
"""

import os
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    CHANGE_BUCKETS, NUM_CLASSES, FEATURE_COLUMNS,
    FORWARD_WINDOWS, NUM_HORIZONS,
    RELATIVE_CHANGE_BUCKETS, NUM_RELATIVE_CLASSES,
    MARKET_CONTEXT_FEATURES, INDEX_MEMBERSHIP_FEATURES,
    TUSHARE_TOKEN,
    SPRING_FESTIVAL_DATES, QINGMING_DATES, DRAGON_BOAT_DATES,
    MID_AUTUMN_DATES, DOUBLE_NINTH_DATES, WINTER_SOLSTICE_DATES,
    QIXI_DATES, LABA_DATES, DAILY_BASIC_COLUMNS,
    CS_NORMALIZE_TECH_FEATURES,
    _FUTURE_FEAT_IDX,
)

# Import numba-optimized functions
from .numba_optimizations import (
    compute_holiday_distances,
    compute_cci,
    detect_w_bottom_numba,
    detect_m_top_numba,
    detect_patterns_multi_window_numba
)

# ============================================================================
# Cross-sectional normalization
# ============================================================================

# Per-stock valuation features that drift with market regime.
# Absolute PE/PB of 50 in a bull market ≠ PE/PB of 50 in a bear market (different context).
# Normalising each stock's value against the daily cross-section (all stocks on that day)
# converts absolute levels into relative ranks, which are regime-invariant.
CS_NORMALIZE_FEATURES = ['pe_ttm', 'pe', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm']


def compute_daily_cs_stats(daily_basic_dict: Dict[str, 'pd.DataFrame']) -> Dict[int, Dict[str, tuple]]:
    """
    Compute per-date cross-section median and std for valuation features.

    Uses the already-grouped daily_basic_dict (no extra file I/O).

    Returns:
        {date_int (YYYYMMDD): {feature: (median, std)}}
    """
    from collections import defaultdict

    # Accumulate values: {date_int: {feature: [value, ...]}}
    accum: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for _bare_code, basic_df in daily_basic_dict.items():
        if 'trade_date' not in basic_df.columns:
            continue
        dates = basic_df['trade_date']
        if pd.api.types.is_datetime64_any_dtype(dates):
            date_ints = (dates.dt.year * 10000
                         + dates.dt.month * 100
                         + dates.dt.day).values.astype(np.int32)
        else:
            dt = pd.to_datetime(dates.astype(str))
            date_ints = (dt.dt.year * 10000
                         + dt.dt.month * 100
                         + dt.dt.day).values.astype(np.int32)

        for feat in CS_NORMALIZE_FEATURES:
            if feat not in basic_df.columns:
                continue
            vals = basic_df[feat].values.astype('float32')
            for d, v in zip(date_ints, vals):
                # Exclude non-positive values (negative PE = loss-making; 0 = missing)
                if np.isfinite(v) and v > 0:
                    accum[int(d)][feat].append(float(v))

    # Convert to (median, std) pairs
    cs_stats: Dict[int, Dict[str, tuple]] = {}
    for date_int, feat_dict in accum.items():
        cs_stats[date_int] = {}
        for feat, values in feat_dict.items():
            arr = np.array(values, dtype='float32')
            cs_stats[date_int][feat] = (
                float(np.median(arr)),
                max(float(arr.std()), 1e-6),
            )

    return cs_stats


def apply_cs_normalization(
    df: 'pd.DataFrame',
    cs_stats: Dict[int, Dict[str, tuple]],
    features: Optional[List[str]] = None,
) -> 'pd.DataFrame':
    """
    Replace feature values with their daily cross-section z-score.

    After this transform, each feature value means "how many σ above/below the
    cross-section mean this stock is today" — invariant to market-regime levels.
    Falls back to 0.0 if no cross-section data for a given date.

    Args:
        features: feature names to normalize. Defaults to CS_NORMALIZE_FEATURES
                  (valuation ratios). Pass CS_NORMALIZE_TECH_FEATURES for
                  technical indicators.
    """
    if not cs_stats:
        return df
    if features is None:
        features = CS_NORMALIZE_FEATURES

    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df = df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    date_ints = (df['trade_date'].dt.year * 10000
                 + df['trade_date'].dt.month * 100
                 + df['trade_date'].dt.day).values.astype(np.int32)

    for feat in features:
        if feat not in df.columns:
            continue
        vals = df[feat].values.astype('float64')

        medians = np.array([
            cs_stats[int(d)][feat][0] if int(d) in cs_stats and feat in cs_stats[int(d)] else np.nan
            for d in date_ints
        ], dtype='float64')
        stds = np.array([
            cs_stats[int(d)][feat][1] if int(d) in cs_stats and feat in cs_stats[int(d)] else np.nan
            for d in date_ints
        ], dtype='float64')

        valid = np.isfinite(medians) & (stds > 1e-6)
        normalized = np.where(valid, (vals - medians) / stds, 0.0)
        df[feat] = normalized.astype('float32')

    return df


def compute_cross_section_tech_stats(
    stock_files: List[Tuple[str, str]],
    min_data_points: int = 100,
) -> Dict[int, Dict[str, tuple]]:
    """
    Lightweight first pass over raw price/volume files to build per-date
    cross-sectional statistics (mean, std) for technical features.

    Only the features computable from close + vol are included here —
    no full feature engineering pipeline needed, so this pass is fast.

    Returns:
        {date_int (YYYYMMDD): {feature_name: (mean, std)}}
    """
    from collections import defaultdict
    from math import sqrt

    # Welford online mean/variance accumulator:
    # {date_int: {feat: [n, mean, M2]}}
    running: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))

    total = len(stock_files)
    for idx, (ts_code, filepath) in enumerate(stock_files):
        try:
            df = pd.read_csv(filepath, usecols=['trade_date', 'close', 'vol'])
            if len(df) < min_data_points:
                continue
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date').reset_index(drop=True)

            c = df['close'].values.astype('float64')
            v = df['vol'].values.astype('float64')

            # Compute lightweight technical features row-by-row
            feat_arrays: Dict[str, np.ndarray] = {}
            with np.errstate(divide='ignore', invalid='ignore'):
                ret   = np.where(c[:-1] > 0, (c[1:] - c[:-1]) / c[:-1], np.nan)
                lret  = np.where(c[:-1] > 0, np.log(c[1:] / c[:-1]), np.nan)
                vchg  = np.where(v[:-1] > 0, (v[1:] - v[:-1]) / v[:-1], np.nan)

                # Prepend NaN for lag-0 alignment
                ret  = np.concatenate([[np.nan], ret])
                lret = np.concatenate([[np.nan], lret])
                vchg = np.concatenate([[np.nan], vchg])

            n = len(c)
            feat_arrays['returns']       = ret
            feat_arrays['log_returns']   = lret
            feat_arrays['volume_change'] = vchg

            for lag in [1, 2, 3, 5]:
                lagged = np.empty(n); lagged[:] = np.nan
                lagged[lag:] = ret[lag - (lag - lag):n - lag] if lag == 0 else ret[:n - lag]
                # Simpler: shift ret by lag
                shifted = np.empty(n); shifted[:] = np.nan
                shifted[lag:] = ret[:-lag] if lag > 0 else ret
                feat_arrays[f'return_lag_{lag}'] = shifted

            for p in [5, 10, 20]:
                roc = np.empty(n); roc[:] = np.nan
                mom = np.empty(n); mom[:] = np.nan
                vol_f = np.empty(n); vol_f[:] = np.nan
                dhi = np.empty(n); dhi[:] = np.nan
                dlo = np.empty(n); dlo[:] = np.nan

                for i in range(p, n):
                    if c[i - p] > 0:
                        roc[i] = 100.0 * (c[i] - c[i - p]) / c[i - p]
                    mom[i] = c[i] - c[i - p]
                    window_ret = ret[i - p + 1:i + 1]
                    valid_r = window_ret[np.isfinite(window_ret)]
                    if len(valid_r) > 1:
                        vol_f[i] = float(np.std(valid_r, ddof=1))
                    roll_hi = np.max(c[i - p + 1:i + 1])
                    roll_lo = np.min(c[i - p + 1:i + 1])
                    if roll_hi > 0:
                        dhi[i] = (c[i] - roll_hi) / roll_hi
                    if roll_lo > 0:
                        dlo[i] = (c[i] - roll_lo) / roll_lo

                if p <= 10:
                    feat_arrays[f'momentum_{p}'] = mom
                feat_arrays[f'roc_{p}']        = roc
                feat_arrays[f'volatility_{p}'] = vol_f
                if p == 20:
                    feat_arrays['dist_from_high_20'] = dhi
                    feat_arrays['dist_from_low_20']  = dlo

            date_ints = (df['trade_date'].dt.year * 10000
                         + df['trade_date'].dt.month * 100
                         + df['trade_date'].dt.day).values.astype(np.int32)

            for feat, arr in feat_arrays.items():
                for d, val in zip(date_ints, arr):
                    if not np.isfinite(val):
                        continue
                    state = running[int(d)][feat]  # [n, mean, M2]
                    state[0] += 1
                    delta = val - state[1]
                    state[1] += delta / state[0]
                    state[2] += delta * (val - state[1])

        except Exception:
            continue

        if (idx + 1) % 100 == 0:
            print(f"  CS tech stats: {idx + 1}/{total} stocks...")

    # Convert to (mean, std) pairs
    cs_stats: Dict[int, Dict[str, tuple]] = {}
    for date_int, feat_dict in running.items():
        cs_stats[date_int] = {}
        for feat, (n, mean, M2) in feat_dict.items():
            if n < 2:
                continue
            std = max(float((M2 / (n - 1)) ** 0.5), 1e-6)
            cs_stats[date_int][feat] = (float(mean), std)

    print(f"  CS tech stats: built for {len(cs_stats)} trading days "
          f"across {len(stock_files)} stocks")
    return cs_stats


# ============================================================================
# Holiday Functions
# ============================================================================

def get_chinese_holidays_for_year(year: int) -> List[Tuple[datetime, str, int]]:
    """Get Chinese holiday dates for a given year."""
    holidays = []

    # Fixed holidays
    holidays.append((datetime(year, 1, 1), 'New Year', 1))
    holidays.append((datetime(year, 5, 1), 'Labor Day', 5))
    holidays.append((datetime(year, 10, 1), 'National Day', 7))

    # Spring Festival + Lantern Festival
    if year in SPRING_FESTIVAL_DATES:
        m, d = SPRING_FESTIVAL_DATES[year]
        holidays.append((datetime(year, m, d), 'Spring Festival', 7))
        lantern_date = datetime(year, m, d) + timedelta(days=15)
        holidays.append((lantern_date, 'Lantern Festival', 1))

    # Qingming
    if year in QINGMING_DATES:
        m, d = QINGMING_DATES[year]
        holidays.append((datetime(year, m, d), 'Qingming', 3))
    else:
        holidays.append((datetime(year, 4, 4), 'Qingming', 3))

    # Dragon Boat
    if year in DRAGON_BOAT_DATES:
        m, d = DRAGON_BOAT_DATES[year]
        holidays.append((datetime(year, m, d), 'Dragon Boat', 3))

    # Mid-Autumn
    if year in MID_AUTUMN_DATES:
        m, d = MID_AUTUMN_DATES[year]
        holidays.append((datetime(year, m, d), 'Mid-Autumn', 3))

    # Double Ninth
    if year in DOUBLE_NINTH_DATES:
        m, d = DOUBLE_NINTH_DATES[year]
        holidays.append((datetime(year, m, d), 'Double Ninth', 1))

    # Winter Solstice
    if year in WINTER_SOLSTICE_DATES:
        m, d = WINTER_SOLSTICE_DATES[year]
        holidays.append((datetime(year, m, d), 'Winter Solstice', 1))

    # Qixi
    if year in QIXI_DATES:
        m, d = QIXI_DATES[year]
        holidays.append((datetime(year, m, d), 'Qixi', 1))

    # Laba Festival
    if year in LABA_DATES:
        m, d = LABA_DATES[year]
        try:
            holidays.append((datetime(year, m, d), 'Laba Festival', 1))
        except:
            pass

    return holidays


# ============================================================================
# Pattern Detection (Vectorized)
# ============================================================================

def detect_w_bottom(prices: np.ndarray, window: int = 20, threshold: float = 0.02) -> np.ndarray:
    """Detect W bottom pattern using vectorized operations."""
    n = len(prices)
    signals = np.zeros(n)

    if n < window:
        return signals

    for i in range(window, n):
        segment = prices[i - window:i]
        mid = window // 2
        quarter = window // 4

        left_min_idx = np.argmin(segment[:mid])
        right_min_idx = mid + np.argmin(segment[mid:])
        middle_max_idx = left_min_idx + np.argmax(segment[left_min_idx:right_min_idx + 1]) if right_min_idx > left_min_idx else mid

        if left_min_idx < middle_max_idx < right_min_idx:
            left_min = segment[left_min_idx]
            right_min = segment[right_min_idx]
            middle_max = segment[middle_max_idx]

            if abs(left_min - right_min) / left_min < threshold:
                if middle_max > left_min * (1 + threshold) and middle_max > right_min * (1 + threshold):
                    if segment[-1] > middle_max:
                        signals[i] = 1.0
                    elif segment[-1] > (left_min + middle_max) / 2:
                        signals[i] = 0.5

    return signals


def detect_m_top(prices: np.ndarray, window: int = 20, threshold: float = 0.02) -> np.ndarray:
    """Detect M top pattern using vectorized operations."""
    n = len(prices)
    signals = np.zeros(n)

    if n < window:
        return signals

    for i in range(window, n):
        segment = prices[i - window:i]
        mid = window // 2

        left_max_idx = np.argmax(segment[:mid])
        right_max_idx = mid + np.argmax(segment[mid:])
        middle_min_idx = left_max_idx + np.argmin(segment[left_max_idx:right_max_idx + 1]) if right_max_idx > left_max_idx else mid

        if left_max_idx < middle_min_idx < right_max_idx:
            left_max = segment[left_max_idx]
            right_max = segment[right_max_idx]
            middle_min = segment[middle_min_idx]

            if abs(left_max - right_max) / left_max < threshold:
                if middle_min < left_max * (1 - threshold) and middle_min < right_max * (1 - threshold):
                    if segment[-1] < middle_min:
                        signals[i] = 1.0
                    elif segment[-1] < (left_max + middle_min) / 2:
                        signals[i] = 0.5

    return signals


def detect_patterns_multi_window(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect patterns with multiple window sizes."""
    w_short = detect_w_bottom(prices, window=10)
    w_long = detect_w_bottom(prices, window=20)
    m_short = detect_m_top(prices, window=10)
    m_long = detect_m_top(prices, window=20)
    return w_short, w_long, m_short, m_long


# ============================================================================
# Feature Engineering (Vectorized with pandas apply)
# ============================================================================

def calculate_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cyclical date/time features and holiday indicators."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    # Extract date components (vectorized)
    df['year'] = df['trade_date'].dt.year
    df['month'] = df['trade_date'].dt.month
    df['day'] = df['trade_date'].dt.day
    df['day_of_week'] = df['trade_date'].dt.dayofweek
    df['day_of_year'] = df['trade_date'].dt.dayofyear
    df['week_of_year'] = df['trade_date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['trade_date'].dt.quarter

    # Cyclical encoding (vectorized)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    df['dom_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['dom_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # Trading day indicators (vectorized)
    df['is_monday'] = (df['day_of_week'] == 0).astype(float)
    df['is_friday'] = (df['day_of_week'] == 4).astype(float)
    df['is_month_start'] = (df['day'] <= 3).astype(float)
    df['is_month_end'] = (df['day'] >= 28).astype(float)
    df['is_year_start'] = ((df['month'] == 1) & (df['day'] <= 5)).astype(float)
    df['is_year_end'] = ((df['month'] == 12) & (df['day'] >= 25)).astype(float)

    # Collect all holidays across all years in the data
    all_holiday_starts = []
    all_holiday_ends = []

    for year in df['year'].unique():
        holidays = get_chinese_holidays_for_year(year)
        for holiday_date, holiday_name, duration in holidays:
            all_holiday_starts.append(holiday_date)
            all_holiday_ends.append(holiday_date + timedelta(days=duration))

    # Convert to numpy arrays (timestamps in nanoseconds for numba)
    trade_dates_ns = df['trade_date'].values.astype('datetime64[ns]').astype(np.int64)
    holiday_starts_ns = np.array([np.datetime64(d).astype('datetime64[ns]').astype(np.int64)
                                   for d in all_holiday_starts], dtype=np.int64)
    holiday_ends_ns = np.array([np.datetime64(d).astype('datetime64[ns]').astype(np.int64)
                                 for d in all_holiday_ends], dtype=np.int64)

    # Use numba-optimized function for holiday calculations
    is_pre, is_post, days_to, days_from = compute_holiday_distances(
        trade_dates_ns, holiday_starts_ns, holiday_ends_ns,
        pre_holiday_days=7, post_holiday_days=5
    )

    df['is_pre_holiday'] = is_pre
    df['is_post_holiday'] = is_post
    df['days_to_holiday'] = days_to
    df['days_from_holiday'] = days_from

    # Normalize and create holiday effect
    df['days_to_holiday_norm'] = 1.0 / (df['days_to_holiday'] + 1)
    df['days_from_holiday_norm'] = 1.0 / (df['days_from_holiday'] + 1)
    df['holiday_effect'] = df['is_pre_holiday'] * 0.5 + df['is_post_holiday'] * 0.3

    # Special periods
    df['is_january'] = (df['month'] == 1).astype(float)
    df['is_december'] = (df['month'] == 12).astype(float)
    df['is_earnings_season'] = df['month'].isin([1, 4, 7, 10]).astype(float)
    df['is_weak_season'] = df['month'].isin([5, 6, 9]).astype(float)

    return df


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators using vectorized operations."""
    df = df.copy()

    # Basic features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low'] - 1
    df['close_open_ratio'] = df['close'] / df['open'] - 1

    # Moving averages (vectorized)
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'sma_{window}_ratio'] = df['close'] / df[f'sma_{window}'] - 1
        df[f'vol_sma_{window}'] = df['vol'].rolling(window=window).mean()
        df[f'vol_sma_{window}_ratio'] = df['vol'] / df[f'vol_sma_{window}'] - 1
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    bb_sma = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # Volume and price features
    df['volume_change'] = df['vol'].pct_change()
    rolling_high = df['high'].rolling(window=20).max()
    rolling_low = df['low'].rolling(window=20).min()
    df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']

    # OBV
    obv = np.where(df['close'] > df['close'].shift(1), df['vol'],
                   np.where(df['close'] < df['close'].shift(1), -df['vol'], 0))
    df['obv'] = np.cumsum(obv)
    df['obv_sma'] = pd.Series(df['obv']).rolling(window=20).mean()
    # Normalize by the rolling std of OBV (always positive, proportional to OBV magnitude)
    # rather than the SMA, which can pass through zero and cause division-by-near-zero.
    obv_std20 = pd.Series(df['obv']).rolling(window=20).std().fillna(0)
    obv_denom = np.maximum(df['obv_sma'].abs(), obv_std20).clip(lower=df['vol'].rolling(20).mean().clip(lower=1.0))
    df['obv_ratio'] = (df['obv'] / obv_denom).clip(-5, 5)

    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)

    # CCI (using numba-optimized function)
    df['cci'] = compute_cci(
        df['high'].values.astype(np.float64),
        df['low'].values.astype(np.float64),
        df['close'].values.astype(np.float64),
        window=20
    )

    # ROC and Momentum
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = 100 * (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10)
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().abs() * -1
    plus_dm = np.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm.abs() > plus_dm) & (minus_dm < 0), minus_dm.abs(), 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / (df['atr_14'] + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / (df['atr_14'] + 1e-10)
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(window=14).mean()
    df['di_diff'] = plus_di - minus_di

    # MA Crossovers
    df['sma_5_10_cross'] = (df['sma_5'] - df['sma_10']) / (df['sma_10'] + 1e-10)
    df['sma_10_20_cross'] = (df['sma_10'] - df['sma_20']) / (df['sma_20'] + 1e-10)
    df['ema_12_26_cross'] = (ema12 - ema26) / (ema26 + 1e-10)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)

    # Gaps
    df['gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10)
    df['gap_abs'] = abs(df['gap'])

    # Trend indicators
    df['above_sma_5'] = (df['close'] > df['sma_5']).astype(float)
    df['above_sma_10'] = (df['close'] > df['sma_10']).astype(float)
    df['above_sma_20'] = (df['close'] > df['sma_20']).astype(float)
    df['trend_score'] = df['above_sma_5'] + df['above_sma_10'] + df['above_sma_20']

    # Candle patterns
    df['body_size'] = abs(df['close'] - df['open']) / (df['open'] + 1e-10)
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['open'] + 1e-10)
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['open'] + 1e-10)
    df['is_bullish_candle'] = (df['close'] > df['open']).astype(float)

    # Consecutive days
    df['up_day'] = (df['returns'] > 0).astype(int)
    df['consecutive_up'] = df['up_day'].groupby((df['up_day'] != df['up_day'].shift()).cumsum()).cumsum()
    df['consecutive_down'] = (1 - df['up_day']).groupby(((1 - df['up_day']) != (1 - df['up_day']).shift()).cumsum()).cumsum()

    # Distance from highs/lows
    df['dist_from_high_20'] = (df['close'] - df['high'].rolling(20).max()) / (df['high'].rolling(20).max() + 1e-10)
    df['dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / (df['low'].rolling(20).min() + 1e-10)

    # VWAP
    df['vwap'] = df['amount'] / (df['vol'] + 1e-10)
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)

    # Pattern detection (using numba-optimized function)
    prices = df['close'].values.astype(np.float64)
    w_short, w_long, m_short, m_long = detect_patterns_multi_window_numba(prices)
    df['w_bottom_short'] = w_short
    df['w_bottom_long'] = w_long
    df['m_top_short'] = m_short
    df['m_top_long'] = m_long
    df['w_bottom_signal'] = (df['w_bottom_short'] + df['w_bottom_long']) / 2
    df['m_top_signal'] = (df['m_top_short'] + df['m_top_long']) / 2
    df['pattern_bias'] = df['w_bottom_signal'] - df['m_top_signal']

    # Date features
    df = calculate_date_features(df)

    return df


# ============================================================================
# Data Loading
# ============================================================================

def load_market_context_data(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Load market-wide context features from index data directories.

    Sources (all under stock_data/index/):
      index_dailybasic/000300_SH.csv  — CSI300 pe_ttm, pb, turnover_rate
      index_dailybasic/000905_SH.csv  — CSI500 pe_ttm, pb
      index_dailybasic/000001_SH.csv  — SSE Composite pe_ttm
      idx_factor_pro/000300_SH.csv    — CSI300 RSI6, MACD, CCI, BIAS1, KDJ_K
      index_global/DJI.csv            — Dow Jones pct_chg (lag-1)
      index_global/HSI.csv            — Hang Seng pct_chg (lag-1)
      index_global/IXIC.csv           — NASDAQ pct_chg (lag-1)
      index_global/N225.csv           — Nikkei pct_chg (lag-1)

    Returns a DataFrame indexed by trade_date (int YYYYMMDD) with columns
    matching MARKET_CONTEXT_FEATURES.  Returns None if no index data found.

    Global returns are lagged 1 trading day to prevent lookahead: Chinese
    stocks open at 09:30 CST while DJI/IXIC close the previous night; a
    1-day lag is the conservative choice that is safe for all four markets.
    """
    index_dir = os.path.join(data_dir, 'index')
    if not os.path.isdir(index_dir):
        print(f"[market_context] Index directory not found: {index_dir} — skipping")
        return None

    frames: List[pd.DataFrame] = []

    # ── index_dailybasic ──────────────────────────────────────────────────
    idb_dir = os.path.join(index_dir, 'index_dailybasic')
    idb_map = {
        '000300_SH.csv': {'pe_ttm': 'csi300_pe_ttm', 'pb': 'csi300_pb',
                          'turnover_rate': 'csi300_turnover'},
        '000905_SH.csv': {'pe_ttm': 'csi500_pe_ttm', 'pb': 'csi500_pb'},
        '000001_SH.csv': {'pe_ttm': 'sse_pe_ttm'},
        '000016_SH.csv': {'pe_ttm': 'sse50_pe_ttm', 'pb': 'sse50_pb',
                          'turnover_rate': 'sse50_turnover'},
        '399006_SZ.csv': {'pe_ttm': 'gem_pe_ttm', 'pb': 'gem_pb',
                          'turnover_rate': 'gem_turnover'},
        '000852_SH.csv': {'pe_ttm': 'csi1000_pe_ttm', 'pb': 'csi1000_pb'},
    }
    for fname, col_map in idb_map.items():
        fpath = os.path.join(idb_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            tmp = pd.read_csv(fpath, usecols=['trade_date'] + list(col_map.keys()))
            tmp = tmp.rename(columns=col_map).set_index('trade_date')
            frames.append(tmp)
        except Exception as e:
            print(f"[market_context] Warning: could not load {fname}: {e}")

    # ── idx_factor_pro (multiple indices) ────────────────────────────────
    # Each entry: (filename, feature_prefix)
    _ifp_indices = [
        ('000300_SH.csv', 'csi300'),
        ('000905_SH.csv', 'csi500'),
        ('000016_SH.csv', 'sse50'),
        ('399006_SZ.csv', 'gem'),
        ('000852_SH.csv', 'csi1000'),
    ]
    _ifp_src_cols = [
        'rsi_bfq_6', 'rsi_bfq_12', 'rsi_bfq_24',
        'macd_bfq', 'cci_bfq', 'bias1_bfq',
        'kdj_k_bfq', 'kdj_d_bfq',
        'dmi_adx_bfq', 'dmi_pdi_bfq', 'dmi_mdi_bfq',
        'obv_bfq', 'mfi_bfq',
        'updays', 'downdays',
        'roc_bfq', 'mtm_bfq',
    ]
    _ifp_dst_suffixes = [
        'rsi6', 'rsi12', 'rsi24',
        'macd', 'cci', 'bias1',
        'kdj_k', 'kdj_d',
        'adx', 'pdi', 'mdi',
        'obv', 'mfi',
        'updays', 'downdays',
        'roc', 'mtm',
    ]
    for ifp_fname, prefix in _ifp_indices:
        ifp_path = os.path.join(index_dir, 'idx_factor_pro', ifp_fname)
        if not os.path.exists(ifp_path):
            continue
        col_map = {src: f'{prefix}_{dst}'
                   for src, dst in zip(_ifp_src_cols, _ifp_dst_suffixes)}
        try:
            available = pd.read_csv(ifp_path, nrows=0).columns.tolist()
            keep_src  = [c for c in col_map if c in available]
            keep_map  = {c: col_map[c] for c in keep_src}
            if keep_src:
                tmp = pd.read_csv(ifp_path, usecols=['trade_date'] + keep_src)
                tmp = tmp.rename(columns=keep_map).set_index('trade_date')
                frames.append(tmp)
        except Exception as e:
            print(f"[market_context] Warning: could not load idx_factor_pro/{ifp_fname}: {e}")

    # ── index_global (lagged 1 day) ────────────────────────────────────────
    ig_dir = os.path.join(index_dir, 'index_global')
    ig_map = {
        'DJI.csv':  'dji_ret_lag1',
        'HSI.csv':  'hsi_ret_lag1',
        'IXIC.csv': 'ixic_ret_lag1',
        'N225.csv': 'n225_ret_lag1',
        'SPX.csv':  'spx_ret_lag1',
    }
    for fname, col_name in ig_map.items():
        fpath = os.path.join(ig_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            tmp = pd.read_csv(fpath, usecols=['trade_date', 'pct_chg'])
            tmp = tmp.sort_values('trade_date').reset_index(drop=True)
            # Lag by 1 row: each date receives the PREVIOUS day's return
            tmp[col_name] = tmp['pct_chg'].shift(1)
            tmp = tmp[['trade_date', col_name]].dropna().set_index('trade_date')
            frames.append(tmp)
        except Exception as e:
            print(f"[market_context] Warning: could not load {fname}: {e}")

    if not frames:
        print("[market_context] No index data loaded — market context features will be zero")
        return None

    ctx = pd.concat(frames, axis=1)

    # Fill any gaps left by joining sparse index series
    ctx = ctx.sort_index().ffill().bfill()

    # Ensure all expected columns exist (fill missing ones with 0)
    missing_ctx = [col for col in MARKET_CONTEXT_FEATURES if col not in ctx.columns]
    if missing_ctx:
        ctx = pd.concat(
            [ctx, pd.DataFrame(0.0, index=ctx.index, columns=missing_ctx)], axis=1)

    ctx = ctx[MARKET_CONTEXT_FEATURES].astype('float32')
    print(
        f"[market_context] Loaded {len(ctx)} trading days of market context "
        f"({int(ctx.index.min())} → {int(ctx.index.max())})"
    )
    return ctx


def load_index_membership_data(data_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load CSI300 / CSI500 / SSE50 constituent weights from index_weight files.

    Uses the LATEST available trade_date snapshot for each stock (index
    constituents rebalance quarterly; the snapshot is a reasonable proxy).

    Returns ``{bare_ts_code: {'is_csi300': 0/1, 'csi300_weight': float,
                              'is_csi500': 0/1, 'is_sse50': 0/1}}``
    where ``bare_ts_code`` is the numeric-only code (e.g. ``'600000'``).
    """
    iw_dir = os.path.join(data_dir, 'index', 'index_weight')
    if not os.path.isdir(iw_dir):
        print(f"[index_membership] index_weight directory not found: {iw_dir} — skipping")
        return {}

    index_files = {
        '000300_SH.csv': ('is_csi300', 'csi300_weight'),
        '000905_SH.csv': ('is_csi500', None),
        '000016_SH.csv': ('is_sse50',  None),
    }

    membership: Dict[str, Dict[str, float]] = {}

    for fname, (flag_col, weight_col) in index_files.items():
        fpath = os.path.join(iw_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
            # Take the most recent snapshot for each constituent
            df = df.sort_values('trade_date').groupby('con_code').last().reset_index()
            for _, row in df.iterrows():
                bare = str(row['con_code']).split('.')[0]
                if bare not in membership:
                    membership[bare] = {f: 0.0 for f in INDEX_MEMBERSHIP_FEATURES}
                membership[bare][flag_col] = 1.0
                if weight_col:
                    try:
                        membership[bare][weight_col] = float(row.get('weight', 0.0))
                    except (ValueError, TypeError):
                        membership[bare][weight_col] = 0.0
        except Exception as e:
            print(f"[index_membership] Warning: could not load {fname}: {e}")

    print(f"[index_membership] Loaded membership data for {len(membership)} stocks")
    return membership


def merge_market_context(df: pd.DataFrame, market_ctx: pd.DataFrame) -> pd.DataFrame:
    """
    Merge market-wide context features into a stock's daily DataFrame.

    ``df`` must have a ``trade_date`` column (datetime or YYYYMMDD int/str).
    ``market_ctx`` is indexed by trade_date as int (YYYYMMDD).

    Missing dates (weekends, holidays not in index data) are forward-filled
    then backward-filled so every row has a value.  Remaining NaNs (before
    the first index record) are filled with 0.
    """
    if market_ctx is None or market_ctx.empty:
        return pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=MARKET_CONTEXT_FEATURES)],
            axis=1
        )

    # Normalise trade_date to int for the join key
    if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        date_key = df['trade_date'].dt.strftime('%Y%m%d').astype(int)
    else:
        date_key = df['trade_date'].astype(str).str.replace('-', '').astype(int)

    merged   = market_ctx.reindex(date_key.values)
    new_cols = {}
    for col in MARKET_CONTEXT_FEATURES:
        vals = merged[col].values if col in merged.columns else np.zeros(len(df), dtype='float32')
        s = pd.Series(vals, dtype='float32')
        new_cols[col] = s.ffill().bfill().fillna(0.0).values
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def merge_index_membership(
    df: pd.DataFrame,
    membership: Dict[str, Dict[str, float]],
    ts_code: str,
) -> pd.DataFrame:
    """
    Add static index membership columns to every row of a stock's DataFrame.

    The bare ts_code (numeric part only, e.g. ``'600000'``) is used for the
    membership dict lookup so SH/SZ suffix differences don't cause misses.
    """
    bare      = ts_code.split('.')[0]
    stock_mem = membership.get(bare, {f: 0.0 for f in INDEX_MEMBERSHIP_FEATURES})
    new_cols  = {col: float(stock_mem.get(col, 0.0)) for col in INDEX_MEMBERSHIP_FEATURES}
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def load_sector_data(data_dir: str) -> pd.DataFrame:
    """Load sector data from CSV file."""
    sector_path = os.path.join(data_dir, 'stock_sectors.csv')

    if os.path.exists(sector_path):
        print(f"Loading existing sector data from {sector_path}")
        return pd.read_csv(sector_path)
    else:
        print("Sector data not found. Returning empty DataFrame.")
        return pd.DataFrame()


def load_daily_basic_data(data_dir: str, last_n_files: int = 0) -> pd.DataFrame:
    """
    Load daily basic data from stock_data/daily_basic folder.

    Args:
        data_dir: Root data directory.
        last_n_files: If > 0, load only the most recent N trading-day files.
                      Use for prediction (e.g. last_n_files=300 ≈ 14 months).
                      Default 0 = load all files (for training).
    """
    daily_basic_dir = os.path.join(data_dir, 'daily_basic')

    if not os.path.exists(daily_basic_dir):
        print(f"Daily basic directory not found: {daily_basic_dir}")
        return pd.DataFrame()

    files = sorted(f for f in os.listdir(daily_basic_dir) if f.endswith('.csv'))
    if last_n_files > 0:
        files = files[-last_n_files:]

    if not files:
        print("No daily basic CSV files found.")
        return pd.DataFrame()

    print(f"Loading {len(files)} daily basic files...")

    dfs = []
    for f in files:
        try:
            filepath = os.path.join(daily_basic_dir, f)
            df = pd.read_csv(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    daily_basic = pd.concat(dfs, ignore_index=True)

    # Convert trade_date to datetime for merging
    daily_basic['trade_date'] = pd.to_datetime(daily_basic['trade_date'].astype(str))

    # Drop duplicates (same stock, same date)
    daily_basic = daily_basic.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    # Cast float columns to float32 to halve memory usage
    float_cols = daily_basic.select_dtypes(include='float64').columns
    if len(float_cols) > 0:
        daily_basic[float_cols] = daily_basic[float_cols].astype('float32')

    print(f"Loaded {len(daily_basic)} daily basic records")

    return daily_basic


def merge_daily_basic(df: pd.DataFrame, daily_basic: pd.DataFrame, ts_code: str = None) -> pd.DataFrame:
    """
    Merge daily basic data with stock data and calculate derived features.

    Args:
        df: Stock dataframe with trade_date column
        daily_basic: Daily basic dataframe with all stocks
        ts_code: Stock code to filter daily_basic data

    Returns:
        DataFrame with daily_basic features merged and derived features calculated
    """
    if daily_basic is None or len(daily_basic) == 0:
        # Add empty columns if no daily_basic data
        for col in ['turnover_rate', 'turnover_rate_f', 'volume_ratio',
                    'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                    'dv_ratio', 'dv_ttm', 'total_mv_norm', 'circ_mv_norm',
                    'float_ratio', 'free_ratio']:
            df[col] = 0.0
        return df

    # Filter daily_basic for this stock (skip if already pre-filtered, ts_code=None)
    if ts_code is not None:
        stock_basic = daily_basic[daily_basic['ts_code'] == ts_code].copy()
    else:
        stock_basic = daily_basic.copy()

    if len(stock_basic) == 0:
        # Add empty columns if no data for this stock
        for col in ['turnover_rate', 'turnover_rate_f', 'volume_ratio',
                    'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                    'dv_ratio', 'dv_ttm', 'total_mv_norm', 'circ_mv_norm',
                    'float_ratio', 'free_ratio']:
            df[col] = 0.0
        return df

    # Ensure trade_date is datetime in both
    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    # Select columns to merge (exclude ts_code as we'll use df's index)
    merge_cols = ['trade_date', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                  'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
                  'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']

    # Keep only columns that exist
    merge_cols = [c for c in merge_cols if c in stock_basic.columns]
    stock_basic = stock_basic[merge_cols]

    # Merge on trade_date
    df = df.merge(stock_basic, on='trade_date', how='left')

    # Fill NaN values with forward fill, then backward fill, then 0
    basic_cols = ['turnover_rate', 'turnover_rate_f', 'volume_ratio',
                  'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
                  'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']

    for col in basic_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    # Calculate derived features
    # Normalized market cap (log scale for better distribution)
    if 'total_mv' in df.columns:
        df['total_mv_norm'] = np.log1p(df['total_mv']) / 20.0  # Normalize to ~0-1 range
    else:
        df['total_mv_norm'] = 0.0

    if 'circ_mv' in df.columns:
        df['circ_mv_norm'] = np.log1p(df['circ_mv']) / 20.0
    else:
        df['circ_mv_norm'] = 0.0

    # Float and free share ratios
    if 'total_share' in df.columns and 'float_share' in df.columns:
        df['float_ratio'] = df['float_share'] / (df['total_share'] + 1e-10)
    else:
        df['float_ratio'] = 0.0

    if 'total_share' in df.columns and 'free_share' in df.columns:
        df['free_ratio'] = df['free_share'] / (df['total_share'] + 1e-10)
    else:
        df['free_ratio'] = 0.0

    # Clip extreme values for numerical stability
    for col in ['pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm']:
        if col in df.columns:
            df[col] = df[col].clip(-1000, 1000)

    return df


def load_stk_limit_data(data_dir: str, last_n_files: int = 0) -> pd.DataFrame:
    """
    Load stk_limit data from stock_data/stk_limit/ folder.

    Args:
        last_n_files: If > 0, load only the most recent N files (for prediction).
    """
    stk_limit_dir = os.path.join(data_dir, 'stk_limit')
    if not os.path.exists(stk_limit_dir):
        return pd.DataFrame()

    files = sorted(f for f in os.listdir(stk_limit_dir) if f.endswith('.csv'))
    if last_n_files > 0:
        files = files[-last_n_files:]
    if not files:
        return pd.DataFrame()

    print(f"Loading {len(files)} stk_limit files...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(os.path.join(stk_limit_dir, f)))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    stk_limit = pd.concat(dfs, ignore_index=True)
    stk_limit['trade_date'] = pd.to_datetime(stk_limit['trade_date'].astype(str))
    stk_limit = stk_limit.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    print(f"Loaded {len(stk_limit)} stk_limit records")
    return stk_limit


def load_moneyflow_data(data_dir: str, last_n_files: int = 0) -> pd.DataFrame:
    """
    Load moneyflow data from stock_data/moneyflow/ folder.

    Args:
        last_n_files: If > 0, load only the most recent N files (for prediction).
    """
    mf_dir = os.path.join(data_dir, 'moneyflow')
    if not os.path.exists(mf_dir):
        return pd.DataFrame()

    files = sorted(f for f in os.listdir(mf_dir) if f.endswith('.csv'))
    if last_n_files > 0:
        files = files[-last_n_files:]
    if not files:
        return pd.DataFrame()

    print(f"Loading {len(files)} moneyflow files...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(os.path.join(mf_dir, f)))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    mf = pd.concat(dfs, ignore_index=True)
    mf['trade_date'] = pd.to_datetime(mf['trade_date'].astype(str))
    mf = mf.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    float_cols = mf.select_dtypes(include='float64').columns
    if len(float_cols) > 0:
        mf[float_cols] = mf[float_cols].astype('float32')
    print(f"Loaded {len(mf)} moneyflow records")
    return mf


def merge_stk_limit(df: pd.DataFrame, stk_limit: pd.DataFrame, ts_code: Optional[str]) -> pd.DataFrame:
    """
    Merge stk_limit features into a stock's daily DataFrame.

    Adds:
      is_limit_up   — 1 if close >= up_limit (locked limit-up)
      is_limit_down — 1 if close <= down_limit (locked limit-down)

    Pass ts_code=None when stk_limit is already pre-filtered to this stock
    (avoids an O(N) scan of the full DataFrame).
    """
    zero_cols = ['is_limit_up', 'is_limit_down']
    if stk_limit is None or len(stk_limit) == 0:
        return pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=zero_cols)], axis=1)

    if ts_code is not None:
        stock_sl = stk_limit[stk_limit['ts_code'] == ts_code].copy()
    else:
        stock_sl = stk_limit  # already pre-filtered
    if len(stock_sl) == 0:
        return pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=zero_cols)], axis=1)

    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    stock_sl = stock_sl[['trade_date', 'up_limit', 'down_limit']]
    df = df.merge(stock_sl, on='trade_date', how='left')

    df['up_limit'] = df['up_limit'].ffill().bfill().fillna(0)
    df['down_limit'] = df['down_limit'].ffill().bfill().fillna(0)

    df['is_limit_up'] = ((df['close'] >= df['up_limit']) & (df['up_limit'] > 0)).astype(float)
    df['is_limit_down'] = ((df['close'] <= df['down_limit']) & (df['down_limit'] > 0)).astype(float)

    df = df.drop(columns=['up_limit', 'down_limit'])
    return df


def merge_moneyflow(df: pd.DataFrame, moneyflow: pd.DataFrame, ts_code: Optional[str]) -> pd.DataFrame:
    """
    Merge moneyflow features into a stock's daily DataFrame.

    Adds:
      net_lg_flow_ratio  — (buy_lg_vol - sell_lg_vol) / vol  (large-order net)
      net_elg_flow_ratio — (buy_elg_vol - sell_elg_vol) / vol (extra-large net)

    Pass ts_code=None when moneyflow is already pre-filtered to this stock
    (avoids an O(N) scan of the full DataFrame).
    """
    zero_cols = ['net_lg_flow_ratio', 'net_elg_flow_ratio']
    if moneyflow is None or len(moneyflow) == 0:
        return pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=zero_cols)], axis=1)

    if ts_code is not None:
        stock_mf = moneyflow[moneyflow['ts_code'] == ts_code].copy()
    else:
        stock_mf = moneyflow  # already pre-filtered
    if len(stock_mf) == 0:
        return pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=zero_cols)], axis=1)

    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    keep_cols = ['trade_date']
    for c in ['buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol']:
        if c in stock_mf.columns:
            keep_cols.append(c)

    stock_mf = stock_mf[keep_cols]
    df = df.merge(stock_mf, on='trade_date', how='left')

    vol = df['vol'].clip(lower=1.0)

    if 'buy_lg_vol' in df.columns and 'sell_lg_vol' in df.columns:
        df['buy_lg_vol'] = df['buy_lg_vol'].ffill().bfill().fillna(0)
        df['sell_lg_vol'] = df['sell_lg_vol'].ffill().bfill().fillna(0)
        df['net_lg_flow_ratio'] = ((df['buy_lg_vol'] - df['sell_lg_vol']) / vol).clip(-5, 5)
        df = df.drop(columns=['buy_lg_vol', 'sell_lg_vol'])
    else:
        df['net_lg_flow_ratio'] = 0.0

    if 'buy_elg_vol' in df.columns and 'sell_elg_vol' in df.columns:
        df['buy_elg_vol'] = df['buy_elg_vol'].ffill().bfill().fillna(0)
        df['sell_elg_vol'] = df['sell_elg_vol'].ffill().bfill().fillna(0)
        df['net_elg_flow_ratio'] = ((df['buy_elg_vol'] - df['sell_elg_vol']) / vol).clip(-5, 5)
        df = df.drop(columns=['buy_elg_vol', 'sell_elg_vol'])
    else:
        df['net_elg_flow_ratio'] = 0.0

    return df


def load_single_stock(file_path: str, min_data_points: int = 100) -> Optional[Tuple[str, pd.DataFrame]]:
    """Load a single stock file. Used for parallel processing."""
    try:
        df = pd.read_csv(file_path)
        if len(df) < min_data_points:
            return None

        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df = df.sort_values('trade_date').reset_index(drop=True)

        ts_code = os.path.basename(file_path).replace('.csv', '')
        return (ts_code, df)
    except Exception as e:
        return None


def load_stock_data(
    data_dir: str,
    market: str = 'sh',
    max_stocks: int = None,
    min_data_points: int = 100,
    num_workers: int = 4
) -> Dict[str, pd.DataFrame]:
    """Load stock data with parallel processing."""
    market_dir = os.path.join(data_dir, market)
    stocks = {}

    if not os.path.exists(market_dir):
        print(f"Market directory not found: {market_dir}")
        return stocks

    files = [os.path.join(market_dir, f) for f in os.listdir(market_dir) if f.endswith('.csv')]

    if max_stocks is not None and len(files) > max_stocks:
        np.random.shuffle(files)
        files = files[:max_stocks]

    print(f"Loading up to {len(files)} stocks from {market.upper()} market using {num_workers} workers...")

    # Use multiprocessing for faster loading
    load_func = partial(load_single_stock, min_data_points=min_data_points)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_func, f): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                ts_code, df = result
                stocks[ts_code] = df

    print(f"Loaded {len(stocks)} stocks with sufficient data")
    return stocks


def get_stock_files(
    data_dir: str,
    market: str,
    max_stocks: int = None,
) -> List[Tuple[str, str]]:
    """
    Return list of (ts_code, filepath) without loading any data into memory.
    Used by memory-efficient mode to stream stocks from disk one at a time.
    """
    market_dir = os.path.join(data_dir, market)
    if not os.path.exists(market_dir):
        return []
    files = [os.path.join(market_dir, f) for f in os.listdir(market_dir) if f.endswith('.csv')]
    if max_stocks and len(files) > max_stocks:
        np.random.shuffle(files)
        files = files[:max_stocks]
    return [(os.path.basename(f).replace('.csv', ''), f) for f in files]


# ============================================================================
# Sequence Preparation
# ============================================================================

def pct_change_to_class(pct_change: float) -> int:
    """Convert percentage change to class label."""
    for i, (min_pct, max_pct, _) in enumerate(CHANGE_BUCKETS):
        if min_pct <= pct_change < max_pct:
            return i
    return NUM_CLASSES - 1


def pct_change_to_relative_class(pct_change: float) -> int:
    """Convert relative-return percentage (stock − CSI300) to class label."""
    for i, (min_pct, max_pct, _) in enumerate(RELATIVE_CHANGE_BUCKETS):
        if min_pct <= pct_change < max_pct:
            return i
    return NUM_RELATIVE_CLASSES - 1


def build_csi300_forward_returns(data_dir: str) -> Dict[int, List[float]]:
    """
    Build a per-date dict of CSI300 forward returns for each prediction horizon.

    Reads idx_factor_pro/000300_SH.csv (already downloaded as part of market context),
    extracts the sorted daily close price series, and computes close-to-close returns
    over each forward window using the same formula as stock labels:
        ret = (close[i + fw] - close[i]) / close[i] * 100

    Returns:
        {date_int (YYYYMMDD): [ret_fw3, ret_fw4, ret_fw5]}
        Dates within max(FORWARD_WINDOWS) of the end are omitted (no lookahead available).
    """
    ifp_path = os.path.join(data_dir, 'index', 'idx_factor_pro', '000300_SH.csv')
    if not os.path.exists(ifp_path):
        print("[csi300_fw] idx_factor_pro/000300_SH.csv not found — relative labels will be 'neutral'")
        return {}

    try:
        df = pd.read_csv(ifp_path, usecols=['trade_date', 'close'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = df['trade_date'].astype(int)
        closes = df['close'].values.astype('float64')
        dates  = df['trade_date'].values
        n = len(closes)
        max_fw = max(FORWARD_WINDOWS)

        fw_rets: Dict[int, List[float]] = {}
        for i in range(n - max_fw):
            d = int(dates[i])
            fw_rets[d] = [
                100.0 * (closes[i + fw] - closes[i]) / closes[i]
                for fw in FORWARD_WINDOWS
            ]
        print(f"[csi300_fw] Built CSI300 forward returns for {len(fw_rets)} trading days")
        return fw_rets
    except Exception as e:
        print(f"[csi300_fw] Warning: could not build CSI300 forward returns: {e}")
        return {}


def process_single_stock(
    args: Tuple
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Process a single stock to create sequences. Used for parallel processing."""
    # Unpack — stk_limit and moneyflow are optional trailing args
    ts_code, df, sector_data, daily_basic, sequence_length, max_sequences, sector_to_id, forward_window = args[:8]
    stk_limit = args[8] if len(args) > 8 else None
    moneyflow = args[9] if len(args) > 9 else None

    try:
        # Merge daily basic data
        df = merge_daily_basic(df, daily_basic, ts_code)

        # Limit up/down features
        df = merge_stk_limit(df, stk_limit, ts_code)

        # Money flow features
        df = merge_moneyflow(df, moneyflow, ts_code)

        # Calculate features
        df = calculate_technical_features(df)

        # Ensure every FEATURE_COLUMN exists — market context / index membership
        # are not available in standard (non-disk) mode, so default them to 0.
        missing_fc = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_fc:
            df = pd.concat(
                [df, pd.DataFrame(0.0, index=df.index, columns=missing_fc)], axis=1)

        # Get sector
        sector_info = sector_data[sector_data['ts_code'] == ts_code] if len(sector_data) > 0 else pd.DataFrame()
        sector = sector_info['sector'].values[0] if len(sector_info) > 0 else 'Unknown'
        sector_id = sector_to_id.get(sector, sector_to_id.get('Unknown', 0))

        # Remove NaN rows
        df = df.dropna(subset=FEATURE_COLUMNS)

        max_fw = max(FORWARD_WINDOWS)
        if len(df) < sequence_length + max_fw:
            return None

        # Extract features
        features = df[FEATURE_COLUMNS].values
        closes = df['close'].values
        highs = df['high'].values

        # Leave max(FORWARD_WINDOWS) rows at the end so we can look ahead for all horizons
        valid_indices = list(range(sequence_length, len(df) - max_fw + 1))

        if max_sequences and len(valid_indices) > max_sequences:
            valid_indices = list(np.random.choice(valid_indices, max_sequences, replace=False))

        # Create sequences
        sequences = []
        future_seqs = []
        labels = []
        sectors = []
        dates = []

        _fut_idx = np.array(_FUTURE_FEAT_IDX, dtype=np.intp)

        for i in valid_indices:
            seq = features[i - sequence_length:i]
            # Multi-horizon labels: one class per forward window (day 3, 4, 5).
            # Close-to-close return avoids max(high) path-dependency.
            label_vec = [
                pct_change_to_class(
                    100.0 * (closes[i + fw - 1] - closes[i - 1]) / closes[i - 1]
                )
                for fw in FORWARD_WINDOWS
            ]

            # Date of the last day in the sequence window (YYYYMMDD int) for time-based split
            date_val = int(df['trade_date'].iloc[i - 1].strftime('%Y%m%d'))

            # Known-future calendar features for TFT decoder (rows i..i+max_fw-1)
            # These rows always exist: valid_indices stops at len(df)-max_fw+1.
            future_feat = features[i:i + max_fw][:, _fut_idx]   # (max_fw, 27)

            sequences.append(seq)
            future_seqs.append(future_feat)
            labels.append(label_vec)
            sectors.append(sector_id)
            dates.append(date_val)

        if len(sequences) == 0:
            return None

        return (np.array(sequences),
                np.array(future_seqs, dtype=np.float32),   # (N, max_fw, 27)
                np.array(labels, dtype=np.int64),           # (N, NUM_HORIZONS)
                np.array(sectors),
                np.array(dates, dtype=np.int32))

    except Exception as e:
        return None


def _init_worker():
    """Initialize worker process with numba warmup."""
    from .numba_optimizations import warmup
    warmup()


def prepare_dataset(
    stocks: Dict[str, pd.DataFrame],
    sector_data: pd.DataFrame,
    daily_basic: pd.DataFrame = None,
    sequence_length: int = 30,
    forward_window: int = 5,
    max_sequences_per_stock: int = None,
    num_workers: int = 4,
    stk_limit: Optional[pd.DataFrame] = None,
    moneyflow: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare sequences with optional parallel processing.

    Note: With numba optimizations, sequential processing (num_workers=1) is often
    faster than multiprocessing due to process spawning and data serialization overhead.
    """
    # Create sector encoding
    if len(sector_data) > 0:
        sector_to_id = {sector: i for i, sector in enumerate(sector_data['sector'].unique())}
    else:
        sector_to_id = {}
    sector_to_id['Unknown'] = len(sector_to_id)

    # Handle empty daily_basic
    if daily_basic is None:
        daily_basic = pd.DataFrame()

    all_sequences = []
    all_future_seqs = []
    all_labels = []
    all_sectors = []
    all_dates = []

    # Sequential processing (recommended with numba - avoids multiprocessing overhead)
    if num_workers <= 1:
        print(f"Preparing sequences sequentially (numba-optimized)...")

        processed = 0
        for ts_code, df in stocks.items():
            args = (ts_code, df, sector_data, daily_basic, sequence_length,
                    max_sequences_per_stock, sector_to_id, forward_window,
                    stk_limit, moneyflow)
            result = process_single_stock(args)

            if result is not None:
                seqs, fut_seqs, labs, secs, dts = result
                all_sequences.append(seqs)
                all_future_seqs.append(fut_seqs)
                all_labels.append(labs)
                all_sectors.append(secs)
                all_dates.append(dts)

            processed += 1
            if processed % 50 == 0:
                total_seqs = sum(len(s) for s in all_sequences)
                print(f"Processed {processed}/{len(stocks)} stocks, {total_seqs} sequences...")
    else:
        # Parallel processing (has overhead from process spawning and data serialization)
        print(f"Preparing sequences using {num_workers} workers...")
        print("  (Tip: num_workers=1 may be faster with numba optimizations)")

        args_list = [
            (ts_code, df, sector_data, daily_basic, sequence_length,
             max_sequences_per_stock, sector_to_id, forward_window,
             stk_limit, moneyflow)
            for ts_code, df in stocks.items()
        ]

        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
            futures = {executor.submit(process_single_stock, args): args[0] for args in args_list}

            processed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    seqs, fut_seqs, labs, secs, dts = result
                    all_sequences.append(seqs)
                    all_future_seqs.append(fut_seqs)
                    all_labels.append(labs)
                    all_sectors.append(secs)
                    all_dates.append(dts)

                processed += 1
                if processed % 50 == 0:
                    total_seqs = sum(len(s) for s in all_sequences)
                    print(f"Processed {processed}/{len(stocks)} stocks, {total_seqs} sequences...")

    # Concatenate results
    if len(all_sequences) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([], dtype=np.int32)

    sequences    = np.concatenate(all_sequences,    axis=0).astype(np.float32)
    future_seqs  = np.concatenate(all_future_seqs,  axis=0).astype(np.float32)
    labels       = np.concatenate(all_labels,       axis=0)
    sectors      = np.concatenate(all_sectors,      axis=0)
    dates        = np.concatenate(all_dates,        axis=0)

    print(f"Total sequences created: {len(sequences)}")

    return sequences, future_seqs, labels, sectors, dates


def normalize_data(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    test_sequences: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Normalize sequences using StandardScaler fitted on training data."""
    n_train, seq_len, n_features = train_sequences.shape

    scaler = StandardScaler()
    train_flat = train_sequences.reshape(-1, n_features)
    scaler.fit(train_flat)

    train_normalized = scaler.transform(train_flat).reshape(n_train, seq_len, n_features)
    val_normalized = scaler.transform(val_sequences.reshape(-1, n_features)).reshape(val_sequences.shape)
    test_normalized = scaler.transform(test_sequences.reshape(-1, n_features)).reshape(test_sequences.shape)

    # Handle NaN/Inf
    train_normalized = np.nan_to_num(train_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    val_normalized = np.nan_to_num(val_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    test_normalized = np.nan_to_num(test_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return train_normalized, val_normalized, test_normalized, scaler


def split_data(
    sequences: np.ndarray,
    labels: np.ndarray,
    sectors: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    dates: Optional[np.ndarray] = None,  # unused — kept for API compatibility
) -> Tuple:
    """
    Split data into train/val/test using a random permutation.

    Random splitting across the full date range ensures val/test cover all
    market regimes (bull, bear, sideways) rather than a single fixed window,
    giving a more representative estimate of generalisation.

    Split ratio: 70 / 15 / 15 (default).
    Equal-sized val and test sets allow direct metric comparison between them.
    """
    n_samples = len(sequences)
    indices   = np.random.permutation(n_samples)
    train_end = int(n_samples * train_ratio)
    val_end   = train_end + int(n_samples * val_ratio)

    n_tr = train_end
    n_va = val_end - train_end
    n_te = n_samples - val_end
    print(f"Random split: train {n_tr:,}  val {n_va:,}  test {n_te:,}")

    return (
        sequences[indices[:train_end]],          labels[indices[:train_end]],          sectors[indices[:train_end]],
        sequences[indices[train_end:val_end]],   labels[indices[train_end:val_end]],   sectors[indices[train_end:val_end]],
        sequences[indices[val_end:]],            labels[indices[val_end:]],            sectors[indices[val_end:]],
    )


# ============================================================================
# Memory-Efficient Processing (Disk-Based)
# ============================================================================

def prepare_dataset_to_disk(
    stock_files: List[Tuple[str, str]],
    sector_data: pd.DataFrame,
    daily_basic: pd.DataFrame,
    output_dir: str,
    sequence_length: int = 30,
    forward_window: int = 5,
    min_data_points: int = 100,
    max_sequences_per_stock: int = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    market_context: Optional[pd.DataFrame] = None,
    index_membership: Optional[Dict[str, Dict[str, float]]] = None,
    stk_limit: Optional[pd.DataFrame] = None,
    moneyflow: Optional[pd.DataFrame] = None,
    split_mode: str = 'regime',
    data_dir: Optional[str] = None,
    cs_tech_stats: Optional[Dict] = None,
) -> Dict:
    """
    Process stocks and save directly to disk using memmap for memory efficiency.

    Streams stocks one-at-a-time from disk — the all_stocks dict is never
    built in RAM. daily_basic is pre-grouped by ts_code (O(1) lookup per stock)
    and freed immediately after grouping.

    Args:
        stock_files: List of (ts_code, filepath) — no DataFrames pre-loaded
        sector_data: Sector classification data
        daily_basic: Daily basic DataFrame (pre-grouped and freed internally)
        output_dir: Directory to save memmap files
        sequence_length: Length of each sequence
        min_data_points: Minimum rows required per stock
        max_sequences_per_stock: Maximum sequences per stock
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with metadata and scaler
    """
    from .memmap_dataset import MemmapDataWriter

    # Create sector encoding
    if len(sector_data) > 0:
        sector_to_id = {sector: i for i, sector in enumerate(sector_data['sector'].unique())}
    else:
        sector_to_id = {}
    sector_to_id['Unknown'] = len(sector_to_id)

    # Create industry encoding (more granular than sector)
    if len(sector_data) > 0 and 'industry' in sector_data.columns:
        industry_to_id = {ind: i for i, ind in enumerate(sector_data['industry'].dropna().unique())}
    else:
        industry_to_id = {}
    industry_to_id['Unknown'] = len(industry_to_id)

    # Build industry dict for O(1) per-stock lookup (mirrors sector_dict below)
    industry_dict: Dict[str, str] = {}
    if len(sector_data) > 0 and 'industry' in sector_data.columns:
        full_ind = sector_data.set_index('ts_code')['industry'].to_dict()
        for key, val in full_ind.items():
            industry_dict[str(key)] = str(val)
            industry_dict[str(key).split('.')[0]] = str(val)

    # Pre-group daily_basic by ts_code for O(1) per-stock lookup.
    # Keys are stored as bare numeric codes (e.g. '600000') so they match the
    # basenames of sh/sz CSV files regardless of exchange suffix in the data.
    daily_basic_dict: Dict[str, pd.DataFrame] = {}
    if daily_basic is not None and len(daily_basic) > 0:
        print("Pre-grouping daily basic data by stock code...")
        for key, group in daily_basic.groupby('ts_code'):
            bare = str(key).split('.')[0]   # '600000.SH' → '600000'
            daily_basic_dict[bare] = group.drop(columns=['ts_code'], errors='ignore').reset_index(drop=True)
        del daily_basic  # free the large concatenated DataFrame immediately
        gc.collect()
        print(f"  Pre-grouped {len(daily_basic_dict)} stocks")

    # Compute cross-section PE/PB stats from the already-loaded daily_basic_dict.
    # No extra file I/O: iterate once over the grouped DataFrames to build
    # {date_int → {feature → (median, std)}} used in apply_cs_normalization().
    daily_cs_stats: Dict[int, Dict[str, tuple]] = {}
    if daily_basic_dict:
        print("Computing daily cross-section valuation stats (PE/PB normalization)...")
        daily_cs_stats = compute_daily_cs_stats(daily_basic_dict)
        print(f"  CS stats ready for {len(daily_cs_stats)} trading days")

    # Pre-group stk_limit by bare ts_code for O(1) per-stock lookup.
    stk_limit_dict: Dict[str, pd.DataFrame] = {}
    if stk_limit is not None and len(stk_limit) > 0:
        print("Pre-grouping stk_limit data by stock code...")
        for key, group in stk_limit.groupby('ts_code'):
            bare = str(key).split('.')[0]
            stk_limit_dict[bare] = group.reset_index(drop=True)
        del stk_limit
        gc.collect()
        print(f"  Pre-grouped {len(stk_limit_dict)} stocks (stk_limit)")

    # Pre-group moneyflow by bare ts_code for O(1) per-stock lookup.
    moneyflow_dict: Dict[str, pd.DataFrame] = {}
    if moneyflow is not None and len(moneyflow) > 0:
        print("Pre-grouping moneyflow data by stock code...")
        for key, group in moneyflow.groupby('ts_code'):
            bare = str(key).split('.')[0]
            moneyflow_dict[bare] = group.reset_index(drop=True)
        del moneyflow
        gc.collect()
        print(f"  Pre-grouped {len(moneyflow_dict)} stocks (moneyflow)")

    # Build sector dict for O(1) lookup (bare code → sector name).
    sector_dict: Dict[str, str] = {}
    if len(sector_data) > 0:
        full_dict = sector_data.set_index('ts_code')['sector'].to_dict()
        for key, val in full_dict.items():
            sector_dict[str(key)] = val                    # full code (e.g. '600000.SH')
            sector_dict[str(key).split('.')[0]] = val      # bare code (e.g. '600000')

    # Build CSI300 forward-return lookup used for relative labels.
    # Dict: {date_int → [ret_fw3, ret_fw4, ret_fw5]}
    csi300_fw_rets: Dict[int, List[float]] = {}
    if data_dir is not None:
        csi300_fw_rets = build_csi300_forward_returns(data_dir)

    # Initialize writer
    writer = MemmapDataWriter(
        output_dir=output_dir,
        seq_length=sequence_length,
        n_features=len(FEATURE_COLUMNS),
        num_horizons=NUM_HORIZONS,
    )

    print(f"Processing {len(stock_files)} stocks to disk (streaming one at a time)...")

    processed = 0
    skipped = 0
    total_sequences = 0

    for ts_code, filepath in stock_files:
        try:
            # Load one stock from disk — freed at end of iteration
            df = pd.read_csv(filepath)
            if len(df) < min_data_points:
                skipped += 1
                continue

            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date').reset_index(drop=True)

            # O(1) dict lookup — key is bare code (e.g. '600000'); see groupby above
            stock_basic = daily_basic_dict.get(ts_code, pd.DataFrame())
            df = merge_daily_basic(df, stock_basic, ts_code=None)  # pre-filtered

            # Cross-section normalise per-stock PE/PB: replace absolute valuation
            # levels with daily z-scores (this stock's PE vs all peers today).
            # Makes features regime-invariant: a stock's relative valuation rank
            # is consistent across bull/bear even when market-wide PE shifts.
            if daily_cs_stats:
                df = apply_cs_normalization(df, daily_cs_stats)

            # Market-wide context (index valuation + global returns)
            if market_context is not None:
                df = merge_market_context(df, market_context)
            else:
                for col in MARKET_CONTEXT_FEATURES:
                    df[col] = 0.0

            # Per-stock index membership (CSI300/500/SSE50)
            if index_membership is not None:
                df = merge_index_membership(df, index_membership, ts_code)
            else:
                for col in INDEX_MEMBERSHIP_FEATURES:
                    df[col] = 0.0

            # Limit up/down features — O(1) dict lookup, pre-filtered DataFrame
            df = merge_stk_limit(df, stk_limit_dict.get(ts_code), None)

            # Money flow features — O(1) dict lookup, pre-filtered DataFrame
            df = merge_moneyflow(df, moneyflow_dict.get(ts_code), None)

            df = calculate_technical_features(df)

            # Cross-section normalise technical indicators: replace each feature
            # with its daily z-score across all stocks → removes market-regime level.
            if cs_tech_stats:
                df = apply_cs_normalization(df, cs_tech_stats, CS_NORMALIZE_TECH_FEATURES)

            # Sector + industry lookup — O(1) dict lookups
            sector      = sector_dict.get(ts_code, 'Unknown')
            sector_id   = sector_to_id.get(sector, sector_to_id.get('Unknown', 0))
            industry    = industry_dict.get(ts_code, 'Unknown')
            industry_id = industry_to_id.get(industry, industry_to_id.get('Unknown', 0))

            df = df.dropna(subset=FEATURE_COLUMNS)
            max_fw = max(FORWARD_WINDOWS)
            if len(df) < sequence_length + max_fw:
                del df
                continue

            # Cast to float32 at source — halves memory vs default float64
            features = df[FEATURE_COLUMNS].values.astype('float32')
            closes = df['close'].values
            highs = df['high'].values

            # Leave max(FORWARD_WINDOWS) rows at the end so we can look ahead for all horizons
            valid_indices = list(range(sequence_length, len(df) - max_fw + 1))
            if max_sequences_per_stock and len(valid_indices) > max_sequences_per_stock:
                valid_indices = list(np.random.choice(valid_indices, max_sequences_per_stock, replace=False))

            sequences = np.array([features[i - sequence_length:i] for i in valid_indices])

            # Known-future calendar features for TFT decoder: rows i..i+max_fw-1
            _fut_idx = np.array(_FUTURE_FEAT_IDX, dtype=np.intp)
            future_inputs_arr = np.array(
                [features[i:i + max_fw][:, _fut_idx] for i in valid_indices],
                dtype=np.float32,
            )   # (N, max_fw, 27)

            # Date of the last day in each sequence window (YYYYMMDD) for time-based split
            trade_date_ints = (
                df['trade_date'].dt.year * 10000
                + df['trade_date'].dt.month * 100
                + df['trade_date'].dt.day
            ).values.astype(np.int32)
            dates_arr = np.array([trade_date_ints[i - 1] for i in valid_indices], dtype=np.int32)

            # Main labels: cross-sectional relative returns (stock − CSI300).
            # Regime-invariant: model cannot win by predicting overall market direction.
            # Fall back to neutral when CSI300 data unavailable for a date.
            _neutral_fw = [0.0] * len(FORWARD_WINDOWS)
            labels = np.array([
                [
                    pct_change_to_class(
                        100.0 * (closes[i + fw - 1] - closes[i - 1]) / closes[i - 1]
                        - csi300_fw_rets.get(int(dates_arr[si]), _neutral_fw)[fw_idx]
                    )
                    for fw_idx, fw in enumerate(FORWARD_WINDOWS)
                ]
                for si, i in enumerate(valid_indices)
            ], dtype=np.int64)
            sectors_arr    = np.full(len(valid_indices), sector_id,   dtype=np.int64)
            industries_arr = np.full(len(valid_indices), industry_id, dtype=np.int64)

            # Auxiliary relative-return labels (same signal, different bucket boundaries).
            relative_labels = np.array([
                [
                    pct_change_to_relative_class(
                        100.0 * (closes[i + fw - 1] - closes[i - 1]) / closes[i - 1]
                        - csi300_fw_rets.get(int(dates_arr[si]), _neutral_fw)[fw_idx]
                    )
                    for fw_idx, fw in enumerate(FORWARD_WINDOWS)
                ]
                for si, i in enumerate(valid_indices)
            ], dtype=np.int64)

            writer.add_stock_data(sequences, labels, sectors_arr, dates_arr, industries_arr,
                                  relative_labels=relative_labels,
                                  future_inputs=future_inputs_arr)
            total_sequences += len(sequences)
            del df, features, sequences, future_inputs_arr, labels, relative_labels, sectors_arr, industries_arr

        except Exception as e:
            print(f"  Error processing {ts_code}: {e}")
            continue

        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{len(stock_files)} stocks, {total_sequences:,} sequences...")

    print(f"Processed {processed} stocks ({skipped} skipped), {total_sequences:,} total sequences")
    gc.collect()

    # Finalize: split, normalize, preshuffle, and save to memmap files
    print(f"\nFinalizing dataset...")
    result = writer.finalize(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
        split_mode=split_mode,
        data_dir=data_dir,
    )

    result['industry_to_id'] = industry_to_id
    return result
