"""
Numba-optimized functions for data processing bottlenecks.

Main optimizations:
1. Holiday distance calculation - replaces slow pandas .loc[] loop
2. CCI MAD calculation - replaces rolling().apply(lambda)
3. Pattern detection - optimized W-bottom and M-top detection
"""

import numpy as np
from numba import jit, prange
from typing import Tuple


# ============================================================================
# Holiday Distance Calculation (MAIN BOTTLENECK - 93% of processing time)
# ============================================================================

@jit(nopython=True, cache=True)
def compute_holiday_distances(
    trade_dates: np.ndarray,  # int64 timestamps (nanoseconds)
    holiday_starts: np.ndarray,  # int64 timestamps
    holiday_ends: np.ndarray,  # int64 timestamps
    pre_holiday_days: int = 7,
    post_holiday_days: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute holiday-related features using pure numpy operations with numba JIT.

    Args:
        trade_dates: Array of trading date timestamps (int64 nanoseconds)
        holiday_starts: Array of holiday start timestamps
        holiday_ends: Array of holiday end timestamps
        pre_holiday_days: Days before holiday to flag as pre-holiday
        post_holiday_days: Days after holiday to flag as post-holiday

    Returns:
        is_pre_holiday: Binary array (1.0 if in pre-holiday period)
        is_post_holiday: Binary array (1.0 if in post-holiday period)
        days_to_holiday: Minimum days to nearest upcoming holiday
        days_from_holiday: Minimum days from most recent holiday
    """
    n = len(trade_dates)
    n_holidays = len(holiday_starts)

    # Initialize output arrays
    is_pre_holiday = np.zeros(n, dtype=np.float64)
    is_post_holiday = np.zeros(n, dtype=np.float64)
    days_to_holiday = np.full(n, 30.0, dtype=np.float64)
    days_from_holiday = np.full(n, 30.0, dtype=np.float64)

    # Nanoseconds per day
    ns_per_day = 86400 * 1_000_000_000
    pre_ns = pre_holiday_days * ns_per_day
    post_ns = post_holiday_days * ns_per_day

    for i in range(n):
        date_ns = trade_dates[i]

        for j in range(n_holidays):
            holiday_start_ns = holiday_starts[j]
            holiday_end_ns = holiday_ends[j]

            # Pre-holiday check
            if (date_ns >= holiday_start_ns - pre_ns) and (date_ns < holiday_start_ns):
                is_pre_holiday[i] = 1.0

            # Post-holiday check
            if (date_ns > holiday_end_ns) and (date_ns <= holiday_end_ns + post_ns):
                is_post_holiday[i] = 1.0

            # Days to holiday (for upcoming holidays)
            days_to = (holiday_start_ns - date_ns) / ns_per_day
            if 0 < days_to < days_to_holiday[i]:
                days_to_holiday[i] = days_to

            # Days from holiday (for past holidays)
            days_from = (date_ns - holiday_end_ns) / ns_per_day
            if 0 < days_from < days_from_holiday[i]:
                days_from_holiday[i] = days_from

    return is_pre_holiday, is_post_holiday, days_to_holiday, days_from_holiday


@jit(nopython=True, parallel=True, cache=True)
def compute_holiday_distances_parallel(
    trade_dates: np.ndarray,
    holiday_starts: np.ndarray,
    holiday_ends: np.ndarray,
    pre_holiday_days: int = 7,
    post_holiday_days: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel version of holiday distance computation using numba prange.
    """
    n = len(trade_dates)
    n_holidays = len(holiday_starts)

    is_pre_holiday = np.zeros(n, dtype=np.float64)
    is_post_holiday = np.zeros(n, dtype=np.float64)
    days_to_holiday = np.full(n, 30.0, dtype=np.float64)
    days_from_holiday = np.full(n, 30.0, dtype=np.float64)

    ns_per_day = 86400 * 1_000_000_000
    pre_ns = pre_holiday_days * ns_per_day
    post_ns = post_holiday_days * ns_per_day

    for i in prange(n):
        date_ns = trade_dates[i]

        for j in range(n_holidays):
            holiday_start_ns = holiday_starts[j]
            holiday_end_ns = holiday_ends[j]

            if (date_ns >= holiday_start_ns - pre_ns) and (date_ns < holiday_start_ns):
                is_pre_holiday[i] = 1.0

            if (date_ns > holiday_end_ns) and (date_ns <= holiday_end_ns + post_ns):
                is_post_holiday[i] = 1.0

            days_to = (holiday_start_ns - date_ns) / ns_per_day
            if 0 < days_to < days_to_holiday[i]:
                days_to_holiday[i] = days_to

            days_from = (date_ns - holiday_end_ns) / ns_per_day
            if 0 < days_from < days_from_holiday[i]:
                days_from_holiday[i] = days_from

    return is_pre_holiday, is_post_holiday, days_to_holiday, days_from_holiday


# ============================================================================
# CCI MAD Calculation (3.6% of processing time)
# ============================================================================

@jit(nopython=True, cache=True)
def rolling_mad(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling Mean Absolute Deviation (MAD) efficiently with numba.

    MAD = mean(|x - mean(x)|) for each rolling window

    Args:
        arr: Input array
        window: Rolling window size

    Returns:
        Array of rolling MAD values (NaN for first window-1 values)
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)

    # Fill first window-1 values with NaN
    for i in range(window - 1):
        result[i] = np.nan

    # Compute rolling MAD
    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        mean_val = 0.0
        for j in range(window):
            mean_val += window_data[j]
        mean_val /= window

        mad_val = 0.0
        for j in range(window):
            mad_val += abs(window_data[j] - mean_val)
        mad_val /= window

        result[i] = mad_val

    return result


@jit(nopython=True, cache=True)
def compute_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                window: int = 20) -> np.ndarray:
    """
    Compute Commodity Channel Index (CCI) efficiently with numba.

    CCI = (TP - SMA(TP)) / (0.015 * MAD(TP))
    where TP = (High + Low + Close) / 3

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size (default 20)

    Returns:
        CCI values (NaN for first window-1 values to match pandas behavior)
    """
    n = len(close)

    # Calculate typical price
    tp = np.empty(n, dtype=np.float64)
    for i in range(n):
        tp[i] = (high[i] + low[i] + close[i]) / 3.0

    # Calculate rolling SMA of typical price
    tp_sma = np.empty(n, dtype=np.float64)
    for i in range(window - 1):
        tp_sma[i] = np.nan

    for i in range(window - 1, n):
        sum_val = 0.0
        for j in range(window):
            sum_val += tp[i - window + 1 + j]
        tp_sma[i] = sum_val / window

    # Calculate rolling MAD of typical price
    tp_mad = rolling_mad(tp, window)

    # Calculate CCI
    cci = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(tp_sma[i]) or np.isnan(tp_mad[i]):
            cci[i] = np.nan  # Match pandas behavior: NaN for first window-1 values
        elif tp_mad[i] < 1e-10:
            cci[i] = 0.0  # Avoid division by zero
        else:
            cci[i] = (tp[i] - tp_sma[i]) / (0.015 * tp_mad[i])

    return cci


# ============================================================================
# Pattern Detection (0.6% - already fast, but optimized further)
# ============================================================================

@jit(nopython=True, cache=True)
def detect_w_bottom_numba(prices: np.ndarray, window: int = 20,
                          threshold: float = 0.02) -> np.ndarray:
    """
    Detect W-bottom pattern using numba JIT compilation.

    W-bottom pattern characteristics:
    - Two similar lows separated by a peak
    - Break above the middle peak confirms pattern

    Args:
        prices: Close prices array
        window: Window size to search for pattern
        threshold: Maximum difference ratio between two bottoms

    Returns:
        Signal array (0=no signal, 0.5=forming, 1.0=confirmed)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)

    if n < window:
        return signals

    mid = window // 2

    for i in range(window, n):
        # Extract segment
        segment = prices[i - window:i]

        # Find left minimum in first half
        left_min_idx = 0
        left_min_val = segment[0]
        for j in range(1, mid):
            if segment[j] < left_min_val:
                left_min_val = segment[j]
                left_min_idx = j

        # Find right minimum in second half
        right_min_idx = mid
        right_min_val = segment[mid]
        for j in range(mid + 1, window):
            if segment[j] < right_min_val:
                right_min_val = segment[j]
                right_min_idx = j

        # Find middle maximum between the two mins
        if right_min_idx <= left_min_idx:
            continue

        middle_max_idx = left_min_idx
        middle_max_val = segment[left_min_idx]
        for j in range(left_min_idx + 1, right_min_idx + 1):
            if segment[j] > middle_max_val:
                middle_max_val = segment[j]
                middle_max_idx = j

        # Check W-bottom pattern conditions
        if left_min_idx < middle_max_idx < right_min_idx:
            left_min = segment[left_min_idx]
            right_min = segment[right_min_idx]
            middle_max = segment[middle_max_idx]

            # Two bottoms should be at similar levels
            if abs(left_min - right_min) / (left_min + 1e-10) < threshold:
                # Middle peak should be above both bottoms
                if middle_max > left_min * (1 + threshold) and middle_max > right_min * (1 + threshold):
                    # Check breakout
                    if segment[-1] > middle_max:
                        signals[i] = 1.0
                    elif segment[-1] > (left_min + middle_max) / 2:
                        signals[i] = 0.5

    return signals


@jit(nopython=True, cache=True)
def detect_m_top_numba(prices: np.ndarray, window: int = 20,
                       threshold: float = 0.02) -> np.ndarray:
    """
    Detect M-top pattern using numba JIT compilation.

    M-top pattern characteristics:
    - Two similar highs separated by a trough
    - Break below the middle trough confirms pattern

    Args:
        prices: Close prices array
        window: Window size to search for pattern
        threshold: Maximum difference ratio between two tops

    Returns:
        Signal array (0=no signal, 0.5=forming, 1.0=confirmed)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)

    if n < window:
        return signals

    mid = window // 2

    for i in range(window, n):
        segment = prices[i - window:i]

        # Find left maximum in first half
        left_max_idx = 0
        left_max_val = segment[0]
        for j in range(1, mid):
            if segment[j] > left_max_val:
                left_max_val = segment[j]
                left_max_idx = j

        # Find right maximum in second half
        right_max_idx = mid
        right_max_val = segment[mid]
        for j in range(mid + 1, window):
            if segment[j] > right_max_val:
                right_max_val = segment[j]
                right_max_idx = j

        # Find middle minimum between the two maxes
        if right_max_idx <= left_max_idx:
            continue

        middle_min_idx = left_max_idx
        middle_min_val = segment[left_max_idx]
        for j in range(left_max_idx + 1, right_max_idx + 1):
            if segment[j] < middle_min_val:
                middle_min_val = segment[j]
                middle_min_idx = j

        # Check M-top pattern conditions
        if left_max_idx < middle_min_idx < right_max_idx:
            left_max = segment[left_max_idx]
            right_max = segment[right_max_idx]
            middle_min = segment[middle_min_idx]

            # Two tops should be at similar levels
            if abs(left_max - right_max) / (left_max + 1e-10) < threshold:
                # Middle trough should be below both tops
                if middle_min < left_max * (1 - threshold) and middle_min < right_max * (1 - threshold):
                    # Check breakdown
                    if segment[-1] < middle_min:
                        signals[i] = 1.0
                    elif segment[-1] < (left_max + middle_min) / 2:
                        signals[i] = 0.5

    return signals


@jit(nopython=True, cache=True)
def detect_patterns_multi_window_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect W-bottom and M-top patterns with multiple window sizes.

    Args:
        prices: Close prices array

    Returns:
        w_short: W-bottom signals with window=10
        w_long: W-bottom signals with window=20
        m_short: M-top signals with window=10
        m_long: M-top signals with window=20
    """
    w_short = detect_w_bottom_numba(prices, window=10)
    w_long = detect_w_bottom_numba(prices, window=20)
    m_short = detect_m_top_numba(prices, window=10)
    m_long = detect_m_top_numba(prices, window=20)

    return w_short, w_long, m_short, m_long


# ============================================================================
# Warm-up functions (compile JIT on first import)
# ============================================================================

def warmup():
    """Warm up numba JIT compilation with dummy data."""
    # Small dummy arrays for compilation
    dummy_prices = np.random.randn(100).astype(np.float64)
    dummy_dates = np.arange(100, dtype=np.int64) * 86400 * 1_000_000_000
    dummy_holiday_starts = np.array([50 * 86400 * 1_000_000_000], dtype=np.int64)
    dummy_holiday_ends = np.array([55 * 86400 * 1_000_000_000], dtype=np.int64)

    # Warm up all functions
    compute_holiday_distances(dummy_dates, dummy_holiday_starts, dummy_holiday_ends)
    compute_holiday_distances_parallel(dummy_dates, dummy_holiday_starts, dummy_holiday_ends)
    rolling_mad(dummy_prices, 20)
    compute_cci(dummy_prices, dummy_prices, dummy_prices, 20)
    detect_w_bottom_numba(dummy_prices, 20)
    detect_m_top_numba(dummy_prices, 20)
    detect_patterns_multi_window_numba(dummy_prices)


# Warm up on import
try:
    warmup()
except:
    pass  # Handle any compilation errors gracefully
