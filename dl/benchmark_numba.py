"""
Benchmark numba optimizations vs original implementations.
Tests with 10 stocks to measure speedup.
"""

import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl.config import (
    SPRING_FESTIVAL_DATES, QINGMING_DATES, DRAGON_BOAT_DATES,
    MID_AUTUMN_DATES, DOUBLE_NINTH_DATES, WINTER_SOLSTICE_DATES,
    QIXI_DATES, LABA_DATES
)
from dl.data_processing import (
    load_stock_data, load_daily_basic_data, load_sector_data,
    get_chinese_holidays_for_year, calculate_technical_features
)
from dl.numba_optimizations import (
    compute_holiday_distances, compute_cci,
    detect_w_bottom_numba, detect_m_top_numba, warmup
)


def original_holiday_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """Original slow implementation of holiday calculations."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    df['year'] = df['trade_date'].dt.year

    df['is_pre_holiday'] = 0.0
    df['is_post_holiday'] = 0.0
    df['days_to_holiday'] = 30.0
    df['days_from_holiday'] = 30.0

    for year in df['year'].unique():
        holidays = get_chinese_holidays_for_year(year)
        for holiday_date, holiday_name, duration in holidays:
            holiday_start = holiday_date
            holiday_end = holiday_date + timedelta(days=duration)

            pre_holiday_start = holiday_start - timedelta(days=7)
            mask_pre = (df['trade_date'] >= pre_holiday_start) & (df['trade_date'] < holiday_start)
            df.loc[mask_pre, 'is_pre_holiday'] = 1.0

            post_holiday_end = holiday_end + timedelta(days=5)
            mask_post = (df['trade_date'] > holiday_end) & (df['trade_date'] <= post_holiday_end)
            df.loc[mask_post, 'is_post_holiday'] = 1.0

            # THE SLOW PART - nested loop with .loc[]
            for idx in df.index:
                date = df.loc[idx, 'trade_date']
                days_to = (holiday_start - date).days
                days_from = (date - holiday_end).days

                if 0 < days_to < df.loc[idx, 'days_to_holiday']:
                    df.loc[idx, 'days_to_holiday'] = days_to
                if 0 < days_from < df.loc[idx, 'days_from_holiday']:
                    df.loc[idx, 'days_from_holiday'] = days_from

    return df


def numba_holiday_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """Numba-optimized implementation of holiday calculations."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    df['year'] = df['trade_date'].dt.year

    # Collect all holidays
    all_holiday_starts = []
    all_holiday_ends = []

    for year in df['year'].unique():
        holidays = get_chinese_holidays_for_year(year)
        for holiday_date, holiday_name, duration in holidays:
            all_holiday_starts.append(holiday_date)
            all_holiday_ends.append(holiday_date + timedelta(days=duration))

    # Convert to numpy arrays
    trade_dates_ns = df['trade_date'].values.astype('datetime64[ns]').astype(np.int64)
    holiday_starts_ns = np.array([np.datetime64(d).astype('datetime64[ns]').astype(np.int64)
                                   for d in all_holiday_starts], dtype=np.int64)
    holiday_ends_ns = np.array([np.datetime64(d).astype('datetime64[ns]').astype(np.int64)
                                 for d in all_holiday_ends], dtype=np.int64)

    # Use numba function
    is_pre, is_post, days_to, days_from = compute_holiday_distances(
        trade_dates_ns, holiday_starts_ns, holiday_ends_ns,
        pre_holiday_days=7, post_holiday_days=5
    )

    df['is_pre_holiday'] = is_pre
    df['is_post_holiday'] = is_post
    df['days_to_holiday'] = days_to
    df['days_from_holiday'] = days_from

    return df


def original_cci(df: pd.DataFrame) -> np.ndarray:
    """Original slow CCI calculation with rolling().apply()."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    tp_mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)
    return cci.values


def numba_cci(df: pd.DataFrame) -> np.ndarray:
    """Numba-optimized CCI calculation."""
    return compute_cci(
        df['high'].values.astype(np.float64),
        df['low'].values.astype(np.float64),
        df['close'].values.astype(np.float64),
        window=20
    )


def original_pattern_detection(prices: np.ndarray, window: int = 20, threshold: float = 0.02) -> np.ndarray:
    """Original pattern detection (W-bottom)."""
    n = len(prices)
    signals = np.zeros(n)

    if n < window:
        return signals

    for i in range(window, n):
        segment = prices[i - window:i]
        mid = window // 2

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


def main():
    print("=" * 70)
    print("BENCHMARK: NUMBA OPTIMIZATIONS VS ORIGINAL")
    print("=" * 70)

    # Warm up numba JIT
    print("\n[0] Warming up numba JIT compilation...")
    warmup()
    print("    Done.")

    # Load test data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data')

    print("\n[1] Loading 10 stocks for benchmark...")
    stocks = load_stock_data(data_dir, market='sh', max_stocks=10, num_workers=4)
    print(f"    Loaded {len(stocks)} stocks")

    # ==========================================================================
    # Benchmark 1: Holiday Calculation (MAIN BOTTLENECK)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 1: HOLIDAY CALCULATION (MAIN BOTTLENECK)")
    print("=" * 70)

    original_times = []
    numba_times = []

    for ts_code, df in stocks.items():
        # Original
        start = time.perf_counter()
        result_original = original_holiday_calculation(df)
        original_time = time.perf_counter() - start
        original_times.append(original_time)

        # Numba
        start = time.perf_counter()
        result_numba = numba_holiday_calculation(df)
        numba_time = time.perf_counter() - start
        numba_times.append(numba_time)

        print(f"  {ts_code} ({len(df):4d} rows): original={original_time:.4f}s, numba={numba_time:.4f}s, speedup={original_time/numba_time:.1f}x")

    print(f"\n  TOTAL: original={sum(original_times):.4f}s, numba={sum(numba_times):.4f}s")
    print(f"  AVERAGE SPEEDUP: {sum(original_times)/sum(numba_times):.1f}x")

    # ==========================================================================
    # Benchmark 2: CCI Calculation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 2: CCI CALCULATION")
    print("=" * 70)

    original_times = []
    numba_times = []

    for ts_code, df in stocks.items():
        # Original
        start = time.perf_counter()
        result_original = original_cci(df)
        original_time = time.perf_counter() - start
        original_times.append(original_time)

        # Numba
        start = time.perf_counter()
        result_numba = numba_cci(df)
        numba_time = time.perf_counter() - start
        numba_times.append(numba_time)

        print(f"  {ts_code} ({len(df):4d} rows): original={original_time:.4f}s, numba={numba_time:.4f}s, speedup={original_time/numba_time:.1f}x")

    print(f"\n  TOTAL: original={sum(original_times):.4f}s, numba={sum(numba_times):.4f}s")
    print(f"  AVERAGE SPEEDUP: {sum(original_times)/sum(numba_times):.1f}x")

    # ==========================================================================
    # Benchmark 3: Pattern Detection
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 3: PATTERN DETECTION (W-BOTTOM)")
    print("=" * 70)

    original_times = []
    numba_times = []

    for ts_code, df in stocks.items():
        prices = df['close'].values.astype(np.float64)

        # Original
        start = time.perf_counter()
        result_original = original_pattern_detection(prices)
        original_time = time.perf_counter() - start
        original_times.append(original_time)

        # Numba
        start = time.perf_counter()
        result_numba = detect_w_bottom_numba(prices)
        numba_time = time.perf_counter() - start
        numba_times.append(numba_time)

        print(f"  {ts_code} ({len(df):4d} rows): original={original_time:.4f}s, numba={numba_time:.4f}s, speedup={original_time/numba_time:.1f}x")

    print(f"\n  TOTAL: original={sum(original_times):.4f}s, numba={sum(numba_times):.4f}s")
    print(f"  AVERAGE SPEEDUP: {sum(original_times)/sum(numba_times):.1f}x")

    # ==========================================================================
    # Benchmark 4: Full Technical Feature Calculation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 4: FULL TECHNICAL FEATURES (with numba)")
    print("=" * 70)

    total_time = 0
    total_rows = 0

    for ts_code, df in stocks.items():
        start = time.perf_counter()
        result = calculate_technical_features(df)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_rows += len(df)
        print(f"  {ts_code} ({len(df):4d} rows): {elapsed:.4f}s")

    print(f"\n  TOTAL: {total_time:.4f}s for {total_rows} rows")
    print(f"  AVG PER STOCK: {total_time/len(stocks):.4f}s")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key optimizations implemented:

1. HOLIDAY CALCULATION (was 93% of processing time):
   - Replaced nested for-loop with pandas .loc[] indexing
   - Now uses numba JIT-compiled function with numpy arrays
   - Expected speedup: 50-200x

2. CCI CALCULATION (was 3.6% of processing time):
   - Replaced rolling().apply(lambda) with numba rolling MAD
   - Expected speedup: 5-20x

3. PATTERN DETECTION (was 0.6% of processing time):
   - Replaced Python for-loop with numba JIT-compiled version
   - Expected speedup: 5-50x

TRANSFORMER MODEL RAW INPUT:
   - Shape: (batch_size, sequence_length=30, num_features=106)
   - Features include technical indicators, price/volume ratios,
     pattern signals, date/time features, holiday effects, and fundamentals
   - All features are normalized using StandardScaler fitted on training data
""")


if __name__ == '__main__':
    main()
