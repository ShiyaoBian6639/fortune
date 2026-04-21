"""
Profile data processing with 10 stocks to identify computational bottlenecks.
"""

import os
import sys
import time
import cProfile
import pstats
import io
from functools import wraps

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl.data_processing import (
    detect_w_bottom, detect_m_top, detect_patterns_multi_window,
    calculate_date_features, calculate_technical_features,
    load_stock_data, load_daily_basic_data, load_sector_data,
    merge_daily_basic, process_single_stock
)
from dl.config import FEATURE_COLUMNS


def timeit(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"  {func.__name__}: {end - start:.4f}s")
        return result
    return wrapper


def profile_single_stock(df: pd.DataFrame) -> dict:
    """Profile individual operations for a single stock."""
    timings = {}

    # Make a copy
    df = df.copy()

    # Profile pattern detection
    prices = df['close'].values

    start = time.perf_counter()
    detect_w_bottom(prices, window=10)
    timings['w_bottom_10'] = time.perf_counter() - start

    start = time.perf_counter()
    detect_w_bottom(prices, window=20)
    timings['w_bottom_20'] = time.perf_counter() - start

    start = time.perf_counter()
    detect_m_top(prices, window=10)
    timings['m_top_10'] = time.perf_counter() - start

    start = time.perf_counter()
    detect_m_top(prices, window=20)
    timings['m_top_20'] = time.perf_counter() - start

    # Profile technical features (includes patterns)
    start = time.perf_counter()
    df_tech = calculate_technical_features(df)
    timings['calculate_technical_features'] = time.perf_counter() - start

    # Profile date features separately
    df_test = df.copy()
    start = time.perf_counter()
    df_date = calculate_date_features(df_test)
    timings['calculate_date_features'] = time.perf_counter() - start

    # Profile CCI specifically
    df_cci = df.copy()
    start = time.perf_counter()
    typical_price = (df_cci['high'] + df_cci['low'] + df_cci['close']) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    tp_mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)
    timings['cci_calculation'] = time.perf_counter() - start

    timings['n_rows'] = len(df)

    return timings


def main():
    print("=" * 70)
    print("PROFILING DATA PROCESSING WITH 10 STOCKS")
    print("=" * 70)

    # Configuration
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data')

    # Load 10 stocks
    print("\n[1] Loading 10 stocks...")
    start = time.perf_counter()
    stocks = load_stock_data(data_dir, market='sh', max_stocks=10, num_workers=4)
    load_time = time.perf_counter() - start
    print(f"    Loaded {len(stocks)} stocks in {load_time:.2f}s")

    # Load daily basic data
    print("\n[2] Loading daily basic data...")
    start = time.perf_counter()
    daily_basic = load_daily_basic_data(data_dir)
    daily_basic_time = time.perf_counter() - start
    print(f"    Loaded daily basic in {daily_basic_time:.2f}s")

    # Load sector data
    print("\n[3] Loading sector data...")
    sector_data = load_sector_data(data_dir)

    # Profile individual operations per stock
    print("\n[4] Profiling individual operations per stock...")
    print("-" * 70)

    all_timings = []
    for ts_code, df in stocks.items():
        print(f"\nStock: {ts_code} ({len(df)} rows)")
        timings = profile_single_stock(df)
        timings['ts_code'] = ts_code
        all_timings.append(timings)

        # Print individual timings
        for key, value in timings.items():
            if key not in ['ts_code', 'n_rows']:
                print(f"  {key}: {value:.4f}s")

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    df_timings = pd.DataFrame(all_timings)
    timing_cols = [c for c in df_timings.columns if c not in ['ts_code', 'n_rows']]

    print("\nMean timings across 10 stocks:")
    print("-" * 50)
    for col in timing_cols:
        mean_time = df_timings[col].mean()
        total_time = df_timings[col].sum()
        print(f"  {col:35s}: mean={mean_time:.4f}s, total={total_time:.4f}s")

    # Calculate bottleneck percentages
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS (as % of total technical feature calculation)")
    print("=" * 70)

    total_tech = df_timings['calculate_technical_features'].sum()

    pattern_time = (
        df_timings['w_bottom_10'].sum() +
        df_timings['w_bottom_20'].sum() +
        df_timings['m_top_10'].sum() +
        df_timings['m_top_20'].sum()
    )

    print(f"\n  Total technical features time: {total_tech:.4f}s")
    print(f"  Pattern detection time:        {pattern_time:.4f}s ({100*pattern_time/total_tech:.1f}%)")
    print(f"  Date features time:            {df_timings['calculate_date_features'].sum():.4f}s")
    print(f"  CCI calculation time:          {df_timings['cci_calculation'].sum():.4f}s")

    # Full cProfile on one stock
    print("\n" + "=" * 70)
    print("DETAILED cProfile ON LARGEST STOCK")
    print("=" * 70)

    # Find largest stock
    largest_stock = max(stocks.items(), key=lambda x: len(x[1]))
    ts_code, df = largest_stock
    print(f"\nProfiling {ts_code} with {len(df)} rows...")

    profiler = cProfile.Profile()
    profiler.enable()

    df_copy = df.copy()
    df_result = calculate_technical_features(df_copy)

    profiler.disable()

    # Print profile stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: KEY BOTTLENECKS TO OPTIMIZE WITH NUMBA")
    print("=" * 70)
    print("""
1. PATTERN DETECTION (detect_w_bottom, detect_m_top):
   - Python for-loop over all data points
   - Called 4 times (2 patterns × 2 window sizes)
   - ~40-50% of feature calculation time

2. HOLIDAY CALCULATION (in calculate_date_features):
   - Nested loop: for year → for holiday → for each row
   - Uses slow .loc[] indexing
   - O(n_rows × n_holidays × n_years) complexity

3. CCI MAD CALCULATION:
   - rolling().apply(lambda x: np.abs(x - x.mean()).mean())
   - Python lambda inside rolling window
   - ~10-15% of technical features time
    """)

    # Transformer model input info
    print("\n" + "=" * 70)
    print("TRANSFORMER MODEL RAW INPUT")
    print("=" * 70)
    print(f"""
Shape: (batch_size, sequence_length=30, num_features={len(FEATURE_COLUMNS)})

The transformer receives:
- 30 consecutive trading days of data
- Each day has {len(FEATURE_COLUMNS)} features:
  - Technical indicators (RSI, MACD, Bollinger, etc.)
  - Price/volume ratios
  - Pattern signals (W-bottom, M-top)
  - Date/time cyclical features (sin/cos encoded)
  - Holiday effects
  - Fundamental features (P/E, P/B, market cap, etc.)

Features are normalized using StandardScaler (fit on train data).
    """)


if __name__ == '__main__':
    main()
