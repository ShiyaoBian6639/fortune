"""
Profile data processing to identify computational bottlenecks.
"""

import os
import sys
import time
import cProfile
import pstats
from io import StringIO
from functools import wraps

import numpy as np
import pandas as pd

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl.config import get_config, FEATURE_COLUMNS
from dl.data_processing import (
    load_stock_data, load_sector_data, load_daily_basic_data,
    calculate_technical_features, calculate_date_features,
    detect_w_bottom, detect_m_top, detect_patterns_multi_window,
    merge_daily_basic, process_single_stock, prepare_dataset
)


def time_function(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"  {func.__name__}: {end - start:.4f}s")
        return result
    return wrapper


def profile_single_stock_processing(df: pd.DataFrame, ts_code: str, daily_basic: pd.DataFrame):
    """Profile individual components of stock processing."""
    print(f"\n{'='*60}")
    print(f"Profiling single stock: {ts_code} ({len(df)} rows)")
    print('='*60)

    timings = {}

    # 1. Merge daily basic
    start = time.perf_counter()
    df_merged = merge_daily_basic(df.copy(), daily_basic, ts_code)
    timings['merge_daily_basic'] = time.perf_counter() - start
    print(f"  merge_daily_basic: {timings['merge_daily_basic']:.4f}s")

    # 2. Calculate technical features (which includes pattern detection + date features)
    start = time.perf_counter()
    df_features = calculate_technical_features(df_merged.copy())
    timings['calculate_technical_features_total'] = time.perf_counter() - start
    print(f"  calculate_technical_features (total): {timings['calculate_technical_features_total']:.4f}s")

    # Break down technical features
    print("\n  Breaking down calculate_technical_features:")

    df_test = df_merged.copy()

    # Basic features + moving averages + indicators (vectorized)
    start = time.perf_counter()
    df_test['returns'] = df_test['close'].pct_change()
    df_test['log_returns'] = np.log(df_test['close'] / df_test['close'].shift(1))
    df_test['high_low_ratio'] = df_test['high'] / df_test['low'] - 1
    df_test['close_open_ratio'] = df_test['close'] / df_test['open'] - 1
    for window in [5, 10, 20]:
        df_test[f'sma_{window}'] = df_test['close'].rolling(window=window).mean()
        df_test[f'sma_{window}_ratio'] = df_test['close'] / df_test[f'sma_{window}'] - 1
        df_test[f'vol_sma_{window}'] = df_test['vol'].rolling(window=window).mean()
        df_test[f'vol_sma_{window}_ratio'] = df_test['vol'] / df_test[f'vol_sma_{window}'] - 1
        df_test[f'volatility_{window}'] = df_test['returns'].rolling(window=window).std()
    timings['basic_features_and_ma'] = time.perf_counter() - start
    print(f"    basic_features + moving_averages: {timings['basic_features_and_ma']:.4f}s")

    # RSI
    start = time.perf_counter()
    delta = df_test['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df_test['rsi'] = 100 - (100 / (1 + rs))
    timings['rsi'] = time.perf_counter() - start
    print(f"    RSI: {timings['rsi']:.4f}s")

    # MACD
    start = time.perf_counter()
    ema12 = df_test['close'].ewm(span=12, adjust=False).mean()
    ema26 = df_test['close'].ewm(span=26, adjust=False).mean()
    df_test['macd'] = ema12 - ema26
    df_test['macd_signal'] = df_test['macd'].ewm(span=9, adjust=False).mean()
    df_test['macd_diff'] = df_test['macd'] - df_test['macd_signal']
    timings['macd'] = time.perf_counter() - start
    print(f"    MACD: {timings['macd']:.4f}s")

    # CCI with rolling apply (BOTTLENECK)
    start = time.perf_counter()
    typical_price = (df_test['high'] + df_test['low'] + df_test['close']) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    tp_mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df_test['cci'] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)
    timings['cci'] = time.perf_counter() - start
    print(f"    CCI (rolling apply - BOTTLENECK): {timings['cci']:.4f}s")

    # Pattern detection (BOTTLENECK)
    start = time.perf_counter()
    prices = df_test['close'].values
    w_short, w_long, m_short, m_long = detect_patterns_multi_window(prices)
    timings['pattern_detection'] = time.perf_counter() - start
    print(f"    Pattern detection (W/M - BOTTLENECK): {timings['pattern_detection']:.4f}s")

    # Date features (BOTTLENECK)
    start = time.perf_counter()
    df_date = calculate_date_features(df_test.copy())
    timings['date_features'] = time.perf_counter() - start
    print(f"    Date features (holiday loops - BOTTLENECK): {timings['date_features']:.4f}s")

    # Other vectorized calculations
    start = time.perf_counter()
    # Bollinger Bands
    bb_sma = df_test['close'].rolling(window=20).mean()
    bb_std = df_test['close'].rolling(window=20).std()
    # Stochastic
    low_14 = df_test['low'].rolling(window=14).min()
    high_14 = df_test['high'].rolling(window=14).max()
    df_test['stoch_k'] = 100 * (df_test['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df_test['stoch_d'] = df_test['stoch_k'].rolling(window=3).mean()
    # ATR
    high_low = df_test['high'] - df_test['low']
    high_close = abs(df_test['high'] - df_test['close'].shift(1))
    low_close = abs(df_test['low'] - df_test['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_test['atr_14'] = true_range.rolling(window=14).mean()
    timings['other_vectorized'] = time.perf_counter() - start
    print(f"    Other vectorized (BB, Stoch, ATR, etc.): {timings['other_vectorized']:.4f}s")

    return timings


def profile_full_pipeline(config):
    """Profile the full data processing pipeline."""
    print("\n" + "="*60)
    print("PROFILING FULL PIPELINE WITH 10 STOCKS")
    print("="*60)

    data_dir = config['data_dir']

    # Load sector data
    start = time.perf_counter()
    sector_data = load_sector_data(data_dir)
    sector_load_time = time.perf_counter() - start
    print(f"\n1. Load sector data: {sector_load_time:.4f}s")

    # Load daily basic data
    start = time.perf_counter()
    daily_basic = load_daily_basic_data(data_dir)
    daily_basic_load_time = time.perf_counter() - start
    print(f"2. Load daily basic data: {daily_basic_load_time:.4f}s ({len(daily_basic)} records)")

    # Load 10 stocks
    start = time.perf_counter()
    stocks = load_stock_data(data_dir, market='sh', max_stocks=10, num_workers=4)
    stock_load_time = time.perf_counter() - start
    print(f"3. Load 10 stocks: {stock_load_time:.4f}s ({len(stocks)} loaded)")

    # Profile single stock in detail
    if stocks:
        ts_code, df = list(stocks.items())[0]
        single_stock_timings = profile_single_stock_processing(df, ts_code, daily_basic)

    # Profile full dataset preparation
    print(f"\n{'='*60}")
    print("PROFILING prepare_dataset (10 stocks)")
    print("="*60)

    start = time.perf_counter()
    sequences, labels, sectors = prepare_dataset(
        stocks, sector_data, daily_basic,
        sequence_length=30,
        max_sequences_per_stock=600,
        num_workers=1  # Use 1 worker for profiling
    )
    prepare_time = time.perf_counter() - start
    print(f"\nTotal prepare_dataset time: {prepare_time:.4f}s")
    print(f"Sequences shape: {sequences.shape}")

    # Summary
    print(f"\n{'='*60}")
    print("BOTTLENECK SUMMARY")
    print("="*60)

    if 'single_stock_timings' in dir():
        bottlenecks = [
            ('CCI (rolling apply)', single_stock_timings.get('cci', 0)),
            ('Pattern detection (W/M)', single_stock_timings.get('pattern_detection', 0)),
            ('Date features (holidays)', single_stock_timings.get('date_features', 0)),
        ]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)

        print("\nTop bottlenecks per stock (sorted by time):")
        for name, time_s in bottlenecks:
            print(f"  {name}: {time_s:.4f}s")

        total_bottleneck = sum(t for _, t in bottlenecks)
        total_features = single_stock_timings.get('calculate_technical_features_total', 1)
        print(f"\nBottleneck contribution: {100*total_bottleneck/total_features:.1f}% of feature calculation")
        print(f"Estimated time for 10 stocks: {10 * total_features:.2f}s")
        print(f"Estimated time for 100 stocks: {100 * total_features:.2f}s")

    return sequences, labels, sectors


def main():
    """Main profiling entry point."""
    config = get_config(max_stocks=10)

    print("="*60)
    print("TRANSFORMER MODEL RAW INPUT")
    print("="*60)
    print(f"\nInput shape: (batch_size, seq_len={config['sequence_length']}, num_features={len(FEATURE_COLUMNS)})")
    print(f"\nFeature columns ({len(FEATURE_COLUMNS)} features):")
    for i, col in enumerate(FEATURE_COLUMNS):
        print(f"  {i+1:3d}. {col}")

    # Profile the pipeline
    sequences, labels, sectors = profile_full_pipeline(config)

    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR NUMBA OPTIMIZATION")
    print("="*60)
    print("""
1. Pattern Detection (detect_w_bottom, detect_m_top):
   - Convert to @numba.jit(nopython=True)
   - Use pure numpy arrays, no Python objects

2. CCI MAD Calculation:
   - Replace pandas rolling().apply(lambda) with numba rolling MAD
   - Pre-compute in a single pass with numba

3. Holiday Date Features:
   - Replace df.loc[idx] iteration with vectorized numpy operations
   - Pre-compute holiday distance arrays with numba
""")


if __name__ == '__main__':
    main()
