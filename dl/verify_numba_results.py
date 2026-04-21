"""
Verify that numba-optimized functions produce identical results to original implementations.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl.data_processing import get_chinese_holidays_for_year, load_stock_data
from dl.numba_optimizations import (
    compute_holiday_distances, compute_cci,
    detect_w_bottom_numba, detect_m_top_numba, warmup
)


def original_holiday_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """Original implementation."""
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
    """Numba implementation."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df['year'] = df['trade_date'].dt.year

    all_holiday_starts = []
    all_holiday_ends = []

    for year in df['year'].unique():
        holidays = get_chinese_holidays_for_year(year)
        for holiday_date, holiday_name, duration in holidays:
            all_holiday_starts.append(holiday_date)
            all_holiday_ends.append(holiday_date + timedelta(days=duration))

    trade_dates_ns = df['trade_date'].values.astype('datetime64[ns]').astype(np.int64)
    holiday_starts_ns = np.array([np.datetime64(d).astype('datetime64[ns]').astype(np.int64)
                                   for d in all_holiday_starts], dtype=np.int64)
    holiday_ends_ns = np.array([np.datetime64(d).astype('datetime64[ns]').astype(np.int64)
                                 for d in all_holiday_ends], dtype=np.int64)

    is_pre, is_post, days_to, days_from = compute_holiday_distances(
        trade_dates_ns, holiday_starts_ns, holiday_ends_ns,
        pre_holiday_days=7, post_holiday_days=5
    )

    df['is_pre_holiday'] = is_pre
    df['is_post_holiday'] = is_post
    df['days_to_holiday'] = days_to
    df['days_from_holiday'] = days_from

    return df


def original_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Original CCI implementation."""
    typical_price = (high + low + close) / 3
    tp_series = pd.Series(typical_price)
    tp_sma = tp_series.rolling(window=20).mean()
    tp_mad = tp_series.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)
    return cci.values


def original_w_bottom(prices: np.ndarray, window: int = 20, threshold: float = 0.02) -> np.ndarray:
    """Original W-bottom detection."""
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


def original_m_top(prices: np.ndarray, window: int = 20, threshold: float = 0.02) -> np.ndarray:
    """Original M-top detection."""
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


def compare_arrays(name: str, original: np.ndarray, numba: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare two arrays and report differences."""
    # Handle NaN values
    orig_nan = np.isnan(original)
    numba_nan = np.isnan(numba)

    if not np.array_equal(orig_nan, numba_nan):
        print(f"  {name}: NaN positions differ!")
        print(f"    Original NaN count: {orig_nan.sum()}, Numba NaN count: {numba_nan.sum()}")
        return False

    # Compare non-NaN values
    mask = ~orig_nan
    if mask.sum() == 0:
        print(f"  {name}: All NaN - PASS")
        return True

    orig_valid = original[mask]
    numba_valid = numba[mask]

    if np.allclose(orig_valid, numba_valid, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(orig_valid - numba_valid))
        print(f"  {name}: PASS (max diff: {max_diff:.2e})")
        return True
    else:
        diff = np.abs(orig_valid - numba_valid)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        num_diff = np.sum(diff > atol)
        print(f"  {name}: FAIL")
        print(f"    Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"    Values differ at {num_diff}/{len(orig_valid)} positions")

        # Show some examples
        diff_idx = np.where(diff > atol)[0][:5]
        for idx in diff_idx:
            print(f"    idx={idx}: original={orig_valid[idx]:.6f}, numba={numba_valid[idx]:.6f}")
        return False


def main():
    print("=" * 70)
    print("VERIFICATION: NUMBA VS ORIGINAL IMPLEMENTATIONS")
    print("=" * 70)

    # Warm up numba
    print("\n[0] Warming up numba JIT...")
    warmup()

    # Load test data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data')
    print("\n[1] Loading 3 stocks for verification...")
    stocks = load_stock_data(data_dir, market='sh', max_stocks=3, num_workers=2)
    print(f"    Loaded {len(stocks)} stocks")

    all_passed = True

    for ts_code, df in stocks.items():
        print(f"\n{'=' * 70}")
        print(f"STOCK: {ts_code} ({len(df)} rows)")
        print("=" * 70)

        # =====================================================================
        # Test 1: Holiday Calculation
        # =====================================================================
        print("\n[TEST 1] Holiday Calculation:")

        result_orig = original_holiday_calculation(df)
        result_numba = numba_holiday_calculation(df)

        passed = True
        passed &= compare_arrays("is_pre_holiday",
                                  result_orig['is_pre_holiday'].values,
                                  result_numba['is_pre_holiday'].values)
        passed &= compare_arrays("is_post_holiday",
                                  result_orig['is_post_holiday'].values,
                                  result_numba['is_post_holiday'].values)
        passed &= compare_arrays("days_to_holiday",
                                  result_orig['days_to_holiday'].values,
                                  result_numba['days_to_holiday'].values)
        passed &= compare_arrays("days_from_holiday",
                                  result_orig['days_from_holiday'].values,
                                  result_numba['days_from_holiday'].values)

        if passed:
            print("  Holiday calculation: ALL PASSED ✓")
        else:
            print("  Holiday calculation: SOME FAILED ✗")
            all_passed = False

        # =====================================================================
        # Test 2: CCI Calculation
        # =====================================================================
        print("\n[TEST 2] CCI Calculation:")

        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)

        cci_orig = original_cci(high, low, close)
        cci_numba = compute_cci(high, low, close, window=20)

        if compare_arrays("cci", cci_orig, cci_numba, rtol=1e-4, atol=1e-6):
            print("  CCI calculation: PASSED ✓")
        else:
            print("  CCI calculation: FAILED ✗")
            all_passed = False

        # =====================================================================
        # Test 3: Pattern Detection
        # =====================================================================
        print("\n[TEST 3] Pattern Detection:")

        prices = df['close'].values.astype(np.float64)

        # W-bottom
        w_orig_10 = original_w_bottom(prices, window=10)
        w_numba_10 = detect_w_bottom_numba(prices, window=10)
        passed_w10 = compare_arrays("w_bottom_10", w_orig_10, w_numba_10)

        w_orig_20 = original_w_bottom(prices, window=20)
        w_numba_20 = detect_w_bottom_numba(prices, window=20)
        passed_w20 = compare_arrays("w_bottom_20", w_orig_20, w_numba_20)

        # M-top
        m_orig_10 = original_m_top(prices, window=10)
        m_numba_10 = detect_m_top_numba(prices, window=10)
        passed_m10 = compare_arrays("m_top_10", m_orig_10, m_numba_10)

        m_orig_20 = original_m_top(prices, window=20)
        m_numba_20 = detect_m_top_numba(prices, window=20)
        passed_m20 = compare_arrays("m_top_20", m_orig_20, m_numba_20)

        if passed_w10 and passed_w20 and passed_m10 and passed_m20:
            print("  Pattern detection: ALL PASSED ✓")
        else:
            print("  Pattern detection: SOME FAILED ✗")
            all_passed = False

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION RESULT")
    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED - Numba implementations produce identical results!")
    else:
        print("\n✗ SOME TESTS FAILED - Results differ between implementations!")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
