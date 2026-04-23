"""
Daily Basic Data Acquisition from Tushare Pro
- Fetches daily basic indicators for stocks
- Includes: PE, PB, PS, turnover rate, market cap, shares, etc.

Usage:
    from dailybasic import get_daily_basic, run

    # Get data for a specific date
    df = get_daily_basic(trade_date='20240408')

    # Get data for a specific stock
    df = get_daily_basic(ts_code='000001.SZ')

    # Run batch download
    run('download')
    run('download', start_date='20240101', end_date='20240408')
"""

import tushare as ts
import pandas as pd
import os
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR = Path('./stock_data/daily_basic')
CALL_INTERVAL = 0.3  # legacy: used only by single-date path
MAX_RETRIES = 5  # max retries for network errors
RETRY_DELAY = 2  # initial retry delay in seconds
# Parallelism for batch/range downloads. Tushare 8000-pt tier allows ~500/min
# for daily_basic; 8 workers × 8 calls/s keeps us well under per-minute quotas.
WORKERS = 8
CALLS_PER_SEC = 8.0


class _RateLimiter:
    """Token-bucket-style limiter shared across worker threads."""
    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._last + self._interval - now
                if wait <= 0:
                    self._last = now
                    return
            time.sleep(max(0.001, wait))


_limiter = _RateLimiter(CALLS_PER_SEC)


def init_tushare():
    """Initialize Tushare API connection."""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)
    return pro


def setup_directories():
    """Create necessary directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_with_retry(fetch_func, *args, max_retries=MAX_RETRIES, **kwargs):
    """
    Execute a fetch function with retry logic for network errors.

    Args:
        fetch_func: Function to call
        *args: Positional arguments for fetch_func
        max_retries: Maximum retry attempts
        **kwargs: Keyword arguments for fetch_func

    Returns:
        Result from fetch_func or None on failure
    """
    for attempt in range(max_retries):
        _limiter.acquire()
        try:
            return fetch_func(*args, **kwargs)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                Exception) as e:
            error_name = type(e).__name__
            err = str(e)
            # Tushare rate-limit responses — back off for a minute.
            if any(k in err for k in ('exceed', '超出', '频率', 'too many')):
                wait_time = 60 * (attempt + 1)
                print(f"  [rate limit] sleeping {wait_time}s ...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"  {error_name}: Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  {error_name}: Failed after {max_retries} attempts")
                return None
    return None


def get_daily_basic(ts_code=None, trade_date=None, start_date=None, end_date=None):
    """
    Fetch daily basic data from Tushare Pro.

    Args:
        ts_code: Stock code (e.g., '000001.SZ')
        trade_date: Single trade date (YYYYMMDD format)
        start_date: Start date for range query (YYYYMMDD)
        end_date: End date for range query (YYYYMMDD)

    Returns:
        DataFrame with columns:
        - ts_code: Stock code
        - trade_date: Trade date
        - close: Closing price
        - turnover_rate: Turnover rate (%)
        - turnover_rate_f: Turnover rate (free float)
        - volume_ratio: Volume ratio
        - pe: P/E ratio (total market cap / net profit)
        - pe_ttm: P/E ratio (TTM)
        - pb: P/B ratio (total market cap / net assets)
        - ps: P/S ratio
        - ps_ttm: P/S ratio (TTM)
        - dv_ratio: Dividend yield (%)
        - dv_ttm: Dividend yield (TTM) (%)
        - total_share: Total shares (10k shares)
        - float_share: Float shares (10k shares)
        - free_share: Free float shares (10k shares)
        - total_mv: Total market cap (10k CNY)
        - circ_mv: Circulating market cap (10k CNY)
    """
    pro = init_tushare()

    fields = [
        'ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f',
        'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
        'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share',
        'total_mv', 'circ_mv'
    ]

    df = pro.daily_basic(
        ts_code=ts_code or '',
        trade_date=trade_date or '',
        start_date=start_date or '',
        end_date=end_date or '',
        fields=','.join(fields)
    )

    return df


def download_daily_basic_by_date(trade_date, save=True):
    """
    Download daily basic data for a specific date with retry support.

    Args:
        trade_date: Date string in YYYYMMDD format
        save: Whether to save to CSV file

    Returns:
        DataFrame with daily basic data
    """
    setup_directories()

    print(f"Fetching daily basic data for {trade_date}...")
    df = fetch_with_retry(get_daily_basic, trade_date=trade_date)

    if df is not None and not df.empty:
        print(f"Retrieved {len(df)} records")
        if save:
            filepath = DATA_DIR / f'daily_basic_{trade_date}.csv'
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"Saved to {filepath}")
    else:
        print(f"No data available for {trade_date}")

    return df


def download_daily_basic_range(start_date, end_date, save=True, skip_existing=True):
    """
    Download daily basic data for a date range with retry and resume support.

    Args:
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        save: Whether to save to CSV files
        skip_existing: Skip dates that already have downloaded files

    Returns:
        List of DataFrames
    """
    setup_directories()
    pro = init_tushare()

    # Get trading calendar with retry
    print("Fetching trading calendar...")
    cal = fetch_with_retry(
        pro.trade_cal,
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )

    if cal is None or cal.empty:
        print("Failed to fetch trading calendar")
        return []

    trade_dates = cal['cal_date'].tolist()

    # Check which dates already exist
    if skip_existing:
        existing_dates = set()
        for f in DATA_DIR.glob('daily_basic_*.csv'):
            date_str = f.stem.replace('daily_basic_', '')
            existing_dates.add(date_str)

        remaining_dates = [d for d in trade_dates if d not in existing_dates]
        skipped = len(trade_dates) - len(remaining_dates)
        if skipped > 0:
            print(f"Skipping {skipped} already downloaded dates")
        trade_dates = remaining_dates

    print(f"Need to download {len(trade_dates)} trading days")

    if not trade_dates:
        print("All dates already downloaded!")
        return []

    def _fetch_one(date):
        df = fetch_with_retry(get_daily_basic, trade_date=date)
        if df is not None and not df.empty:
            if save:
                filepath = DATA_DIR / f'daily_basic_{date}.csv'
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
            return date, df, 'ok'
        if df is not None:
            return date, None, 'empty'
        return date, None, 'fail'

    results = []
    failed_dates = []
    total = len(trade_dates)
    ok = empty = fail = 0
    t0 = time.monotonic()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_fetch_one, d): d for d in trade_dates}
        done = 0
        for fut in as_completed(futures):
            done += 1
            date, df, status = fut.result()
            with lock:
                if status == 'ok':
                    ok += 1
                    results.append(df)
                elif status == 'empty':
                    empty += 1
                else:
                    fail += 1
                    failed_dates.append(date)
            if done % 20 == 0 or done == total:
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}]  ok={ok} empty={empty} fail={fail}  "
                      f"{rate:.1f} dates/s  ETA {eta:.0f}s")

    # Summary
    print(f"\nDownload complete in {time.monotonic()-t0:.0f}s: "
          f"{ok} succeeded, {empty} empty, {fail} failed")
    if failed_dates:
        print(f"Failed dates: {failed_dates[:10]}{'...' if len(failed_dates) > 10 else ''}")

    return results


def download_daily_basic_by_stock(ts_code, start_date=None, end_date=None, save=True):
    """
    Download daily basic data for a specific stock with retry support.

    Args:
        ts_code: Stock code (e.g., '000001.SZ')
        start_date: Start date (YYYYMMDD), defaults to 2017-01-01
        end_date: End date (YYYYMMDD), defaults to yesterday
        save: Whether to save to CSV file

    Returns:
        DataFrame with daily basic data
    """
    setup_directories()

    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    if start_date is None:
        start_date = '20170101'

    print(f"Fetching daily basic for {ts_code} from {start_date} to {end_date}...")
    df = fetch_with_retry(get_daily_basic, ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df is not None and not df.empty:
        df = df.sort_values('trade_date')
        print(f"Retrieved {len(df)} records")
        if save:
            code_clean = ts_code.replace('.', '_')
            filepath = DATA_DIR / f'daily_basic_{code_clean}.csv'
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"Saved to {filepath}")
    else:
        print(f"No data available for {ts_code}")

    return df


def run(action='download', **kwargs):
    """
    Main entry point for daily basic data operations.

    Args:
        action:
            'download' - Download today's data
            'range' - Download date range
            'stock' - Download for specific stock

        **kwargs:
            trade_date: Specific date (YYYYMMDD)
            start_date: Range start date
            end_date: Range end date
            ts_code: Stock code
    """
    if action == 'download':
        trade_date = kwargs.get('trade_date', (datetime.now() - timedelta(days=1)).strftime('%Y%m%d'))
        return download_daily_basic_by_date(trade_date)

    elif action == 'range':
        start_date = kwargs.get('start_date', '20170101')
        end_date = kwargs.get('end_date', (datetime.now() - timedelta(days=1)).strftime('%Y%m%d'))
        return download_daily_basic_range(start_date, end_date)

    elif action == 'stock':
        ts_code = kwargs.get('ts_code')
        if not ts_code:
            print("Error: ts_code is required for 'stock' action")
            return None
        return download_daily_basic_by_stock(
            ts_code,
            start_date=kwargs.get('start_date'),
            end_date=kwargs.get('end_date')
        )

    else:
        print(f"Unknown action: {action}")
        print("Available actions: download, range, stock")
        return None


if __name__ == '__main__':
    # Example: Download yesterday's daily basic data
    df = run('download')
    if df is not None:
        print("\nSample data:")
        print(df.head())
