"""
Historical Stock Data Acquisition for SH and SZ Markets
- Downloads 5 years of daily data
- Uses Tushare for historical data (basic account compatible)
- Gets stock list from AKShare (free, no registration)
- Supports resume from checkpoint
- Stores data in CSV format

Usage (PyCharm/Interactive):
    from get_original_data import run
    run('init')           # Initialize stock list
    run('download')       # Start downloading
    run('download', 500)  # Download 500 stocks per batch
    run('status')         # Check progress
"""

import tushare as ts
import akshare as ak
import pandas as pd
import os
import time
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR = Path('./stock_data')
CHECKPOINT_FILE = DATA_DIR / 'checkpoint.json'
STOCK_LIST_FILE = DATA_DIR / 'stock_list.csv'

# Rate limiting for basic Tushare account
# Basic account: ~200 calls per minute for daily endpoint
# Set to 0.1s for faster downloads; increase if you hit rate limits
CALL_INTERVAL = 0.1  # seconds between calls

# Date range (5 years from today)
END_DATE = datetime.now().strftime('%Y%m%d')
START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')


def init_tushare():
    """Initialize Tushare API connection."""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)
    return pro


def setup_directories():
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / 'sh').mkdir(exist_ok=True)
    (DATA_DIR / 'sz').mkdir(exist_ok=True)


def load_checkpoint():
    """Load download progress checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Checkpoint file corrupted, starting fresh. Error: {e}")
            return {'completed': [], 'failed': [], 'last_index': 0}
    return {'completed': [], 'failed': [], 'last_index': 0}


def save_checkpoint(checkpoint):
    """Save download progress checkpoint."""
    # Convert numpy types to native Python types for JSON serialization
    checkpoint_clean = {
        'completed': [str(x) for x in checkpoint['completed']],
        'failed': [str(x) for x in checkpoint['failed']],
        'last_index': int(checkpoint['last_index'])
    }
    # Write to temp file first, then rename (atomic operation)
    temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_clean, f, indent=2)
    temp_file.replace(CHECKPOINT_FILE)


def get_stock_list(refresh=False, max_retries=5):
    """
    Get list of all stocks in SH and SZ markets.
    Uses AKShare for stock list (free), Tushare for data.
    """
    if STOCK_LIST_FILE.exists() and not refresh:
        print("Loading cached stock list...")
        return pd.read_csv(STOCK_LIST_FILE)

    print("Fetching stock list from AKShare (free)...")

    # Get all A-share stocks with retry logic
    stock_list = None
    for attempt in range(max_retries):
        try:
            stock_list = ak.stock_info_a_code_name()
            break
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            print(f"  Connection error (attempt {attempt+1}/{max_retries}): {type(e).__name__}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to fetch stock list after {max_retries} attempts: {e}")

    if stock_list is None:
        raise Exception("Failed to fetch stock list")

    # stock_list has columns: code, name
    stock_list = stock_list.rename(columns={'code': 'symbol', 'name': 'name'})

    # Determine exchange based on code prefix
    # SH: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx (starts with 6)
    # SZ: 000xxx, 001xxx, 002xxx, 300xxx, 301xxx (starts with 0 or 3)
    def get_exchange(code):
        if code.startswith('6'):
            return 'SH'
        elif code.startswith('0') or code.startswith('3'):
            return 'SZ'
        else:
            return 'OTHER'

    stock_list['exchange'] = stock_list['symbol'].apply(get_exchange)

    # Filter only SH and SZ
    stock_list = stock_list[stock_list['exchange'].isin(['SH', 'SZ'])].copy()

    # Create ts_code format for Tushare API
    stock_list['ts_code'] = stock_list['symbol'] + '.' + stock_list['exchange']

    # Sort by symbol
    stock_list = stock_list.sort_values('symbol').reset_index(drop=True)

    # Save to cache
    stock_list.to_csv(STOCK_LIST_FILE, index=False)

    sh_count = len(stock_list[stock_list['exchange'] == 'SH'])
    sz_count = len(stock_list[stock_list['exchange'] == 'SZ'])
    print(f"Total stocks: {len(stock_list)} (SH: {sh_count}, SZ: {sz_count})")

    return stock_list


def download_stock_data(pro, ts_code, start_date, end_date, max_retries=3):
    """
    Download daily data for a single stock using Tushare.

    Returns DataFrame or None if failed.
    """
    for attempt in range(max_retries):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            error_msg = str(e).lower()
            if 'exceed' in error_msg or 'limit' in error_msg or '频率' in str(e):
                # Rate limit hit, wait longer
                wait_time = 60 * (attempt + 1)
                print(f"\n  Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif 'permission' in error_msg or '权限' in str(e):
                # Permission error - skip this stock
                print(f"\n  Permission denied for {ts_code}")
                return pd.DataFrame()  # Return empty df to mark as completed
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)

    return None


def save_stock_data(df, ts_code):
    """Save stock data to CSV file."""
    if df is None or df.empty:
        return False

    exchange = 'sh' if ts_code.endswith('.SH') else 'sz'
    symbol = ts_code.split('.')[0]
    filepath = DATA_DIR / exchange / f"{symbol}.csv"

    df.to_csv(filepath, index=False)
    return True


def get_download_stats(checkpoint, total_stocks):
    """Get download statistics."""
    completed = len(checkpoint['completed'])
    failed = len(checkpoint['failed'])
    remaining = total_stocks - completed - failed

    return {
        'total': total_stocks,
        'completed': completed,
        'failed': failed,
        'remaining': remaining,
        'progress': f"{(completed/total_stocks)*100:.1f}%" if total_stocks > 0 else "0%"
    }


def download_all_stocks(pro, stock_list, batch_size=None, start_from_index=None):
    """
    Download historical data for all stocks using Tushare.

    Args:
        pro: Tushare API instance
        stock_list: DataFrame with stock list
        batch_size: Optional limit on number of stocks to download in this run
        start_from_index: Optional index to start from (overrides checkpoint)
    """
    checkpoint = load_checkpoint()

    # Determine starting index
    if start_from_index is not None:
        start_idx = start_from_index
    else:
        start_idx = checkpoint.get('last_index', 0)

    completed_set = set(checkpoint['completed'])
    total_stocks = len(stock_list)
    stocks_processed = 0

    print(f"\n{'='*60}")
    print(f"Download Progress: {get_download_stats(checkpoint, total_stocks)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Starting from index: {start_idx}")
    print(f"Using: Tushare API")
    if batch_size:
        print(f"Batch size: {batch_size} stocks")
    print(f"{'='*60}\n")

    try:
        for idx in range(start_idx, total_stocks):
            row = stock_list.iloc[idx]
            ts_code = str(row['ts_code'])
            symbol = str(row['symbol'])

            # Skip if already completed
            if symbol in completed_set or ts_code in completed_set:
                continue

            # Check batch limit
            if batch_size and stocks_processed >= batch_size:
                print(f"\nBatch limit ({batch_size}) reached. Run again to continue.")
                break

            # Download data
            print(f"[{idx+1}/{total_stocks}] Downloading {ts_code} ({row.get('name', 'N/A')})...", end=' ', flush=True)

            df = download_stock_data(pro, ts_code, START_DATE, END_DATE)

            if df is not None and not df.empty:
                save_stock_data(df, ts_code)
                checkpoint['completed'].append(symbol)
                print(f"OK ({len(df)} rows)")
            elif df is not None and df.empty:
                # Stock exists but no data in range
                checkpoint['completed'].append(symbol)
                print("Empty (no data in range)")
            else:
                checkpoint['failed'].append(symbol)
                print("FAILED")

            checkpoint['last_index'] = int(idx + 1)
            stocks_processed += 1

            # Save checkpoint periodically
            if stocks_processed % 10 == 0:
                save_checkpoint(checkpoint)

            # Rate limiting
            time.sleep(CALL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")

    finally:
        save_checkpoint(checkpoint)
        stats = get_download_stats(checkpoint, total_stocks)
        print(f"\n{'='*60}")
        print(f"Session Complete!")
        print(f"Downloaded: {stocks_processed} stocks this session")
        print(f"Overall Progress: {stats}")
        print(f"{'='*60}")


def retry_failed(pro):
    """Retry downloading failed stocks."""
    checkpoint = load_checkpoint()
    failed_stocks = checkpoint['failed'].copy()

    if not failed_stocks:
        print("No failed stocks to retry.")
        return

    print(f"Retrying {len(failed_stocks)} failed stocks...")

    stock_list = get_stock_list()

    # Reset failed list
    checkpoint['failed'] = []
    save_checkpoint(checkpoint)

    for symbol in failed_stocks:
        # Get ts_code from stock list
        stock_row = stock_list[stock_list['symbol'] == symbol]
        if stock_row.empty:
            print(f"Skipping {symbol} (not in stock list)")
            continue

        ts_code = stock_row.iloc[0]['ts_code']
        name = stock_row.iloc[0].get('name', 'N/A')

        print(f"Retrying {ts_code} ({name})...", end=' ', flush=True)

        df = download_stock_data(pro, ts_code, START_DATE, END_DATE)

        if df is not None and not df.empty:
            save_stock_data(df, ts_code)
            checkpoint['completed'].append(symbol)
            print(f"OK ({len(df)} rows)")
        elif df is not None and df.empty:
            checkpoint['completed'].append(symbol)
            print("Empty")
        else:
            checkpoint['failed'].append(symbol)
            print("FAILED again")

        time.sleep(CALL_INTERVAL)

    save_checkpoint(checkpoint)


def show_status():
    """Show current download status."""
    if not STOCK_LIST_FILE.exists():
        print("Stock list not yet downloaded. Run: run('init')")
        return

    stock_list = pd.read_csv(STOCK_LIST_FILE)
    checkpoint = load_checkpoint()
    stats = get_download_stats(checkpoint, len(stock_list))

    print(f"\n{'='*60}")
    print("Download Status")
    print(f"{'='*60}")
    print(f"Total Stocks: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Remaining: {stats['remaining']}")
    print(f"Progress: {stats['progress']}")
    print(f"{'='*60}")

    if checkpoint['failed']:
        print(f"\nFailed stocks: {', '.join(checkpoint['failed'][:10])}")
        if len(checkpoint['failed']) > 10:
            print(f"  ... and {len(checkpoint['failed'])-10} more")


def reset_progress():
    """Reset all download progress (use with caution)."""
    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
    print("Progress reset. All data files retained.")


def run(command, batch=None):
    """
    Interactive entry point for PyCharm/Jupyter.

    Args:
        command: One of 'init', 'download', 'status', 'retry', 'reset', 'refresh'
        batch: Optional batch size for download command

    Examples:
        run('init')           # Get stock list
        run('download')       # Download all stocks
        run('download', 500)  # Download 500 stocks per run
        run('status')         # Show progress
        run('retry')          # Retry failed downloads
    """
    setup_directories()

    if command == 'status':
        show_status()
    elif command == 'reset':
        reset_progress()
    elif command == 'init':
        get_stock_list(refresh=False)
        print("Initialization complete. Run: run('download')")
    elif command == 'refresh':
        get_stock_list(refresh=True)
        print("Stock list refreshed.")
    elif command == 'retry':
        pro = init_tushare()
        retry_failed(pro)
    elif command == 'download':
        pro = init_tushare()
        stock_list = get_stock_list()
        download_all_stocks(pro, stock_list, batch_size=batch)
    else:
        print("Available commands: 'init', 'download', 'status', 'retry', 'reset', 'refresh'")
        print("Example: run('download', 500)")


def main():
    """Command line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Download SH/SZ stock historical data')
    parser.add_argument('--init', action='store_true', help='Initialize: fetch stock list')
    parser.add_argument('--download', action='store_true', help='Download stock data')
    parser.add_argument('--batch', type=int, default=None, help='Number of stocks to download per run')
    parser.add_argument('--retry', action='store_true', help='Retry failed downloads')
    parser.add_argument('--status', action='store_true', help='Show download status')
    parser.add_argument('--reset', action='store_true', help='Reset progress (keep data)')
    parser.add_argument('--refresh-list', action='store_true', help='Refresh stock list from API')

    # Use parse_known_args to ignore PyCharm console arguments
    args, unknown = parser.parse_known_args()

    setup_directories()

    if args.status:
        show_status()
        return

    if args.reset:
        reset_progress()
        return

    if args.init or args.refresh_list:
        get_stock_list(refresh=args.refresh_list)
        print("Initialization complete. Run with --download to start downloading.")
        return

    if args.retry:
        pro = init_tushare()
        retry_failed(pro)
        return

    if args.download:
        pro = init_tushare()
        stock_list = get_stock_list()
        download_all_stocks(pro, stock_list, batch_size=args.batch)
        return

    # Default: show help
    parser.print_help()
    print("\n" + "="*60)
    print("Quick Start (PyCharm Console):")
    print("  from get_original_data import run")
    print("  run('init')")
    print("  run('download')")
    print("  run('download', 500)  # batch mode")
    print("="*60)


if __name__ == '__main__':
    main()
