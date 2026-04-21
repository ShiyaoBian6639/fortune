"""
US Stock Data Acquisition using Tushare
- Downloads daily data from 2017-01-01 to present
- Uses trade_date based query (efficient: ~2000 calls vs 5500+ by stock)
- Supports resume from checkpoint
- Stores data in CSV format per stock

Usage (PyCharm/Interactive):
    from get_us_data import run
    run('download')       # Start downloading by date
    run('status')         # Check progress
    run('reset')          # Reset progress
"""

import tushare as ts
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
US_DATA_DIR = DATA_DIR / 'us'
CHECKPOINT_FILE = US_DATA_DIR / 'checkpoint.json'
DAILY_DATA_DIR = US_DATA_DIR / 'daily'  # Store raw daily files

# Rate limiting: us_daily allows 2 calls per minute
CALL_INTERVAL = 31  # seconds between calls

# Date range
START_DATE = '20170101'
END_DATE = datetime.now().strftime('%Y%m%d')


def init_tushare():
    """Initialize Tushare API connection."""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)
    return pro


def setup_directories():
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    US_DATA_DIR.mkdir(exist_ok=True)
    DAILY_DATA_DIR.mkdir(exist_ok=True)


def load_checkpoint():
    """Load download progress checkpoint."""
    default = {'completed_dates': [], 'last_date': None}
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                # Handle old checkpoint format
                if 'completed_dates' not in data:
                    print("Old checkpoint format detected, starting fresh.")
                    return default
                return data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Checkpoint file corrupted, starting fresh. Error: {e}")
            return default
    return default


def save_checkpoint(checkpoint):
    """Save download progress checkpoint."""
    temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    temp_file.replace(CHECKPOINT_FILE)


def get_trade_dates(start_date, end_date):
    """
    Generate list of potential trade dates (weekdays).
    Actual trading days will be confirmed by API returning data.
    """
    dates = []
    current = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    while current <= end:
        # Skip weekends (5=Saturday, 6=Sunday)
        if current.weekday() < 5:
            dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    return dates


def download_daily_data(pro, trade_date, max_retries=3):
    """
    Download all US stock data for a single trade date.

    Returns DataFrame or None if failed.
    """
    for attempt in range(max_retries):
        try:
            df = pro.us_daily(trade_date=trade_date)
            return df
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            if 'exceed' in error_lower or 'limit' in error_lower or '频率' in error_msg or '每分钟' in error_msg:
                wait_time = 60 * (attempt + 1)
                print(f"\n  Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n  Error for {trade_date}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)

    return None


def save_daily_data(df, trade_date):
    """Save daily data to CSV file."""
    if df is None or df.empty:
        return False

    filepath = DAILY_DATA_DIR / f"{trade_date}.csv"
    df.to_csv(filepath, index=False)
    return True


def download_all_dates(pro, batch_size=None):
    """
    Download US stock data by trade date.

    Args:
        pro: Tushare API instance
        batch_size: Optional limit on number of dates to download in this run
    """
    checkpoint = load_checkpoint()
    completed_set = set(checkpoint['completed_dates'])

    # Get all potential trade dates
    all_dates = get_trade_dates(START_DATE, END_DATE)

    # Filter out completed dates
    pending_dates = [d for d in all_dates if d not in completed_set]

    total_dates = len(all_dates)
    completed_count = len(completed_set)
    pending_count = len(pending_dates)

    print(f"\n{'='*60}")
    print(f"US Stock Download (by trade date)")
    print(f"{'='*60}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Total potential dates: {total_dates}")
    print(f"Already completed: {completed_count}")
    print(f"Pending: {pending_count}")
    if batch_size:
        print(f"Batch size: {batch_size} dates")
    print(f"Estimated time: ~{pending_count * CALL_INTERVAL // 60} minutes")
    print(f"{'='*60}\n")

    if not pending_dates:
        print("All dates already downloaded!")
        return

    dates_processed = 0

    try:
        for i, trade_date in enumerate(pending_dates):
            if batch_size and dates_processed >= batch_size:
                print(f"\nBatch limit ({batch_size}) reached. Run again to continue.")
                break

            progress = completed_count + dates_processed + 1
            print(f"[{progress}/{total_dates}] Downloading {trade_date}...", end=' ', flush=True)

            df = download_daily_data(pro, trade_date)

            if df is not None and not df.empty:
                save_daily_data(df, trade_date)
                checkpoint['completed_dates'].append(trade_date)
                checkpoint['last_date'] = trade_date
                print(f"OK ({len(df)} stocks)")
            elif df is not None and df.empty:
                # No trading on this date (holiday)
                checkpoint['completed_dates'].append(trade_date)
                checkpoint['last_date'] = trade_date
                print("Holiday/No trading")
            else:
                print("FAILED")

            dates_processed += 1

            # Save checkpoint periodically
            if dates_processed % 5 == 0:
                save_checkpoint(checkpoint)

            # Rate limiting
            if i < len(pending_dates) - 1:  # Don't wait after last call
                time.sleep(CALL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")

    finally:
        save_checkpoint(checkpoint)
        print(f"\n{'='*60}")
        print(f"Session Complete!")
        print(f"Downloaded: {dates_processed} dates this session")
        print(f"Total completed: {len(checkpoint['completed_dates'])}/{total_dates}")
        print(f"{'='*60}")


def merge_to_stock_files():
    """
    Merge daily CSV files into per-stock CSV files.
    Call this after downloading all daily data.
    """
    print("Merging daily files into per-stock files...")

    # Read all daily files
    daily_files = sorted(DAILY_DATA_DIR.glob('*.csv'))

    if not daily_files:
        print("No daily files found. Run download first.")
        return

    print(f"Found {len(daily_files)} daily files")

    # Combine all data
    all_data = []
    for f in daily_files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No data to merge")
        return

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total records: {len(combined)}")

    # Group by stock and save
    stocks = combined['ts_code'].unique()
    print(f"Unique stocks: {len(stocks)}")

    for ts_code in stocks:
        stock_data = combined[combined['ts_code'] == ts_code].copy()
        stock_data = stock_data.sort_values('trade_date')

        # Clean filename
        safe_name = ts_code.replace('/', '_').replace('\\', '_')
        filepath = US_DATA_DIR / f"{safe_name}.csv"
        stock_data.to_csv(filepath, index=False)

    print(f"Merged into {len(stocks)} stock files in {US_DATA_DIR}")


def show_status():
    """Show current download status."""
    checkpoint = load_checkpoint()
    all_dates = get_trade_dates(START_DATE, END_DATE)

    completed = len(checkpoint['completed_dates'])
    total = len(all_dates)
    remaining = total - completed

    print(f"\n{'='*60}")
    print("US Stock Download Status")
    print(f"{'='*60}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Total potential dates: {total}")
    print(f"Completed: {completed}")
    print(f"Remaining: {remaining}")
    print(f"Progress: {(completed/total)*100:.1f}%" if total > 0 else "0%")
    if checkpoint['last_date']:
        print(f"Last downloaded date: {checkpoint['last_date']}")
    print(f"{'='*60}")

    # Count daily files
    daily_files = list(DAILY_DATA_DIR.glob('*.csv')) if DAILY_DATA_DIR.exists() else []
    stock_files = [f for f in US_DATA_DIR.glob('*.csv') if f.name != 'us_stock_list.csv']
    print(f"\nDaily files: {len(daily_files)}")
    print(f"Stock files: {len(stock_files)}")


def reset_progress():
    """Reset all download progress (use with caution)."""
    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
    print("Progress reset. Data files retained.")


def run(command, batch=None):
    """
    Interactive entry point for PyCharm/Jupyter.

    Args:
        command: One of 'download', 'status', 'reset', 'merge'
        batch: Optional batch size for download command

    Examples:
        run('download')       # Download all dates
        run('download', 100)  # Download 100 dates per run
        run('status')         # Show progress
        run('merge')          # Merge daily files into per-stock files
    """
    setup_directories()

    if command == 'status':
        show_status()
    elif command == 'reset':
        reset_progress()
    elif command == 'merge':
        merge_to_stock_files()
    elif command == 'download':
        pro = init_tushare()
        download_all_dates(pro, batch_size=batch)
    else:
        print("Available commands: 'download', 'status', 'reset', 'merge'")
        print("Example: run('download', 100)")


def main():
    """Command line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Download US stock historical data by date')
    parser.add_argument('--download', action='store_true', help='Download stock data')
    parser.add_argument('--batch', type=int, default=None, help='Number of dates to download per run')
    parser.add_argument('--status', action='store_true', help='Show download status')
    parser.add_argument('--reset', action='store_true', help='Reset progress (keep data)')
    parser.add_argument('--merge', action='store_true', help='Merge daily files into per-stock files')

    args, unknown = parser.parse_known_args()

    setup_directories()

    if args.status:
        show_status()
        return

    if args.reset:
        reset_progress()
        return

    if args.merge:
        merge_to_stock_files()
        return

    if args.download:
        pro = init_tushare()
        download_all_dates(pro, batch_size=args.batch)
        return

    # Default: show help
    parser.print_help()
    print("\n" + "="*60)
    print("Quick Start (PyCharm Console):")
    print("  from get_us_data import run")
    print("  run('download')")
    print("  run('download', 100)  # batch mode")
    print("  run('merge')  # after download completes")
    print("="*60)


if __name__ == '__main__':
    main()
