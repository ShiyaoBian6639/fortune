"""
Stock News Acquisition Module using Tushare API

Downloads 5 years of historical news for each stock.
Supports resume from checkpoint.

Usage (PyCharm/Interactive):
    from features.news import run
    run('init')               # Initialize (load stock list)
    run('download')           # Download news for all stocks
    run('download', 100)      # Download news for 100 stocks per batch
    run('status')             # Check progress
    run('retry')              # Retry failed downloads

Quick queries:
    run('major')              # Get today's major news
    run('cctv')               # Get CCTV news
    run('search', '茅台')      # Search news by keyword
"""

import tushare as ts
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configuration - reuse token from project
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR = Path('./stock_data')
NEWS_DIR = DATA_DIR / 'news'
CHECKPOINT_FILE = NEWS_DIR / 'news_checkpoint.json'
STOCK_LIST_FILE = DATA_DIR / 'stock_list.csv'

# Rate limiting for basic Tushare account
CALL_INTERVAL = 0.3  # seconds between calls

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
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    (NEWS_DIR / 'sh').mkdir(exist_ok=True)
    (NEWS_DIR / 'sz').mkdir(exist_ok=True)


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
    checkpoint_clean = {
        'completed': [str(x) for x in checkpoint['completed']],
        'failed': [str(x) for x in checkpoint['failed']],
        'last_index': int(checkpoint['last_index'])
    }
    temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_clean, f, indent=2)
    temp_file.replace(CHECKPOINT_FILE)


def get_stock_list():
    """Load stock list from cache."""
    if not STOCK_LIST_FILE.exists():
        print("Stock list not found. Run 'run(\"init\")' from get_original_data first.")
        print("Or run: from get_original_data import run; run('init')")
        return None
    return pd.read_csv(STOCK_LIST_FILE)


def get_stock_name(pro, ts_code):
    """Get stock name from ts_code."""
    try:
        df = pro.stock_basic(ts_code=ts_code, fields='ts_code,name')
        if df is not None and not df.empty:
            return df.iloc[0]['name']
    except Exception:
        pass
    return None


def download_stock_news(pro, ts_code, stock_name, start_date, end_date, max_retries=3):
    """
    Download news for a single stock over date range.

    Uses multiple strategies to find stock-related news:
    1. Search by stock name in news content
    2. Search by stock code

    Returns DataFrame or None if failed.
    """
    all_news = []

    # Split into smaller date chunks (1 year each) to avoid API limits
    current_start = datetime.strptime(start_date, '%Y%m%d')
    final_end = datetime.strptime(end_date, '%Y%m%d')

    while current_start < final_end:
        chunk_end = min(current_start + timedelta(days=365), final_end)
        chunk_start_str = current_start.strftime('%Y%m%d')
        chunk_end_str = chunk_end.strftime('%Y%m%d')

        for attempt in range(max_retries):
            try:
                # Try to get news mentioning this stock
                df = pro.news(
                    src='',
                    start_date=chunk_start_str,
                    end_date=chunk_end_str,
                    limit=1000
                )

                if df is not None and not df.empty:
                    # Filter news containing stock name or code
                    symbol = ts_code.split('.')[0]
                    mask = pd.Series([False] * len(df))

                    if 'title' in df.columns:
                        mask |= df['title'].str.contains(stock_name, na=False, regex=False)
                        mask |= df['title'].str.contains(symbol, na=False)
                    if 'content' in df.columns:
                        mask |= df['content'].str.contains(stock_name, na=False, regex=False)
                        mask |= df['content'].str.contains(symbol, na=False)

                    filtered = df[mask].copy()
                    if not filtered.empty:
                        filtered['ts_code'] = ts_code
                        filtered['stock_name'] = stock_name
                        all_news.append(filtered)

                break  # Success, move to next chunk

            except Exception as e:
                error_msg = str(e).lower()
                if 'exceed' in error_msg or 'limit' in error_msg or '频率' in str(e):
                    wait_time = 60 * (attempt + 1)
                    print(f"\n  Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif 'permission' in error_msg or '权限' in str(e):
                    # Try alternative: use major_news
                    try:
                        df = pro.major_news(
                            start_date=chunk_start_str,
                            end_date=chunk_end_str,
                            limit=500
                        )
                        if df is not None and not df.empty:
                            symbol = ts_code.split('.')[0]
                            mask = pd.Series([False] * len(df))
                            if 'title' in df.columns:
                                mask |= df['title'].str.contains(stock_name, na=False, regex=False)
                                mask |= df['title'].str.contains(symbol, na=False)
                            if 'content' in df.columns:
                                mask |= df['content'].str.contains(stock_name, na=False, regex=False)

                            filtered = df[mask].copy()
                            if not filtered.empty:
                                filtered['ts_code'] = ts_code
                                filtered['stock_name'] = stock_name
                                all_news.append(filtered)
                        break
                    except Exception:
                        break
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)

        current_start = chunk_end
        time.sleep(CALL_INTERVAL)

    if all_news:
        result = pd.concat(all_news, ignore_index=True)
        result = result.drop_duplicates(subset=['title', 'pub_time'] if 'pub_time' in result.columns else ['title'])
        return result

    return pd.DataFrame()


def save_stock_news(df, ts_code):
    """Save stock news to CSV file."""
    if df is None or df.empty:
        return False

    exchange = 'sh' if ts_code.endswith('.SH') else 'sz'
    symbol = ts_code.split('.')[0]
    filepath = NEWS_DIR / exchange / f"{symbol}_news.csv"

    df.to_csv(filepath, index=False, encoding='utf-8-sig')
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


def download_all_news(pro, stock_list, batch_size=None, start_from_index=None):
    """
    Download historical news for all stocks.

    Args:
        pro: Tushare API instance
        stock_list: DataFrame with stock list
        batch_size: Optional limit on number of stocks per run
        start_from_index: Optional index to start from
    """
    checkpoint = load_checkpoint()

    if start_from_index is not None:
        start_idx = start_from_index
    else:
        start_idx = checkpoint.get('last_index', 0)

    completed_set = set(checkpoint['completed'])
    total_stocks = len(stock_list)
    stocks_processed = 0

    print(f"\n{'='*60}")
    print(f"News Download Progress: {get_download_stats(checkpoint, total_stocks)}")
    print(f"Date Range: {START_DATE} to {END_DATE} (5 years)")
    print(f"Starting from index: {start_idx}")
    if batch_size:
        print(f"Batch size: {batch_size} stocks")
    print(f"{'='*60}\n")

    # Pre-fetch stock names
    print("Loading stock names...")
    stock_names = {}
    try:
        all_stocks = pro.stock_basic(fields='ts_code,name')
        if all_stocks is not None:
            stock_names = dict(zip(all_stocks['ts_code'], all_stocks['name']))
    except Exception as e:
        print(f"Warning: Could not load stock names: {e}")

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

            # Get stock name
            stock_name = stock_names.get(ts_code) or row.get('name', symbol)

            print(f"[{idx+1}/{total_stocks}] Downloading news for {ts_code} ({stock_name})...", end=' ', flush=True)

            df = download_stock_news(pro, ts_code, stock_name, START_DATE, END_DATE)

            if df is not None and not df.empty:
                save_stock_news(df, ts_code)
                checkpoint['completed'].append(symbol)
                print(f"OK ({len(df)} articles)")
            elif df is not None:
                checkpoint['completed'].append(symbol)
                print("No news found")
            else:
                checkpoint['failed'].append(symbol)
                print("FAILED")

            checkpoint['last_index'] = int(idx + 1)
            stocks_processed += 1

            if stocks_processed % 10 == 0:
                save_checkpoint(checkpoint)

            time.sleep(CALL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")

    finally:
        save_checkpoint(checkpoint)
        stats = get_download_stats(checkpoint, total_stocks)
        print(f"\n{'='*60}")
        print(f"Session Complete!")
        print(f"Processed: {stocks_processed} stocks this session")
        print(f"Overall Progress: {stats}")
        print(f"{'='*60}")


def retry_failed(pro):
    """Retry downloading news for failed stocks."""
    checkpoint = load_checkpoint()
    failed_stocks = checkpoint['failed'].copy()

    if not failed_stocks:
        print("No failed stocks to retry.")
        return

    print(f"Retrying {len(failed_stocks)} failed stocks...")

    stock_list = get_stock_list()
    if stock_list is None:
        return

    # Pre-fetch stock names
    stock_names = {}
    try:
        all_stocks = pro.stock_basic(fields='ts_code,name')
        if all_stocks is not None:
            stock_names = dict(zip(all_stocks['ts_code'], all_stocks['name']))
    except Exception:
        pass

    checkpoint['failed'] = []
    save_checkpoint(checkpoint)

    for symbol in failed_stocks:
        stock_row = stock_list[stock_list['symbol'] == symbol]
        if stock_row.empty:
            print(f"Skipping {symbol} (not in stock list)")
            continue

        ts_code = stock_row.iloc[0]['ts_code']
        stock_name = stock_names.get(ts_code) or stock_row.iloc[0].get('name', symbol)

        print(f"Retrying {ts_code} ({stock_name})...", end=' ', flush=True)

        df = download_stock_news(pro, ts_code, stock_name, START_DATE, END_DATE)

        if df is not None and not df.empty:
            save_stock_news(df, ts_code)
            checkpoint['completed'].append(symbol)
            print(f"OK ({len(df)} articles)")
        elif df is not None:
            checkpoint['completed'].append(symbol)
            print("No news found")
        else:
            checkpoint['failed'].append(symbol)
            print("FAILED again")

        time.sleep(CALL_INTERVAL)

    save_checkpoint(checkpoint)


def show_status():
    """Show current download status."""
    stock_list = get_stock_list()
    if stock_list is None:
        return

    checkpoint = load_checkpoint()
    stats = get_download_stats(checkpoint, len(stock_list))

    print(f"\n{'='*60}")
    print("News Download Status")
    print(f"{'='*60}")
    print(f"Total Stocks: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Remaining: {stats['remaining']}")
    print(f"Progress: {stats['progress']}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"{'='*60}")

    if checkpoint['failed']:
        print(f"\nFailed stocks: {', '.join(checkpoint['failed'][:10])}")
        if len(checkpoint['failed']) > 10:
            print(f"  ... and {len(checkpoint['failed'])-10} more")


def reset_progress():
    """Reset all download progress."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    print("News download progress reset. Data files retained.")


# Quick query functions (keep original functionality)
def get_major_news(pro, start_date=None, end_date=None, src=None, limit=100):
    """Get major financial news."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = end_date

    try:
        df = pro.major_news(
            start_date=start_date,
            end_date=end_date,
            src=src,
            limit=limit
        )
        return df
    except Exception as e:
        print(f"Error fetching major news: {e}")
        return pd.DataFrame()


def get_cctv_news(pro, date=None):
    """Get CCTV news."""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    try:
        df = pro.cctv_news(date=date)
        return df
    except Exception as e:
        print(f"Error fetching CCTV news: {e}")
        return pd.DataFrame()


def search_news(pro, keyword, start_date=None, end_date=None, limit=100):
    """Search news by keyword."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

    try:
        df = pro.major_news(
            start_date=start_date,
            end_date=end_date,
            limit=limit * 2
        )

        if df is None or df.empty:
            return pd.DataFrame()

        mask = df['title'].str.contains(keyword, na=False)
        if 'content' in df.columns:
            mask |= df['content'].str.contains(keyword, na=False)

        return df[mask].head(limit)
    except Exception as e:
        print(f"Error searching news: {e}")
        return pd.DataFrame()


def display_news(df, max_items=10):
    """Display news in a readable format."""
    if df is None or df.empty:
        print("No news found.")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(df)} news items (showing first {min(len(df), max_items)})")
    print(f"{'='*60}\n")

    for i, (idx, row) in enumerate(df.head(max_items).iterrows()):
        title = row.get('title', 'No title')
        pub_time = row.get('pub_time', row.get('date', 'Unknown time'))
        src = row.get('src', 'Unknown source')

        print(f"[{i+1}] {title}")
        print(f"    Source: {src} | Time: {pub_time}")
        print()


def run(command, arg=None, **kwargs):
    """
    Interactive entry point for PyCharm/Jupyter.

    Batch download commands:
        run('init')               # Verify stock list exists
        run('download')           # Download news for all stocks
        run('download', 100)      # Download 100 stocks per batch
        run('status')             # Check progress
        run('retry')              # Retry failed downloads
        run('reset')              # Reset progress

    Quick query commands:
        run('major')              # Get today's major news
        run('major', '20260301')  # Get major news for date
        run('cctv')               # Get CCTV news
        run('search', '茅台')      # Search news by keyword
    """
    setup_directories()

    # Batch download commands
    if command == 'init':
        stock_list = get_stock_list()
        if stock_list is not None:
            print(f"Stock list loaded: {len(stock_list)} stocks")
            print("Ready to download. Run: run('download')")
        return

    elif command == 'download':
        pro = init_tushare()
        stock_list = get_stock_list()
        if stock_list is None:
            return
        batch_size = arg if isinstance(arg, int) else None
        download_all_news(pro, stock_list, batch_size=batch_size)
        return

    elif command == 'status':
        show_status()
        return

    elif command == 'retry':
        pro = init_tushare()
        retry_failed(pro)
        return

    elif command == 'reset':
        reset_progress()
        return

    # Quick query commands
    pro = init_tushare()
    save = kwargs.pop('save', False)
    df = None
    filename = None

    if command == 'major':
        date = arg
        df = get_major_news(pro, start_date=date, end_date=date, **kwargs)
        filename = f"major_news_{date or datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'cctv':
        date = arg
        df = get_cctv_news(pro, date=date)
        filename = f"cctv_news_{date or datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'search':
        if not arg:
            print("Please provide a search keyword. Example: run('search', '茅台')")
            return None
        df = search_news(pro, keyword=arg, **kwargs)
        filename = f"search_{arg}_{datetime.now().strftime('%Y%m%d')}.csv"

    else:
        print("Batch download commands:")
        print("  run('init')           # Verify stock list")
        print("  run('download')       # Download all stocks")
        print("  run('download', 100)  # Download 100 stocks per batch")
        print("  run('status')         # Check progress")
        print("  run('retry')          # Retry failed")
        print("  run('reset')          # Reset progress")
        print("\nQuick query commands:")
        print("  run('major')          # Today's major news")
        print("  run('cctv')           # CCTV news")
        print("  run('search', '茅台')  # Search by keyword")
        return None

    display_news(df)

    if save and filename:
        filepath = NEWS_DIR / filename
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Saved to {filepath}")

    return df


if __name__ == '__main__':
    print("Stock News Module - 5 Year Historical Download")
    print("=" * 50)
    print("Usage:")
    print("  from features.news import run")
    print("  run('init')       # Verify stock list")
    print("  run('download')   # Start downloading")
    print("  run('status')     # Check progress")
