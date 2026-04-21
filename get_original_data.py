"""
Historical Stock Data Acquisition — Date-first approach.

Tushare Pro efficiency tip: pro.daily(trade_date=YYYYMMDD) returns ALL
~5000 stocks for one trading day in a single API call, vs calling
pro.daily(ts_code=X, ...) once per stock.

Result: 2254 API calls (one per trading date) instead of 5190 (one per stock).
Each call returns ~5000 rows; total data is identical.

Data is written to per-stock CSVs (sh/{symbol}.csv, sz/{symbol}.csv) for
full backward compatibility with dl/, deeptime/, and extend_stock_data.py.

Usage:
    python get_original_data.py --download          # full download 2017→today
    python get_original_data.py --download --batch 60   # 60 dates per run
    python get_original_data.py --status
    python get_original_data.py --retry-failed
    python get_original_data.py --reset

    # PyCharm/Jupyter:
    from get_original_data import run
    run('download')
    run('download', batch=60)
    run('status')
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts

warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN  = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR       = Path('./stock_data')
CHECKPOINT_FILE = DATA_DIR / 'checkpoint.json'
STOCK_LIST_FILE = DATA_DIR / 'stock_list.csv'
TARGET_START   = '20170101'

# Flush accumulated data to disk every FLUSH_EVERY dates to bound RAM usage.
# 30 dates × 5000 stocks × ~200 bytes ≈ 30 MB peak — well within 16 GB.
FLUSH_EVERY    = 30
CALL_INTERVAL  = 0.13   # ~7.5 calls/s ≈ 450/min, safe for 7000-pt accounts


# ─── Helpers ──────────────────────────────────────────────────────────────────

def init_tushare():
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api(TUSHARE_TOKEN)


def setup_directories():
    for d in ('sh', 'sz'):
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                ck = json.load(f)
            # Migrate old stock-based checkpoint if present
            if 'completed_dates' not in ck:
                ck = {'completed_dates': [], 'failed_dates': []}
            return ck
        except Exception:
            pass
    return {'completed_dates': [], 'failed_dates': []}


def save_checkpoint(ck: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    ck_clean = {
        'completed_dates': sorted(set(ck['completed_dates'])),
        'failed_dates':    sorted(set(ck['failed_dates'])),
    }
    with open(tmp, 'w') as f:
        json.dump(ck_clean, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)
    ck.update(ck_clean)


# ─── Stock list ───────────────────────────────────────────────────────────────

def get_stock_list(pro=None, refresh=False, max_retries=5) -> pd.DataFrame:
    """Fetch SH/SZ stock list from Tushare Pro stock_basic (no akshare needed)."""
    if STOCK_LIST_FILE.exists() and not refresh:
        return pd.read_csv(STOCK_LIST_FILE)

    if pro is None:
        pro = init_tushare()

    print("Fetching stock list from Tushare Pro (stock_basic)...")
    fields = 'ts_code,symbol,name,area,market,list_date,is_hs'
    for attempt in range(max_retries):
        try:
            df_L = pro.stock_basic(exchange='', list_status='L', fields=fields)
            df_P = pro.stock_basic(exchange='', list_status='P', fields=fields)
            stock_list = pd.concat([df_L, df_P], ignore_index=True)
            break
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError("Failed to fetch stock list after all retries")

    stock_list['exchange'] = stock_list['ts_code'].str.split('.').str[1]
    stock_list = stock_list[stock_list['exchange'].isin(['SH', 'SZ'])].copy()
    stock_list = stock_list.sort_values('symbol').reset_index(drop=True)
    stock_list.to_csv(STOCK_LIST_FILE, index=False)

    sh = (stock_list['exchange'] == 'SH').sum()
    sz = (stock_list['exchange'] == 'SZ').sum()
    print(f"  Stock list saved: {len(stock_list)} stocks (SH: {sh}, SZ: {sz})")
    return stock_list


# ─── Trading calendar ─────────────────────────────────────────────────────────

def get_trading_dates(pro, start_date: str, end_date: str) -> list:
    """Return sorted list of trading dates (YYYYMMDD strings) in [start, end]."""
    df = pro.trade_cal(exchange='SSE', start_date=start_date,
                       end_date=end_date, is_open='1')
    return sorted(df['cal_date'].astype(str).tolist())


# ─── Per-date download ────────────────────────────────────────────────────────

def download_date(pro, trade_date: str, max_retries: int = 3) -> pd.DataFrame:
    """Download all stocks' daily OHLCV for one trading date."""
    for attempt in range(max_retries):
        try:
            df = pro.daily(trade_date=trade_date)
            if df is not None and not df.empty:
                return df
            return pd.DataFrame()
        except Exception as e:
            msg = str(e).lower()
            if 'exceed' in msg or 'limit' in msg or '频率' in msg:
                wait = 60 * (attempt + 1)
                print(f"\n  [rate limit] waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                return None   # signal failure
    return None


# ─── Buffer flush ─────────────────────────────────────────────────────────────

def flush_buffer(buffer: dict):
    """
    Write accumulated {ts_code: DataFrame} buffer to per-stock CSV files.
    Files are stored newest-first (descending trade_date) so that
    extend_stock_data.get_latest_date() can read just the first row.
    """
    for ts_code, new_df in buffer.items():
        exchange = ts_code.split('.')[1].lower()   # 'sh' or 'sz'
        if exchange not in ('sh', 'sz'):
            continue
        symbol = ts_code.split('.')[0]
        path   = DATA_DIR / exchange / f'{symbol}.csv'

        new_df = new_df.sort_values('trade_date', ascending=False)

        if path.exists():
            try:
                existing = pd.read_csv(path, dtype={'trade_date': str})
                combined = (pd.concat([existing, new_df])
                            .drop_duplicates('trade_date')
                            .sort_values('trade_date', ascending=False)
                            .reset_index(drop=True))
                combined.to_csv(path, index=False)
            except Exception:
                new_df.to_csv(path, index=False)
        else:
            new_df.to_csv(path, index=False)


# ─── Main download loop ───────────────────────────────────────────────────────

def download_all(pro, start_date: str = TARGET_START, end_date: str = None,
                 batch: int = None):
    """
    Download all A-share daily OHLCV by iterating trading dates.

    Args:
        pro:        Tushare Pro API instance.
        start_date: First date to download (YYYYMMDD).
        end_date:   Last date (default: today).
        batch:      Stop after downloading this many dates (for incremental runs).
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    ck = load_checkpoint()
    completed_set = set(ck['completed_dates'])
    failed_set    = set(ck['failed_dates'])

    all_dates = get_trading_dates(pro, start_date, end_date)
    pending   = [d for d in all_dates if d not in completed_set]

    total     = len(all_dates)
    done      = len(completed_set)
    remaining = len(pending)

    print(f"\n{'='*60}")
    print(f"Date-first download: {start_date} → {end_date}")
    print(f"  Total trading days: {total}")
    print(f"  Already downloaded: {done}")
    print(f"  Remaining:          {remaining}")
    if batch:
        print(f"  This run limit:     {batch} dates")
    print(f"{'='*60}\n")

    if not pending:
        print("  All dates already downloaded.")
        return

    buffer: dict = {}    # {ts_code: list of DataFrames}
    dates_this_run = 0

    for i, trade_date in enumerate(pending):
        if batch and dates_this_run >= batch:
            print(f"\nBatch limit ({batch} dates) reached. Run again to continue.")
            break

        df = download_date(pro, trade_date)

        if df is None:
            ck['failed_dates'].append(trade_date)
            print(f"  [{done+dates_this_run+1:4d}/{total}] {trade_date}  FAILED")
        elif df.empty:
            ck['completed_dates'].append(trade_date)
        else:
            # Accumulate by ts_code
            for ts_code, grp in df.groupby('ts_code'):
                if ts_code not in buffer:
                    buffer[ts_code] = []
                buffer[ts_code].append(grp)

            ck['completed_dates'].append(trade_date)
            n_stocks = len(df['ts_code'].unique())
            print(f"  [{done+dates_this_run+1:4d}/{total}] {trade_date}  "
                  f"{n_stocks:4d} stocks")

        dates_this_run += 1
        time.sleep(CALL_INTERVAL)

        # Flush buffer periodically to limit RAM usage
        if dates_this_run % FLUSH_EVERY == 0:
            flush_buffer({k: pd.concat(v) for k, v in buffer.items()})
            buffer.clear()
            save_checkpoint(ck)
            print(f"  ... flushed to disk ({done+dates_this_run} total dates done)")

    # Final flush
    if buffer:
        flush_buffer({k: pd.concat(v) for k, v in buffer.items()})
        buffer.clear()

    save_checkpoint(ck)
    print(f"\n{'='*60}")
    print(f"Session complete: {dates_this_run} dates downloaded this run")
    print(f"Total completed: {len(ck['completed_dates'])}/{total}")
    if ck['failed_dates']:
        print(f"Failed dates:    {len(ck['failed_dates'])} (run --retry-failed)")
    print(f"{'='*60}")


def retry_failed(pro):
    """Re-download dates that previously failed."""
    ck = load_checkpoint()
    failed = list(ck['failed_dates'])
    if not failed:
        print("No failed dates to retry.")
        return

    print(f"Retrying {len(failed)} failed dates...")
    ck['failed_dates'] = []
    buffer = {}

    for trade_date in sorted(failed):
        df = download_date(pro, trade_date)
        if df is not None and not df.empty:
            for ts_code, grp in df.groupby('ts_code'):
                if ts_code not in buffer:
                    buffer[ts_code] = []
                buffer[ts_code].append(grp)
            ck['completed_dates'].append(trade_date)
            print(f"  {trade_date}  OK ({len(df)} rows)")
        else:
            ck['failed_dates'].append(trade_date)
            print(f"  {trade_date}  FAILED again")
        time.sleep(CALL_INTERVAL)

    if buffer:
        flush_buffer({k: pd.concat(v) for k, v in buffer.items()})
    save_checkpoint(ck)
    print(f"Done. {len(ck['completed_dates'])} total completed.")


def show_status():
    """Show download status."""
    ck = load_checkpoint()
    pro = init_tushare()
    end_date = datetime.now().strftime('%Y%m%d')
    try:
        all_dates = get_trading_dates(pro, TARGET_START, end_date)
        total = len(all_dates)
    except Exception:
        total = '?'

    sh_files = len(list((DATA_DIR / 'sh').glob('*.csv'))) if (DATA_DIR / 'sh').exists() else 0
    sz_files = len(list((DATA_DIR / 'sz').glob('*.csv'))) if (DATA_DIR / 'sz').exists() else 0

    print(f"\n{'='*60}")
    print(f"Download Status (date-first)")
    print(f"  Trading days downloaded: {len(ck['completed_dates'])}/{total}")
    print(f"  Failed dates:            {len(ck['failed_dates'])}")
    print(f"  Stock CSV files:         SH={sh_files}, SZ={sz_files}")
    if ck['completed_dates']:
        print(f"  Date range covered:      {min(ck['completed_dates'])} → {max(ck['completed_dates'])}")
    print(f"{'='*60}")


def reset_progress():
    """Delete checkpoint (keep all downloaded CSV files)."""
    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)
    print("Checkpoint reset. Downloaded CSV files are preserved.")


# ─── Public API ───────────────────────────────────────────────────────────────

def run(command: str = 'status', batch: int = None):
    """
    Scripted entry point.

    Commands:
        'init'     — fetch and cache stock list only
        'download' — download all trading dates (resumable)
        'status'   — show progress
        'retry'    — retry failed dates
        'reset'    — clear checkpoint (keep CSVs)

    Args:
        batch: Maximum number of dates to download per run (default: all).

    Examples:
        run('download')           # download everything
        run('download', batch=60) # 60 dates per run (~3 min)
        run('status')
    """
    setup_directories()
    pro = init_tushare()

    if command == 'init':
        get_stock_list(pro=pro)
        print("Done. Run run('download') to start.")
    elif command == 'download':
        get_stock_list(pro=pro)   # ensure stock_list.csv exists
        download_all(pro, batch=batch)
    elif command == 'status':
        show_status()
    elif command == 'retry':
        retry_failed(pro)
    elif command == 'reset':
        reset_progress()
    else:
        print("Commands: 'init', 'download', 'status', 'retry', 'reset'")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download SH/SZ daily OHLCV — date-first (efficient)')
    parser.add_argument('--download',      action='store_true', help='Download all dates')
    parser.add_argument('--batch',         type=int, default=None,
                        help='Max dates per run (default: all)')
    parser.add_argument('--status',        action='store_true', help='Show progress')
    parser.add_argument('--retry-failed',  action='store_true', help='Retry failed dates')
    parser.add_argument('--reset',         action='store_true', help='Reset checkpoint')
    parser.add_argument('--init',          action='store_true', help='Fetch stock list only')
    args, _ = parser.parse_known_args()

    setup_directories()
    pro = init_tushare()

    if args.status:
        show_status()
    elif args.reset:
        reset_progress()
    elif args.init:
        get_stock_list(pro=pro)
    elif args.retry_failed:
        retry_failed(pro)
    elif args.download:
        get_stock_list(pro=pro)
        download_all(pro, batch=args.batch)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python get_original_data.py --download")
        print("  python get_original_data.py --download --batch 60  # incremental")


if __name__ == '__main__':
    main()
