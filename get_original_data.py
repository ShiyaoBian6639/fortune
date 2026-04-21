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

# Network settings — increase for slow/unstable connections (remote servers)
API_TIMEOUT    = 120    # seconds; default tushare SDK uses 30 which is too short
MAX_RETRIES    = 5
RETRY_BASE_SEC = 10     # exponential back-off: 10, 20, 40, 80, 160 seconds


# ─── Helpers ──────────────────────────────────────────────────────────────────

def init_tushare():
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)
    # Override the SDK's 30-second timeout — remote servers often need more
    try:
        pro._DataApi__timeout = API_TIMEOUT
    except AttributeError:
        pass   # SDK version doesn't expose it; best-effort
    return pro


def _retry(fn, *args, label='call', max_retries=MAX_RETRIES, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential back-off on any exception.
    Handles: ReadTimeout, connection errors, Tushare rate limits.
    """
    for attempt in range(max_retries):
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            msg = str(e).lower()
            is_rate  = 'exceed' in msg or 'limit' in msg or '频率' in msg
            is_net   = 'timeout' in msg or 'connection' in msg or 'timed out' in msg
            wait = (60 * (attempt + 1)) if is_rate else (RETRY_BASE_SEC * (2 ** attempt))
            if attempt < max_retries - 1:
                print(f"  [{label}] attempt {attempt+1}/{max_retries} failed: "
                      f"{type(e).__name__}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"{label} failed after {max_retries} attempts: {e}") from e
    return None


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

def get_stock_list(pro=None, refresh=False) -> pd.DataFrame:
    """Fetch SH/SZ stock list from Tushare Pro stock_basic (no akshare needed)."""
    if STOCK_LIST_FILE.exists() and not refresh:
        return pd.read_csv(STOCK_LIST_FILE)

    if pro is None:
        pro = init_tushare()

    print("Fetching stock list from Tushare Pro (stock_basic)...")
    fields = 'ts_code,symbol,name,area,market,list_date,is_hs'
    df_L = _retry(pro.stock_basic, exchange='', list_status='L', fields=fields, label='stock_basic(L)')
    df_P = _retry(pro.stock_basic, exchange='', list_status='P', fields=fields, label='stock_basic(P)')
    stock_list = pd.concat([df_L, df_P], ignore_index=True)

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
    df = _retry(
        pro.trade_cal,
        exchange='SSE', start_date=start_date, end_date=end_date, is_open='1',
        label='trade_cal',
    )
    return sorted(df['cal_date'].astype(str).tolist())


# ─── Per-date download ────────────────────────────────────────────────────────

def download_date(pro, trade_date: str) -> pd.DataFrame:
    """Download all stocks' daily OHLCV for one trading date (with retries)."""
    try:
        df = _retry(pro.daily, trade_date=trade_date, label=f'daily({trade_date})')
        if df is not None and not df.empty:
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"  [{trade_date}] FAILED after all retries: {e}")
        return None   # signal permanent failure


# ─── Buffer flush ─────────────────────────────────────────────────────────────

def flush_buffer(buffer: dict, initial_download: bool = False):
    """
    Write accumulated {ts_code: DataFrame} buffer to per-stock CSV files.

    initial_download=True: files don't exist yet → just write, skip read-merge.
    This is ~3× faster for the initial full download.
    Files stored newest-first (descending trade_date) for extend_stock_data compat.
    """
    for ts_code, new_df in buffer.items():
        parts = ts_code.split('.')
        if len(parts) != 2:
            continue
        exchange = parts[1].lower()
        if exchange not in ('sh', 'sz'):
            continue
        symbol = parts[0]
        path   = DATA_DIR / exchange / f'{symbol}.csv'

        new_df = new_df.sort_values('trade_date', ascending=False)

        if initial_download or not path.exists():
            new_df.to_csv(path, index=False)
        else:
            try:
                existing = pd.read_csv(path, dtype={'trade_date': str})
                combined = (pd.concat([existing, new_df])
                            .drop_duplicates('trade_date')
                            .sort_values('trade_date', ascending=False)
                            .reset_index(drop=True))
                combined.to_csv(path, index=False)
            except Exception:
                new_df.to_csv(path, index=False)


# ─── Main download loop ───────────────────────────────────────────────────────

def download_all(pro, start_date: str = TARGET_START, end_date: str = None,
                 batch: int = None, initial: bool = False):
    """
    Download all A-share daily OHLCV by iterating trading dates.

    Args:
        pro:        Tushare Pro API instance.
        start_date: First date to download (YYYYMMDD).
        end_date:   Last date (default: today).
        batch:      Stop after this many dates (for safe incremental runs).
        initial:    True = no existing files, skip read-merge on flush (~3x faster).
                    Auto-detected if stock_data/sh/ is empty.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    # Auto-detect initial download (no stock files exist yet)
    sh_files = list((DATA_DIR / 'sh').glob('*.csv')) if (DATA_DIR / 'sh').exists() else []
    if initial or not sh_files:
        initial = True
        print("  [Mode] Initial download — flush writes new files only (fastest)")

    ck = load_checkpoint()
    completed_set = set(ck['completed_dates'])

    all_dates = get_trading_dates(pro, start_date, end_date)
    pending   = [d for d in all_dates if d not in completed_set]

    total     = len(all_dates)
    done      = len(completed_set)
    remaining = len(pending)

    print(f"\n{'='*60}")
    print(f"Date-first download: {start_date} → {end_date}")
    print(f"  Total trading days : {total}")
    print(f"  Already downloaded : {done}")
    print(f"  Remaining          : {remaining}")
    if batch:
        print(f"  This run limit     : {batch} dates")
    # Time estimate at 0.5s/call avg
    eta_min = remaining * 0.5 / 60
    print(f"  ETA (network only) : ~{eta_min:.0f} min")
    print(f"{'='*60}\n")

    if not pending:
        print("  All dates already downloaded.")
        return

    buffer: dict = {}
    dates_this_run = 0
    t_start = time.time()

    for i, trade_date in enumerate(pending):
        if batch and dates_this_run >= batch:
            print(f"\nBatch limit ({batch} dates) reached. Run again to continue.")
            break

        t0 = time.time()
        df = download_date(pro, trade_date)
        elapsed = time.time() - t0

        if df is None:
            ck['failed_dates'].append(trade_date)
            print(f"  [{done+dates_this_run+1:4d}/{total}] {trade_date}  FAILED")
        elif df.empty:
            ck['completed_dates'].append(trade_date)
        else:
            for ts_code, grp in df.groupby('ts_code'):
                if ts_code not in buffer:
                    buffer[ts_code] = []
                buffer[ts_code].append(grp)

            ck['completed_dates'].append(trade_date)
            n_stocks = df['ts_code'].nunique()

            # ETA update every 50 dates
            dates_done_total = done + dates_this_run + 1
            if (dates_this_run + 1) % 50 == 0:
                rate = (dates_this_run + 1) / (time.time() - t_start)
                left = total - dates_done_total
                eta  = left / rate / 60
                print(f"  [{dates_done_total:4d}/{total}] {trade_date}  "
                      f"{n_stocks:4d} stocks  ({rate:.1f} dates/s, ETA ~{eta:.0f} min)")
            else:
                print(f"  [{dates_done_total:4d}/{total}] {trade_date}  {n_stocks:4d} stocks")

        dates_this_run += 1
        time.sleep(max(0, CALL_INTERVAL - elapsed))

        # Flush buffer periodically
        if dates_this_run % FLUSH_EVERY == 0:
            t_flush = time.time()
            flush_buffer({k: pd.concat(v) for k, v in buffer.items()}, initial)
            buffer.clear()
            save_checkpoint(ck)
            print(f"  ... checkpoint saved  "
                  f"({done+dates_this_run}/{total} dates, flush={time.time()-t_flush:.1f}s)")

    # Final flush
    if buffer:
        flush_buffer({k: pd.concat(v) for k, v in buffer.items()}, initial)
        buffer.clear()

    save_checkpoint(ck)
    total_time = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"Session complete in {total_time:.1f} min")
    print(f"  Dates this run : {dates_this_run}")
    print(f"  Total done     : {len(ck['completed_dates'])}/{total}")
    if ck['failed_dates']:
        print(f"  Failed dates   : {len(ck['failed_dates'])} → run --retry-failed")
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
    parser.add_argument('--initial',       action='store_true',
                        help='Force initial-download mode (skip read-merge, ~3x faster for first run)')
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
        download_all(pro, batch=args.batch, initial=args.initial)
    else:
        parser.print_help()
        print("\nQuick start (fresh server):")
        print("  python get_original_data.py --download          # auto-detects initial mode")
        print("  python get_original_data.py --status            # check progress")
        print("  python get_original_data.py --retry-failed      # retry any failed dates")


if __name__ == '__main__':
    main()
