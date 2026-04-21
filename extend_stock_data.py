"""
Extend Stock Data (Both Directions)
- Downloads historical data from 2017-01-01 to the start of existing data
- Downloads new data from the end of existing data to the most recent trading date
- Combines with existing stock data
- Preserves checkpoint for resume capability
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import tushare as ts
import pandas as pd
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR = Path('./stock_data')
EXTEND_CHECKPOINT = DATA_DIR / 'extend_checkpoint.json'

# Target date range
TARGET_START_DATE = '20170101'


# ─── Rate limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Enforces a global call rate across all worker threads so the combined
    throughput never exceeds the Tushare API limit.

    7000-point accounts: ~500 calls/min → use calls_per_sec=8.0 (480/min,
    leaves a ~4 % safety margin).
    """
    def __init__(self, calls_per_sec: float = 8.0):
        self._interval = 1.0 / calls_per_sec
        self._last     = 0.0
        self._lock     = threading.Lock()

    def wait(self):
        with self._lock:
            gap = self._interval - (time.perf_counter() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.perf_counter()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_today_date():
    """Get today's date in YYYYMMDD format."""
    return datetime.now().strftime('%Y%m%d')


def init_tushare():
    """Initialize Tushare API connection."""
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api(TUSHARE_TOKEN)


def load_extend_checkpoint():
    """Load extension progress checkpoint."""
    if EXTEND_CHECKPOINT.exists():
        try:
            with open(EXTEND_CHECKPOINT, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {'completed': [], 'failed': [], 'skipped': []}


def save_extend_checkpoint(checkpoint):
    """Save extension progress checkpoint (deduplicates lists on save)."""
    deduped = {
        'completed': list(set(checkpoint['completed'])),
        'failed':    list(set(checkpoint['failed'])),
        'skipped':   list(set(checkpoint['skipped'])),
    }
    temp_file = EXTEND_CHECKPOINT.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(deduped, f, indent=2)
    temp_file.replace(EXTEND_CHECKPOINT)
    checkpoint['completed'] = deduped['completed']
    checkpoint['failed']    = deduped['failed']
    checkpoint['skipped']   = deduped['skipped']


def get_existing_stocks():
    """Get list of all existing stock files."""
    stocks = []
    for exchange in ('SH', 'SZ'):
        market_dir = DATA_DIR / exchange.lower()
        if market_dir.exists():
            for f in market_dir.glob('*.csv'):
                stocks.append({
                    'symbol':   f.stem,
                    'exchange': exchange,
                    'ts_code':  f'{f.stem}.{exchange}',
                    'path':     f,
                })
    return stocks


def get_latest_date(filepath):
    """
    Return the most recent trade_date in the CSV.

    Files are saved newest-first (descending sort), so only the first
    data row needs to be read — 2000x faster than scanning the whole file.
    """
    try:
        row = pd.read_csv(filepath, nrows=1)
        if row.empty or 'trade_date' not in row.columns:
            return None
        return str(int(row['trade_date'].iloc[0]))
    except Exception:
        return None


def get_earliest_date(filepath):
    """Return the oldest trade_date in the CSV (reads whole file)."""
    try:
        df = pd.read_csv(filepath, usecols=['trade_date'])
        if df.empty:
            return None
        return str(int(df['trade_date'].min()))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def download_stock_data(pro, ts_code, start_date, end_date, max_retries=3):
    """Download daily data for a single stock."""
    for attempt in range(max_retries):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if 'exceed' in low or 'limit' in low or '频率' in msg:
                wait_time = 60 * (attempt + 1)
                print(f"\n  Rate limit hit on {ts_code}, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif 'permission' in low or '权限' in msg:
                return pd.DataFrame()
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)
    return None


# ─── Parallel forward-update worker ──────────────────────────────────────────

def _forward_update_worker(args):
    """
    Thread worker: pull new daily bars for one stock.

    Fast path  — stock is current: reads only 1 CSV row, zero API calls.
    Update path — stock is stale:  reads 1 row, 1 API call, 1 full CSV read+write.

    Thread-safe because every stock owns its own CSV file.

    Returns (ts_code, result, new_latest_date)
        result ∈ {'updated', 'current', 'failed'}
    """
    pro, ts_code, filepath, today_int, rate_limiter = args

    # Fast check — read only 1 row (file sorted newest-first)
    latest_date = get_latest_date(filepath)
    if latest_date is None:
        return ts_code, 'failed', ''

    latest_int = int(latest_date)

    # Up-to-date: latest is >= yesterday (proper date arithmetic avoids the
    # mid-week false-positive where today_int-3 would wrongly skip Mon data
    # on a Wednesday).  On Mondays the API call for Sat/Sun returns empty
    # and the stock is correctly kept as current with one extra (cheap) call.
    yesterday_int = int((datetime.now() - timedelta(days=1)).strftime('%Y%m%d'))
    if latest_int >= yesterday_int:
        return ts_code, 'current', latest_date

    # Throttle before each real API call
    rate_limiter.wait()

    start_date = str(latest_int + 1)
    new_df = download_stock_data(pro, ts_code, start_date, str(today_int))

    if new_df is None:
        return ts_code, 'failed', latest_date
    if new_df.empty:
        return ts_code, 'current', latest_date

    # Read full CSV, append new rows, dedup, re-sort, save
    existing = pd.read_csv(filepath)
    existing['trade_date'] = existing['trade_date'].astype(int)
    new_df['trade_date']   = new_df['trade_date'].astype(int)

    combined = (
        pd.concat([existing, new_df], ignore_index=True)
          .drop_duplicates(subset=['trade_date'], keep='first')
          .sort_values('trade_date', ascending=False)
          .reset_index(drop=True)
    )
    combined.to_csv(filepath, index=False)

    return ts_code, 'updated', str(combined['trade_date'].iloc[0])


# ─── Backward extension (sequential, run once) ────────────────────────────────

def extend_stock_data(pro, stock_info, extend_backward=True, extend_forward=True):
    """
    Extend one stock backward to TARGET_START_DATE and/or forward to today.
    Used for the one-time backward fill; daily updates use _forward_update_worker.

    Returns: 'updated' | 'completed' | 'skipped' | 'failed'
    """
    ts_code  = stock_info['ts_code']
    filepath = stock_info['path']

    earliest_date = get_earliest_date(filepath)
    latest_date   = get_latest_date(filepath)

    if earliest_date is None or latest_date is None:
        return 'failed'

    today      = get_today_date()
    updated    = False
    data_frames = [pd.read_csv(filepath)]

    if extend_backward and int(earliest_date) > int(TARGET_START_DATE):
        end_date = str(int(earliest_date) - 1)
        hist_df  = download_stock_data(pro, ts_code, TARGET_START_DATE, end_date)
        if hist_df is None:
            return 'failed'
        if not hist_df.empty:
            data_frames.append(hist_df)
            updated = True

    if extend_forward and int(latest_date) < int(today):
        start_date = str(int(latest_date) + 1)
        new_df     = download_stock_data(pro, ts_code, start_date, today)
        if new_df is None:
            return 'failed'
        if not new_df.empty:
            data_frames.append(new_df)
            updated = True

    if not updated:
        has_old = int(earliest_date) <= int(TARGET_START_DATE)
        has_new = int(latest_date)   >= int(today) - 3
        return 'skipped' if (has_old and has_new) else 'completed'

    combined = (
        pd.concat(data_frames, ignore_index=True)
          .assign(trade_date=lambda d: d['trade_date'].astype(int))
          .drop_duplicates(subset=['trade_date'], keep='first')
          .sort_values('trade_date', ascending=False)
          .reset_index(drop=True)
    )
    combined.to_csv(filepath, index=False)
    return 'updated'


# ─── Main driver ─────────────────────────────────────────────────────────────

def extend_all_stocks(
    batch_size=None,
    backward=True,
    forward=True,
    force_update=False,
    max_workers=8,
    calls_per_sec=8.0,
):
    """
    Extend all existing stock files to 2017-01-01 and/or up to today.

    Args:
        batch_size:    Optional limit on stocks processed this run.
        backward:      Download historical data back to 2017.
        forward:       Download new data up to today.
        force_update:  Ignore checkpoint — process every stock.
        max_workers:   Parallel threads (forward-only mode only).
                       8 is safe for 7000-point Tushare accounts.
        calls_per_sec: API call rate limit (default 8.0 = 480 calls/min).
                       Raise to 10–12 if you have a higher-tier account and
                       want to go faster; lower if you see rate-limit errors.
    """
    pro    = init_tushare()
    stocks = get_existing_stocks()
    checkpoint = load_extend_checkpoint()

    completed_set = set(checkpoint['completed'])
    skipped_set   = set(checkpoint['skipped'])
    today         = get_today_date()
    today_int     = int(today)

    # Build work list
    if backward:
        # Backward (rare, one-time): respect checkpoint to allow resume
        if not force_update:
            stocks_to_process = [
                s for s in stocks
                if s['ts_code'] not in completed_set and s['ts_code'] not in skipped_set
            ]
        else:
            stocks_to_process = stocks
    else:
        # Forward-only (daily): always re-evaluate; checkpoint not consulted
        stocks_to_process = stocks

    if batch_size:
        stocks_to_process = stocks_to_process[:batch_size]

    total = len(stocks_to_process)

    print(f"\n{'='*60}")
    print("Extending Stock Data")
    print(f"{'='*60}")
    print(f"Date range: {TARGET_START_DATE} → {today}")
    print(f"Backward (to 2017): {'Yes' if backward else 'No'}")
    print(f"Forward  (to today): {'Yes' if forward else 'No'}")
    print(f"Stocks to check: {total:,}  (total on disk: {len(stocks):,})")
    if not backward:
        print(f"Parallel workers: {max_workers}  |  rate cap: {calls_per_sec:.1f} calls/s")
    print(f"{'='*60}\n")

    processed  = 0
    new_updated = 0
    new_failed  = 0
    checkpoint_lock = threading.Lock()

    def _save_checkpoint_periodic():
        nonlocal processed
        if processed % 50 == 0:
            with checkpoint_lock:
                save_extend_checkpoint(checkpoint)

    try:
        if not backward and forward:
            # ── Parallel forward-only update ──────────────────────────
            rate_limiter = RateLimiter(calls_per_sec=calls_per_sec)

            worker_args = [
                (pro, s['ts_code'], s['path'], today_int, rate_limiter)
                for s in stocks_to_process
            ]

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_code = {
                    pool.submit(_forward_update_worker, a): a[1]
                    for a in worker_args
                }

                for future in as_completed(future_to_code):
                    ts_code_done, result, latest = future.result()

                    with checkpoint_lock:
                        processed += 1
                        if result == 'updated':
                            checkpoint['completed'].append(ts_code_done)
                            new_updated += 1
                            print(f"  [{processed:5d}/{total}] {ts_code_done} updated → {latest}")
                        elif result == 'failed':
                            checkpoint['failed'].append(ts_code_done)
                            new_failed += 1
                            print(f"  [{processed:5d}/{total}] {ts_code_done} FAILED")
                        # 'current' stocks: silent (too noisy for 5000+ up-to-date stocks)
                        elif processed % 500 == 0:
                            print(f"  [{processed:5d}/{total}] ... ({new_updated} updated so far)")

                        if processed % 50 == 0:
                            save_extend_checkpoint(checkpoint)

        else:
            # ── Sequential backward + forward ────────────────────────
            # Backward fetches span years of data per stock; one API call
            # per stock is already slow — parallelism adds little here.
            for stock in stocks_to_process:
                ts_code = stock['ts_code']

                if batch_size and processed >= batch_size:
                    print(f"\nBatch limit ({batch_size}) reached.")
                    break

                print(f"[{processed+1}/{total}] {ts_code} ...", end=' ', flush=True)

                result = extend_stock_data(pro, stock,
                                           extend_backward=backward,
                                           extend_forward=forward)
                processed += 1

                if result == 'updated':
                    checkpoint['completed'].append(ts_code)
                    new_updated += 1
                    latest   = get_latest_date(stock['path'])
                    earliest = get_earliest_date(stock['path'])
                    print(f"Updated  ({earliest} → {latest})")
                elif result == 'completed':
                    checkpoint['completed'].append(ts_code)
                    latest   = get_latest_date(stock['path'])
                    earliest = get_earliest_date(stock['path'])
                    print(f"OK  ({earliest} → {latest})")
                elif result == 'skipped':
                    checkpoint['skipped'].append(ts_code)
                    print("Skipped (already up-to-date)")
                else:
                    checkpoint['failed'].append(ts_code)
                    new_failed += 1
                    print("FAILED")

                if processed % 50 == 0:
                    save_extend_checkpoint(checkpoint)

                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")

    finally:
        save_extend_checkpoint(checkpoint)
        n_current = processed - new_updated - new_failed
        print(f"\n{'='*60}")
        print(f"Session Complete!")
        print(f"  Checked:        {processed:,}")
        print(f"  Updated:        {new_updated:,}")
        print(f"  Already current:{n_current:,}")
        print(f"  Failed:         {new_failed:,}")
        print(f"{'='*60}")


# ─── Status / reset helpers ───────────────────────────────────────────────────

def show_status():
    """Show extension status with sample date ranges."""
    stocks     = get_existing_stocks()
    checkpoint = load_extend_checkpoint()
    today      = get_today_date()

    total     = len(stocks)
    completed = len(set(checkpoint['completed']))
    skipped   = len(set(checkpoint['skipped']))
    failed    = len(set(checkpoint['failed']))

    print(f"\n{'='*60}")
    print("Extension Status")
    print(f"{'='*60}")
    print(f"Target date range: {TARGET_START_DATE} → {today}")
    print(f"Total stocks:  {total}")
    print(f"Completed:     {completed}")
    print(f"Skipped:       {skipped}")
    print(f"Failed:        {failed}")
    print(f"{'='*60}")

    if stocks:
        print("\nSample date ranges (first 5 stocks):")
        for stock in stocks[:5]:
            earliest = get_earliest_date(stock['path'])
            latest   = get_latest_date(stock['path'])
            print(f"  {stock['ts_code']}: {earliest} → {latest}")
        print(f"{'='*60}")


def reset_progress():
    """Reset extension progress."""
    if EXTEND_CHECKPOINT.exists():
        os.remove(EXTEND_CHECKPOINT)
    print("Extension progress reset.")


# ─── Public API ───────────────────────────────────────────────────────────────

def run(command='status', batch=None, backward=True, forward=True, force=False,
        workers=8, rate=8.0):
    """
    Entry point for scripted or interactive use.

    Args:
        command:  'extend' | 'update' | 'status' | 'reset'
        batch:    Optional batch size.
        backward: Download historical data back to 2017 (default True).
        forward:  Download new data up to today (default True).
        force:    Ignore checkpoint and reprocess every stock.
        workers:  Parallel threads for forward-only updates (default 8).
        rate:     Max Tushare API calls/second (default 8.0 = 480/min).

    Examples:
        run('status')
        run('update')                   # fast parallel forward update
        run('update', workers=12, rate=10.0)   # push harder (higher-tier account)
        run('extend')                   # one-time full backward + forward fill
        run('extend', batch=500)        # batch mode for backward fill
    """
    if command == 'status':
        show_status()
    elif command == 'reset':
        reset_progress()
    elif command == 'extend':
        extend_all_stocks(batch_size=batch, backward=backward, forward=forward,
                          force_update=force, max_workers=workers, calls_per_sec=rate)
    elif command == 'update':
        extend_all_stocks(batch_size=batch, backward=False, forward=True,
                          force_update=force, max_workers=workers, calls_per_sec=rate)
    else:
        print("Commands: 'extend', 'update', 'status', 'reset'")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extend stock data (2017 to today)')
    parser.add_argument('--extend',      action='store_true', help='Extend backward + forward')
    parser.add_argument('--update',      action='store_true', help='Update to latest (forward only, parallel)')
    parser.add_argument('--status',      action='store_true', help='Show status')
    parser.add_argument('--reset',       action='store_true', help='Reset checkpoint')
    parser.add_argument('--batch',       type=int,   default=None, help='Batch size')
    parser.add_argument('--force',       action='store_true',       help='Ignore checkpoint')
    parser.add_argument('--workers',     type=int,   default=8,    help='Parallel workers (default 8)')
    parser.add_argument('--rate',        type=float, default=8.0,  help='API calls/sec (default 8.0)')
    parser.add_argument('--no-backward', action='store_true',       help='Skip backward extension')
    parser.add_argument('--no-forward',  action='store_true',       help='Skip forward extension')

    args, _ = parser.parse_known_args()

    if args.status:
        show_status()
    elif args.reset:
        reset_progress()
    elif args.update:
        extend_all_stocks(batch_size=args.batch, backward=False, forward=True,
                          force_update=args.force, max_workers=args.workers,
                          calls_per_sec=args.rate)
    elif args.extend:
        extend_all_stocks(
            batch_size=args.batch,
            backward=not args.no_backward,
            forward=not args.no_forward,
            force_update=args.force,
            max_workers=args.workers,
            calls_per_sec=args.rate,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python extend_stock_data.py --update                  # fast parallel daily update")
        print("  python extend_stock_data.py --update --workers 12     # more threads")
        print("  python extend_stock_data.py --update --rate 10        # higher rate cap")
        print("  python extend_stock_data.py --extend                  # one-time backward fill")
        print("  python extend_stock_data.py --extend --batch 500      # batch backward fill")
