"""
Historical Stock Data Acquisition — Date-first approach with parallel workers.

Rate limit (7000-pt account): 500 calls/min = 8.3 calls/sec
Network latency on remote:    ~0.3-0.5s/call  →  2-3 calls/sec single-threaded
With 4 workers sharing limit: 4 × 2 calls/sec  →  ~4× faster end-to-end

Architecture:
  N worker threads each download different dates, sharing one RateLimiter.
  A single writer thread consumes from a queue and flushes to per-stock CSVs.
  No file race conditions — only the writer touches the CSV files.

Usage:
    python get_original_data.py --download              # 1 worker (default)
    python get_original_data.py --download --workers 4  # 4x faster on remote
    python get_original_data.py --status
    python get_original_data.py --retry-failed
    python get_original_data.py --reset
"""

import argparse
import json
import os
import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Earliest A-share trading day (SSE opened 1990-12-19; SZSE 1991-04-03).
# Tushare Pro `pro.daily` returns empty for dates before a stock listed, so
# this floor is safe for both exchanges. Override via run(start_date=...) for
# faster incremental fills.
TARGET_START   = '19901219'

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


# ─── Thread-safe rate limiter ─────────────────────────────────────────────────

class RateLimiter:
    """
    Token-bucket rate limiter shared across multiple worker threads.
    Ensures combined call rate never exceeds the Tushare account limit.
    7000-pt account: 500 calls/min = 8.3/sec → use 8.0 for safety margin.
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
                 batch: int = None, initial: bool = False,
                 workers: int = 1, rate: float = 8.0):
    """
    Download all A-share daily OHLCV by iterating trading dates.

    Args:
        pro:      Tushare Pro API instance.
        start_date / end_date: Date range (YYYYMMDD).
        batch:    Stop after this many dates (safe incremental runs).
        initial:  Skip read-merge on flush (auto-detected on empty sh/).
        workers:  Parallel download threads (default 1).
                  Recommended: 4 for remote servers (hides network latency).
                  Max effective workers ≈ rate_limit / avg_latency.
        rate:     Total API calls/sec budget shared across all workers.
                  7000-pt account: 8.0 (≈480/min). Don't exceed this.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    sh_files = list((DATA_DIR / 'sh').glob('*.csv')) if (DATA_DIR / 'sh').exists() else []
    if initial or not sh_files:
        initial = True
        print("  [Mode] Initial download — write-only flush (fastest)")

    ck = load_checkpoint()
    completed_set = set(ck['completed_dates'])

    all_dates = get_trading_dates(pro, start_date, end_date)
    pending   = [d for d in all_dates if d not in completed_set]
    if batch:
        pending = pending[:batch]

    total     = len(all_dates)
    done      = len(completed_set)
    remaining = len(pending)

    print(f"\n{'='*60}")
    print(f"Date-first download: {start_date} → {end_date}")
    print(f"  Total trading days : {total}")
    print(f"  Already downloaded : {done}")
    print(f"  Remaining          : {remaining}")
    print(f"  Workers            : {workers}  |  rate cap: {rate:.1f} calls/s")
    eta_single = remaining * 0.5 / 60
    eta_multi  = eta_single / workers
    print(f"  ETA (1 worker)     : ~{eta_single:.0f} min")
    if workers > 1:
        print(f"  ETA ({workers} workers)    : ~{eta_multi:.0f} min  ({workers}x speedup)")
    print(f"{'='*60}\n")

    if not pending:
        print("  All dates already downloaded.")
        return

    if workers == 1:
        _download_single(pro, pending, done, total, initial, ck, rate)
    else:
        _download_parallel(pro, pending, done, total, initial, ck, workers, rate)


def _download_single(pro, pending, done, total, initial, ck, rate):
    """Single-threaded download (original logic)."""
    limiter   = RateLimiter(rate)
    buffer    = {}
    n_run     = 0
    t_start   = time.time()

    for trade_date in pending:
        limiter.wait()
        df = download_date(pro, trade_date)

        if df is None:
            ck['failed_dates'].append(trade_date)
            print(f"  [{done+n_run+1:4d}/{total}] {trade_date}  FAILED")
        elif df.empty:
            ck['completed_dates'].append(trade_date)
        else:
            for ts_code, grp in df.groupby('ts_code'):
                buffer.setdefault(ts_code, []).append(grp)
            ck['completed_dates'].append(trade_date)
            n_done = done + n_run + 1
            if (n_run + 1) % 50 == 0:
                rate_act = (n_run + 1) / (time.time() - t_start)
                eta = (total - n_done) / rate_act / 60
                print(f"  [{n_done:4d}/{total}] {trade_date}  "
                      f"{df['ts_code'].nunique()} stocks  "
                      f"({rate_act:.1f}/s ETA ~{eta:.0f}min)")
            else:
                print(f"  [{n_done:4d}/{total}] {trade_date}  {df['ts_code'].nunique()} stocks")

        n_run += 1
        if n_run % FLUSH_EVERY == 0:
            t0 = time.time()
            flush_buffer({k: pd.concat(v) for k, v in buffer.items()}, initial)
            buffer.clear()
            save_checkpoint(ck)
            print(f"  ... checkpoint  ({done+n_run}/{total} done, flush={time.time()-t0:.1f}s)")

    if buffer:
        flush_buffer({k: pd.concat(v) for k, v in buffer.items()}, initial)
    save_checkpoint(ck)
    _print_summary(t_start, n_run, total, ck)


def _download_parallel(pro, pending, done, total, initial, ck, workers, rate):
    """
    Parallel download: N worker threads share one RateLimiter,
    a single writer thread handles all disk I/O (no file race conditions).

    Architecture:
        workers → shared RateLimiter → results_queue → writer thread → CSV files
    """
    limiter      = RateLimiter(rate)
    results_q    = queue.Queue(maxsize=workers * 4)  # bound to limit RAM
    ck_lock      = threading.Lock()
    counter      = {'n': 0, 'failed': 0}
    counter_lock = threading.Lock()
    t_start      = time.time()

    def worker_fn(date_chunk):
        for trade_date in date_chunk:
            limiter.wait()
            df = download_date(pro, trade_date)
            results_q.put((trade_date, df))   # None df = failed

    def writer_fn():
        buffer   = {}
        n_flushed = 0
        while True:
            item = results_q.get()
            if item is None:
                break   # all workers done

            trade_date, df = item
            with counter_lock:
                counter['n'] += 1
                n_total = counter['n']

            if df is None:
                with ck_lock:
                    ck['failed_dates'].append(trade_date)
                counter['failed'] += 1
                print(f"  [{done+n_total:4d}/{total}] {trade_date}  FAILED")
            elif df.empty:
                with ck_lock:
                    ck['completed_dates'].append(trade_date)
            else:
                for ts_code, grp in df.groupby('ts_code'):
                    buffer.setdefault(ts_code, []).append(grp)
                with ck_lock:
                    ck['completed_dates'].append(trade_date)

                n_done = done + n_total
                if n_total % 50 == 0:
                    rate_act = n_total / (time.time() - t_start)
                    eta = (total - n_done) / rate_act / 60
                    print(f"  [{n_done:4d}/{total}] {trade_date}  "
                          f"{df['ts_code'].nunique()} stocks  "
                          f"({rate_act:.1f}/s ETA ~{eta:.0f}min)")
                else:
                    print(f"  [{n_done:4d}/{total}] {trade_date}  {df['ts_code'].nunique()} stocks")

            # Flush every FLUSH_EVERY dates processed by the writer
            if (n_total - counter['failed']) % FLUSH_EVERY == 0 and buffer:
                t0 = time.time()
                flush_buffer({k: pd.concat(v) for k, v in buffer.items()}, initial)
                buffer.clear()
                with ck_lock:
                    save_checkpoint(ck)
                print(f"  ... checkpoint  ({done+n_total}/{total} done, flush={time.time()-t0:.1f}s)")

            results_q.task_done()

        # Final flush
        if buffer:
            flush_buffer({k: pd.concat(v) for k, v in buffer.items()}, initial)
        save_checkpoint(ck)

    # Partition dates across workers (round-robin preserves temporal spread)
    chunks = [pending[i::workers] for i in range(workers)]

    writer_thread = threading.Thread(target=writer_fn, daemon=True)
    writer_thread.start()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(worker_fn, chunk) for chunk in chunks]
        for f in as_completed(futs):
            f.result()   # propagate exceptions

    results_q.put(None)   # signal writer to finish
    writer_thread.join()

    _print_summary(t_start, counter['n'], total, ck)


def _print_summary(t_start, n_run, total, ck):
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"Session complete in {elapsed:.1f} min")
    print(f"  Dates this run : {n_run}")
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

def run(command: str = 'status', batch: int = None,
        workers: int = 1, rate: float = 8.0,
        start_date: str = TARGET_START):
    """
    Scripted entry point.

    Commands: 'init', 'download', 'status', 'retry', 'reset'

    Args:
        batch:      Max dates per run (default: all).
        workers:    Parallel threads (default 1; use 4 on remote servers).
        rate:       Total API calls/sec across all workers (default 8.0).
        start_date: Earliest trading date to download (YYYYMMDD).
                    Default = '19901219' = SSE first trading day. Use a more
                    recent floor (e.g. '20170101') for faster incremental fills.

    Examples:
        run('download')                                      # full history → today
        run('download', workers=4)                           # 4x faster on remote
        run('download', start_date='20170101', workers=4)    # 2017+ only
        run('status')
    """
    setup_directories()
    pro = init_tushare()

    if command == 'init':
        get_stock_list(pro=pro)
        print("Done. Run run('download') to start.")
    elif command == 'download':
        get_stock_list(pro=pro)
        download_all(pro, start_date=start_date, batch=batch,
                     workers=workers, rate=rate)
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
    parser.add_argument('--workers',       type=int, default=1,
                        help='Parallel download threads (default 1; use 4 on remote servers)')
    parser.add_argument('--rate',          type=float, default=8.0,
                        help='Total API calls/sec across all workers (default 8.0 = 480/min)')
    parser.add_argument('--batch',         type=int, default=None,
                        help='Max dates per run (default: all)')
    parser.add_argument('--start-date',    type=str, default=TARGET_START,
                        help=f'Earliest trade date YYYYMMDD (default {TARGET_START} = SSE first trading day)')
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
        download_all(pro, start_date=args.start_date, batch=args.batch,
                     initial=args.initial, workers=args.workers, rate=args.rate)
    else:
        parser.print_help()
        print("\nQuick start (fresh server, 4 workers ≈ 4× faster):")
        print("  python get_original_data.py --download --workers 4")
        print("  python get_original_data.py --status")
        print("  python get_original_data.py --retry-failed")


if __name__ == '__main__':
    main()
