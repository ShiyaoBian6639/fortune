"""
Block Trade Data Acquisition from Tushare Pro
- Fetches daily block (large) trade records for all stocks
- Saves one file per trading date: stock_data/block_trade/block_trade_YYYYMMDD.csv
- Skips dates that already have a local file (incremental)
- Batches BATCH_DAYS trading days per API call to reduce total calls (key speed optimization:
  one call covers 10 days → 10× fewer calls vs per-day fetching)

Usage:
    from api.block_trade import run

    run('download')                            # catch up to yesterday
    run('range', start_date='20230101', end_date='20230331')
    run('status')
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR      = Path('./stock_data/block_trade')
START_DATE    = '20170101'
WORKERS       = 8       # parallel batch workers
CALLS_PER_SEC = 8.0     # global token-bucket rate
MAX_RETRIES   = 3
RETRY_DELAY   = 2
BATCH_DAYS    = 10      # trading days fetched per API call (reduces total calls 10×)

FIELDS = ['ts_code', 'trade_date', 'price', 'vol', 'amount', 'buyer', 'seller']

# ─── Rate limiter ─────────────────────────────────────────────────────────────

class _RateLimiter:
    """Token-bucket: acquire() sleeps OUTSIDE the lock so other threads aren't blocked."""
    def __init__(self, rate):
        self._interval = 1.0 / rate
        self._lock     = threading.Lock()
        self._last     = 0.0

    def acquire(self):
        while True:
            with self._lock:
                now  = time.monotonic()
                wait = self._last + self._interval - now
                if wait <= 0:
                    self._last = now
                    return          # token granted
            # Sleep outside the lock so other threads can check their turn
            time.sleep(max(0.001, wait))


_limiter  = _RateLimiter(CALLS_PER_SEC)
_pro      = None
_pro_lock = threading.Lock()


def _get_pro():
    global _pro
    with _pro_lock:
        if _pro is None:
            ts.set_token(TUSHARE_TOKEN)
            _pro = ts.pro_api(TUSHARE_TOKEN)
        return _pro


def _fetch(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if any(k in err for k in ('exceed', 'limit', '频率', 'too many')):
                wait = 60 * (attempt + 1)
                print(f"    [rate limit] sleeping {wait}s ...")
                time.sleep(wait)
            elif any(k in err.lower() for k in ('permission', '权限', 'not subscribed')):
                print(f"    [permission denied] {err[:120]}")
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _setup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _existing_dates():
    return {f.stem.replace('block_trade_', '') for f in DATA_DIR.glob('block_trade_*.csv')}


def _trade_calendar(start_date, end_date):
    pro = _get_pro()
    cal = _fetch(pro.trade_cal, exchange='SSE',
                 start_date=start_date, end_date=end_date, is_open='1')
    if cal is None or cal.empty:
        return []
    return cal['cal_date'].tolist()


def _download_batch(dates_in_batch):
    """
    Fetch block trades for a list of consecutive trading dates using a single
    date-range API call. Splits the result by date and saves per-file.
    Returns (n_dates_saved, n_dates_empty, n_fail).
    """
    if not dates_in_batch:
        return 0, 0, 0

    start = dates_in_batch[0]
    end   = dates_in_batch[-1]
    pro   = _get_pro()

    df = _fetch(pro.block_trade, start_date=start, end_date=end,
                fields=','.join(FIELDS))

    if df is None:
        return 0, 0, len(dates_in_batch)

    # Split by trade_date and save per-date file
    saved = empty = 0
    if not df.empty:
        df['trade_date'] = df['trade_date'].astype(str)
        by_date = dict(tuple(df.groupby('trade_date')))
    else:
        by_date = {}

    for date in dates_in_batch:
        fp  = DATA_DIR / f'block_trade_{date}.csv'
        day = by_date.get(date, pd.DataFrame(columns=FIELDS))
        day.to_csv(fp, index=False, encoding='utf-8-sig')
        if len(day) > 0:
            saved += 1
        else:
            empty += 1

    return saved, empty, 0


# ─── Public interface ─────────────────────────────────────────────────────────

def download_range(start_date, end_date, skip_existing=True):
    """Download block trades for every trading day in [start_date, end_date]."""
    _setup()

    print(f"Fetching trading calendar {start_date}→{end_date} ...")
    all_dates = _trade_calendar(start_date, end_date)
    if not all_dates:
        print("No trading dates found.")
        return

    if skip_existing:
        existing    = _existing_dates()
        missing     = [d for d in all_dates if d not in existing]
        print(f"  {len(all_dates)} trading days, "
              f"{len(existing & set(all_dates))} already present, "
              f"{len(missing)} to download")
        todo = missing
    else:
        todo = all_dates

    if not todo:
        print("Nothing to download.")
        return

    # Sort ascending so start_date < end_date in each batch API call
    todo.sort()
    # Split into batches of BATCH_DAYS consecutive trading days
    batches = [todo[i:i + BATCH_DAYS] for i in range(0, len(todo), BATCH_DAYS)]
    total_dates   = len(todo)
    total_batches = len(batches)
    print(f"Downloading {total_dates} dates in {total_batches} batches "
          f"({BATCH_DAYS} days/batch, {WORKERS} workers @ {CALLS_PER_SEC} calls/s) ...")

    saved = empty = fail = done_batches = 0
    t0    = time.monotonic()
    lock  = threading.Lock()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_batch, b): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                s, e, f = fut.result()
            except Exception as exc:
                s, e, f = 0, 0, len(batch)
                print(f"  batch {batch[0]}–{batch[-1]}: exception {exc}")

            with lock:
                saved         += s
                empty         += e
                fail          += f
                done_batches  += 1
                done_dates     = saved + empty + fail

            if done_batches % 10 == 0 or done_batches == total_batches:
                elapsed = time.monotonic() - t0
                rate    = done_dates / elapsed if elapsed > 0 else 0
                eta     = (total_dates - done_dates) / rate if rate > 0 else 0
                print(f"  [{done_batches}/{total_batches} batches | "
                      f"{done_dates}/{total_dates} dates]  "
                      f"saved={saved} empty={empty} fail={fail}  "
                      f"{rate:.1f} dates/s  ETA {eta:.0f}s")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min). "
          f"saved={saved} empty={empty} fail={fail}")
    if fail:
        print("  Re-run to retry failed dates (skipped if already saved).")


def status():
    """Print coverage summary."""
    _setup()
    existing = sorted(_existing_dates())
    if not existing:
        print("No block_trade data downloaded yet.")
        return
    print(f"block_trade/ has {len(existing)} date files")
    print(f"  Earliest: {existing[0]}")
    print(f"  Latest:   {existing[-1]}")

    db_dir = Path('./stock_data/daily_basic')
    if db_dir.exists():
        db_dates = {f.stem.replace('daily_basic_', '') for f in db_dir.glob('daily_basic_*.csv')}
        missing  = sorted(db_dates - set(existing))
        if missing:
            print(f"\n  Missing {len(missing)} dates vs daily_basic/ — run run('download') to fill")
        else:
            print("\n  Coverage is in sync with daily_basic/")


def run(action='download', **kwargs):
    """
    Entry point for block_trade data operations.

    Actions:
        'download'  – incremental catch-up to yesterday (default)
        'range'     – explicit date range
        'status'    – print coverage

    Keyword args:
        start_date  (str)  YYYYMMDD
        end_date    (str)  YYYYMMDD
        force       (bool) – re-download existing files
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    force     = kwargs.get('force', False)

    if action in ('download', 'backfill'):
        start = kwargs.get('start_date', START_DATE)
        end   = kwargs.get('end_date',   yesterday)
        download_range(start, end, skip_existing=not force)

    elif action == 'range':
        start = kwargs.get('start_date', START_DATE)
        end   = kwargs.get('end_date',   yesterday)
        download_range(start, end, skip_existing=not force)

    elif action == 'status':
        status()

    else:
        print(f"Unknown action: {action!r}. Valid: download | range | backfill | status")


if __name__ == '__main__':
    run('status')
    run('download')
