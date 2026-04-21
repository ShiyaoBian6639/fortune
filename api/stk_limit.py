"""
Stock Limit Up/Down Data Acquisition from Tushare Pro
- Fetches daily limit-up / limit-down prices for all stocks
- Saves one file per trading date: stock_data/stk_limit/stk_limit_YYYYMMDD.csv
- Skips dates that already have a local file (incremental)

Usage:
    from api.stk_limit import run

    run('download')                                          # catch up to yesterday
    run('range', start_date='20200623', end_date='20200817') # backfill a range
    run('status')                                            # show coverage
"""

import threading
import tushare as ts
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR      = Path('./stock_data/stk_limit')
START_DATE    = '20170101'
WORKERS       = 4
CALLS_PER_SEC = 4.0
MAX_RETRIES   = 5
RETRY_DELAY   = 2

FIELDS = ['trade_date', 'ts_code', 'up_limit', 'down_limit']

# ─── Rate limiter + shared pro ────────────────────────────────────────────────

class _RateLimiter:
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
                    return
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _setup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_with_retry(fetch_func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if any(k in err for k in ('exceed', 'limit', '频率', 'too many')):
                wait = 60 * (attempt + 1)
                print(f"  [rate limit] sleeping {wait}s ...")
                time.sleep(wait)
            elif any(k in err.lower() for k in ('permission', '权限', 'not subscribed')):
                print(f"  [permission denied] {err[:120]}")
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


def _existing_dates():
    return {f.stem.replace('stk_limit_', '') for f in DATA_DIR.glob('stk_limit_*.csv')}


def _trade_calendar(start_date, end_date):
    pro = _get_pro()
    cal = _fetch_with_retry(pro.trade_cal, exchange='SSE',
                            start_date=start_date, end_date=end_date, is_open='1')
    if cal is None or cal.empty:
        return []
    return cal['cal_date'].tolist()


def _download_one_date(date):
    pro = _get_pro()
    df  = _fetch_with_retry(pro.stk_limit, trade_date=date, fields=','.join(FIELDS))
    if df is not None and not df.empty:
        fp = DATA_DIR / f'stk_limit_{date}.csv'
        df.to_csv(fp, index=False, encoding='utf-8-sig')
        return date, len(df), 'ok'
    return date, 0, 'fail'


# ─── Public interface ─────────────────────────────────────────────────────────

def download_range(start_date, end_date, skip_existing=True):
    """
    Download stk_limit for every trading day in [start_date, end_date].
    Uses parallel workers; safe to interrupt and re-run.
    """
    _setup()

    print(f"Fetching trading calendar {start_date}→{end_date} ...")
    trade_dates = _trade_calendar(start_date, end_date)
    if not trade_dates:
        print("No trading dates found.")
        return

    if skip_existing:
        existing    = _existing_dates()
        missing     = [d for d in trade_dates if d not in existing]
        print(f"  {len(trade_dates)} trading days, "
              f"{len(existing & set(trade_dates))} already present, "
              f"{len(missing)} to download")
        trade_dates = missing

    if not trade_dates:
        print("Nothing to download.")
        return

    total = len(trade_dates)
    ok = fail = 0
    t0 = time.monotonic()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_one_date, d): d for d in trade_dates}
        done = 0
        for fut in as_completed(futures):
            done += 1
            _, n, status = fut.result()
            with lock:
                if status == 'ok':
                    ok += 1
                else:
                    fail += 1
            if done % 20 == 0 or done == total:
                elapsed = time.monotonic() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}]  ok={ok} fail={fail}  "
                      f"{rate:.1f} dates/s  ETA {eta:.0f}s")

    print(f"\nDone in {time.monotonic()-t0:.0f}s. {ok} succeeded, {fail} failed.")
    if fail:
        print("  Re-run to retry failed dates.")


def status():
    """Print coverage summary."""
    _setup()
    existing = sorted(_existing_dates())
    if not existing:
        print("No stk_limit data downloaded yet.")
        return
    print(f"stk_limit/ has {len(existing)} date files")
    print(f"  Earliest: {existing[0]}")
    print(f"  Latest:   {existing[-1]}")

    db_dir = Path('./stock_data/daily_basic')
    if db_dir.exists():
        db_dates = {f.stem.replace('daily_basic_', '') for f in db_dir.glob('daily_basic_*.csv')}
        sl_dates = set(existing)
        missing  = sorted(db_dates - sl_dates)
        if missing:
            print(f"\n  Missing {len(missing)} dates vs daily_basic:")
            for d in missing[:20]:
                print(f"    {d}")
            if len(missing) > 20:
                print(f"    ... and {len(missing)-20} more")
        else:
            print("\n  Coverage is in sync with daily_basic/")


def run(action='download', **kwargs):
    """
    Entry point for stk_limit data operations.

    Actions:
        'download'  – incremental catch-up to yesterday (default)
        'range'     – download an explicit date range
        'status'    – print coverage summary
        'backfill'  – alias for 'range' from START_DATE to yesterday

    Keyword args:
        start_date  (str)  YYYYMMDD
        end_date    (str)  YYYYMMDD
        force       (bool) – re-download even if file exists
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
