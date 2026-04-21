"""
Tushare Pro News Data Acquisition  (parallel + batched)

Fetches financial news from 9 sources via pro.news().
Data is stored one CSV file per (source, day) under:

  stock_data/news/<src>/YYYYMMDD.csv

Performance improvements over v1
----------------------------------
  Batched API calls   — fetches BATCH_DAYS (default 7) calendar days per request
                        instead of one day at a time → ~7× fewer API calls per source.
  Parallel sources    — all sources run in separate threads simultaneously.
  Shared throttle     — a BoundedSemaphore caps concurrent API calls at
                        MAX_CONCURRENT (default 3) across all threads, keeping
                        total request rate within Tushare's limits.
  Thread-local pro    — each worker thread owns its own tushare connection so
                        there is no cross-thread state sharing.
  Batch done-marking  — done entries are flushed once per batch (one file write)
                        instead of once per day.

Incremental logic:
  - Past days (before today): downloaded once, never re-fetched.
  - Today: always re-fetched to capture the latest articles.
  - stock_data/news/<src>/_done.txt tracks completed past days (resume support).

Sources:
  sina         新浪财经      实时资讯
  wallstreetcn 华尔街见闻    快讯
  10jqka       同花顺        财经新闻
  eastmoney    东方财富      财经新闻
  yuncaijing   云财经        财经新闻
  fenghuang    凤凰新闻      财经新闻
  jinrongjie   金融界        财经新闻
  cls          财联社        快讯
  yicai        第一财经      快讯

Usage:
    from api.news import run

    run()                                 # All 9 sources in parallel
    run('cls')                            # One source only
    run(start_date='2024-01-01')          # Custom start, all sources
    run('cls', force_today=True)          # Re-fetch today for CLS
    run(workers=4, batch_days=14)         # 4 parallel threads, 14-day batches
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'

DATA_DIR = Path('./stock_data')
NEWS_DIR = DATA_DIR / 'news'

# How many calendar days to fetch in a single API call.
# 7 days is a safe default: typical news volume is ~100-400 items/source/day,
# well below Tushare's 5000-row per-call limit.
BATCH_DAYS = 7

# Max source-worker threads.  Actual API concurrency is further limited by the
# semaphore below, so setting this to len(SOURCES) is fine.
MAX_WORKERS = 9

# Hard cap on simultaneous tushare news() calls across ALL threads.
# Prevents hammering the API when many sources are running in parallel.
MAX_CONCURRENT = 3

# Seconds to pause after each successful API call (inside the semaphore slot).
# Effectively spreads calls: MAX_CONCURRENT calls every CALL_INTERVAL seconds.
CALL_INTERVAL = 1.0

MAX_RETRIES = 3
RETRY_DELAY = 2   # base seconds for exponential backoff

# Earliest date Tushare news history is available
# (Tested: sina/eastmoney/wallstreetcn data begins ~Oct-Nov 2018; earlier dates return empty)
DEFAULT_START_DATE = '2018-10-01'

# Sources: (src_id, chinese_name, description)
SOURCES = [
    ('sina',         '新浪财经',   '实时资讯'),
    ('wallstreetcn', '华尔街见闻', '快讯'),
    ('10jqka',       '同花顺',     '财经新闻'),
    ('eastmoney',    '东方财富',   '财经新闻'),
    ('yuncaijing',   '云财经',     '财经新闻'),
    ('fenghuang',    '凤凰新闻',   '财经新闻'),
    ('jinrongjie',   '金融界',     '财经新闻'),
    ('cls',          '财联社',     '快讯'),
    ('yicai',        '第一财经',   '快讯'),
]

SOURCE_IDS = [s[0] for s in SOURCES]

# ─── Thread primitives ────────────────────────────────────────────────────────

# Limits how many threads can be inside pro.news() at the same time.
_api_sem = threading.BoundedSemaphore(MAX_CONCURRENT)

# Thread-local tushare connections: each worker thread initialises its own
# pro instance so there is no shared mutable state.
_tls = threading.local()

# Serialises console output so lines from different threads don't interleave.
_print_lock = threading.Lock()

# Per-source locks for safe concurrent writes to each source's _done.txt.
_done_locks: dict[str, threading.Lock] = {}
_done_locks_guard = threading.Lock()


def _get_pro():
    """Return the thread-local tushare pro instance, creating it if needed."""
    if not hasattr(_tls, 'pro') or _tls.pro is None:
        ts.set_token(TUSHARE_TOKEN)
        _tls.pro = ts.pro_api(TUSHARE_TOKEN)
    return _tls.pro


def _log(src: str, msg: str):
    """Thread-safe print prefixed with source id."""
    with _print_lock:
        print(f"  [{src}] {msg}", flush=True)


def _get_done_lock(src: str) -> threading.Lock:
    with _done_locks_guard:
        if src not in _done_locks:
            _done_locks[src] = threading.Lock()
        return _done_locks[src]


# ─── Date helpers ─────────────────────────────────────────────────────────────

def _date_range(start: str, end: str):
    """Yield every calendar date between start and end inclusive ('YYYY-MM-DD')."""
    d = datetime.strptime(start, '%Y-%m-%d').date()
    e = datetime.strptime(end,   '%Y-%m-%d').date()
    while d <= e:
        yield d
        d += timedelta(days=1)


def _batch_window(batch: list) -> tuple:
    """Return (start_dt_str, end_dt_str) covering the full span of a day-batch."""
    return (
        batch[0].strftime('%Y-%m-%d 00:00:00'),
        batch[-1].strftime('%Y-%m-%d 23:59:59'),
    )


# ─── Progress tracking ────────────────────────────────────────────────────────

def _done_file(src: str) -> Path:
    return NEWS_DIR / src / '_done.txt'


def _load_done(src: str) -> set:
    """Return set of YYYYMMDD strings already fully downloaded for `src`."""
    fp = _done_file(src)
    if not fp.exists():
        return set()
    return set(fp.read_text(encoding='utf-8').splitlines())


def _mark_done_batch(src: str, day_strs: list):
    """Append a list of YYYYMMDD strings to the done file in one write."""
    if not day_strs:
        return
    fp = _done_file(src)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with _get_done_lock(src):
        with fp.open('a', encoding='utf-8') as f:
            f.write('\n'.join(day_strs) + '\n')


# ─── Storage ──────────────────────────────────────────────────────────────────

def _day_csv(src: str, day: date) -> Path:
    return NEWS_DIR / src / f"{day.strftime('%Y%m%d')}.csv"


def _save_day(df: pd.DataFrame, src: str, day: date):
    """Write one day's news slice to its CSV file."""
    fp = _day_csv(src, day)
    fp.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    df.to_csv(fp, index=False, encoding='utf-8-sig')


# ─── API fetch (throttled + retried) ──────────────────────────────────────────

def _fetch_news(src: str, start_dt: str, end_dt: str) -> Optional[pd.DataFrame]:
    """
    Fetch one batch (start_dt → end_dt) with shared throttle and retry.

    Acquires the global semaphore so at most MAX_CONCURRENT calls run
    concurrently across all source threads.  Holds the slot for CALL_INTERVAL
    seconds after a successful call to provide natural pacing.
    """
    for attempt in range(MAX_RETRIES):
        with _api_sem:                          # ← shared concurrency cap
            try:
                df = _get_pro().news(src=src, start_date=start_dt, end_date=end_dt)
                time.sleep(CALL_INTERVAL)       # pace while holding the slot
                return df
            except Exception as e:
                err = str(e)
                if any(k in err for k in ('exceed', 'limit', '频率', 'too many')):
                    wait = 60 * (attempt + 1)
                    _log(src, f"[rate limit] sleeping {wait}s (attempt {attempt + 1}) ...")
                    time.sleep(wait)
                elif any(k in err.lower() for k in ('permission', '权限', 'not subscribed')):
                    _log(src, f"[permission denied] {err[:100]}")
                    return None
                else:
                    wait = RETRY_DELAY * (2 ** attempt)
                    _log(src, f"[{type(e).__name__}] retry {attempt + 1}/{MAX_RETRIES} in {wait}s ...")
                    time.sleep(wait)
    return None


# ─── Core per-source downloader ───────────────────────────────────────────────

def fetch_source(
    src: str,
    start_date: str   = DEFAULT_START_DATE,
    force_today: bool = False,
    batch_days: int   = BATCH_DAYS,
) -> dict:
    """
    Incrementally download all news for one source from start_date to today.

    Fetches `batch_days` calendar days per API call, then splits the result
    by date before saving.  Past days are recorded in _done.txt and skipped
    on future runs.  Today's data is always re-fetched.

    Returns:
        {'ok': int, 'empty': int, 'fail': int}
    """
    today     = date.today()
    today_str = today.strftime('%Y%m%d')
    done      = _load_done(src)

    all_days = list(_date_range(start_date, today.strftime('%Y-%m-%d')))
    pending  = [
        d for d in all_days
        if d.strftime('%Y%m%d') not in done
        or (d == today and force_today)
    ]

    if not pending:
        _log(src, f"All {len(all_days)} days already done — skip.")
        return {'ok': 0, 'empty': 0, 'fail': 0}

    # Chunk pending days into fixed-size batches
    batches = [pending[i:i + batch_days]
               for i in range(0, len(pending), batch_days)]

    _log(src, f"{len(pending)} pending days → {len(batches)} batches "
              f"(batch_size={batch_days})")

    ok = fail = skip = 0

    for b_idx, batch in enumerate(batches):
        start_dt, end_dt = _batch_window(batch)
        df = _fetch_news(src, start_dt, end_dt)

        # Days in this batch that are NOT today (safe to mark as permanently done)
        past_day_strs = [d.strftime('%Y%m%d') for d in batch if d != today]

        if df is None:
            # Persistent failure — log and move on; these days will retry next run
            fail += len(batch)
            _log(src, f"  batch {b_idx + 1}/{len(batches)} FAILED  "
                      f"({batch[0].strftime('%Y%m%d')} – {batch[-1].strftime('%Y%m%d')})")
            continue

        if df.empty:
            # Valid empty window (e.g. weekend, no news published)
            skip += len(batch)
            _mark_done_batch(src, past_day_strs)
            continue

        # Split the batch result by calendar day and save each slice
        df['_date_key'] = pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d')
        for day in batch:
            day_str = day.strftime('%Y%m%d')
            day_df  = df[df['_date_key'] == day_str].drop(columns=['_date_key'])
            if not day_df.empty:
                _save_day(day_df, src, day)
                ok += 1
            else:
                skip += 1  # no articles on this specific day within the window

        _mark_done_batch(src, past_day_strs)

        # Progress log every 10 batches (or on the final one)
        if (b_idx + 1) % 10 == 0 or b_idx == len(batches) - 1:
            pct = (b_idx + 1) / len(batches) * 100
            _log(src, f"  {b_idx + 1}/{len(batches)} batches ({pct:.0f}%)  "
                      f"ok={ok} empty={skip} fail={fail}")

    _log(src, f"Done — ok={ok} empty={skip} fail={fail}")
    return {'ok': ok, 'empty': skip, 'fail': fail}


# ─── Public entry point ───────────────────────────────────────────────────────

def run(
    src: str          = None,
    start_date: str   = DEFAULT_START_DATE,
    force_today: bool = False,
    workers: int      = MAX_WORKERS,
    batch_days: int   = BATCH_DAYS,
):
    """
    Download financial news from Tushare Pro (parallel + batched).

    Sources run in parallel threads; actual API concurrency is capped at
    MAX_CONCURRENT (3) by a shared semaphore regardless of `workers`.

    Args:
        src (str | None):
            Specific source ID, or None to run all sources in parallel.
            Valid values: sina, wallstreetcn, 10jqka, eastmoney, yuncaijing,
                          fenghuang, jinrongjie, cls, yicai
        start_date (str):
            Earliest date to download, 'YYYY-MM-DD' (default '2018-10-01').
        force_today (bool):
            Re-fetch today's articles even if already downloaded.
        workers (int):
            Max parallel source-worker threads (default: all 9 sources).
        batch_days (int):
            Calendar days to cover per API call (default 7).

    Storage:
        stock_data/news/<src>/YYYYMMDD.csv   — one file per source per day
        stock_data/news/<src>/_done.txt       — resume tracker

    Examples:
        from api.news import run

        run()                              # All 9 sources in parallel
        run('cls', force_today=True)       # One source, re-fetch today
        run(start_date='2024-01-01')       # Custom start, all sources
        run(workers=3, batch_days=14)      # Conservative: 3 threads, 14-day batches
    """
    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    if src:
        if src not in SOURCE_IDS:
            print(f"Unknown source '{src}'.  Valid: {', '.join(SOURCE_IDS)}")
            return
        sources_to_run = [(s, cn, d) for (s, cn, d) in SOURCES if s == src]
    else:
        sources_to_run = SOURCES

    n_workers    = min(workers, len(sources_to_run))
    today_label  = date.today().strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"Tushare News Acquisition  [{today_label}]")
    print(f"{'='*60}")
    print(f"Sources        : {', '.join(s for s, *_ in sources_to_run)}")
    print(f"Start date     : {start_date}")
    print(f"Batch days     : {batch_days}  (calendar days per API call)")
    print(f"Workers        : {n_workers}  (parallel source threads)")
    print(f"API concurrency: {MAX_CONCURRENT}  (shared semaphore cap)")
    print(f"Force today    : {force_today}")
    print()

    t0 = time.perf_counter()

    if n_workers == 1:
        # Single source — run directly without thread overhead
        for src_id, cn_name, desc in sources_to_run:
            print(f"[{src_id}] {cn_name} — {desc}")
            try:
                fetch_source(src_id, start_date=start_date,
                             force_today=force_today, batch_days=batch_days)
            except Exception as e:
                print(f"  [{src_id}] ERROR: {e}")
    else:
        with ThreadPoolExecutor(max_workers=n_workers,
                                thread_name_prefix='news') as pool:
            futures = {
                pool.submit(
                    fetch_source, src_id,
                    start_date, force_today, batch_days
                ): (src_id, cn_name)
                for src_id, cn_name, _ in sources_to_run
            }
            for fut in as_completed(futures):
                src_id, cn_name = futures[fut]
                try:
                    fut.result()    # re-raise worker exceptions into the main thread
                except Exception as e:
                    with _print_lock:
                        print(f"  [{src_id}] UNHANDLED ERROR: {e}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"News acquisition complete  [{date.today().strftime('%Y-%m-%d')}]")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


# ─── Query helper ─────────────────────────────────────────────────────────────

def load_news(src: str,
              start_date: str = None,
              end_date:   str = None) -> pd.DataFrame:
    """
    Load downloaded news for a source into a single DataFrame.

    Args:
        src:        Source identifier (e.g. 'sina').
        start_date: 'YYYY-MM-DD' filter (inclusive).  None = all available.
        end_date:   'YYYY-MM-DD' filter (inclusive).  None = all available.

    Returns:
        DataFrame sorted by datetime ascending, or empty DataFrame if nothing found.
    """
    src_dir = NEWS_DIR / src
    if not src_dir.exists():
        return pd.DataFrame()

    csv_files = sorted(src_dir.glob('????????.csv'))  # match YYYYMMDD.csv only

    if start_date:
        s = start_date.replace('-', '')
        csv_files = [f for f in csv_files if f.stem >= s]
    if end_date:
        e = end_date.replace('-', '')
        csv_files = [f for f in csv_files if f.stem <= e]

    if not csv_files:
        return pd.DataFrame()

    frames = []
    for fp in csv_files:
        try:
            frames.append(pd.read_csv(fp, encoding='utf-8-sig'))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').reset_index(drop=True)
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Tushare Pro financial news (parallel + batched).'
    )
    parser.add_argument(
        '--src', default=None,
        help=f"Source ID.  One of: {', '.join(SOURCE_IDS)}.  Default: all."
    )
    parser.add_argument(
        '--start-date', default=DEFAULT_START_DATE,
        help=f"Earliest date to fetch (YYYY-MM-DD, default {DEFAULT_START_DATE})"
    )
    parser.add_argument(
        '--force-today', action='store_true',
        help="Re-fetch today's articles even if already downloaded"
    )
    parser.add_argument(
        '--workers', type=int, default=MAX_WORKERS,
        help=f"Parallel source threads (default {MAX_WORKERS})"
    )
    parser.add_argument(
        '--batch-days', type=int, default=BATCH_DAYS,
        help=f"Calendar days per API call (default {BATCH_DAYS})"
    )
    args, _ = parser.parse_known_args()
    run(
        src         = args.src,
        start_date  = args.start_date,
        force_today = args.force_today,
        workers     = args.workers,
        batch_days  = args.batch_days,
    )
