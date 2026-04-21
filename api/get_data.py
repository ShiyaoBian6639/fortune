"""
Tushare Pro Data Acquisition - Index and Fund Data

Datasets (executed in order):
  Index: index_weight, index_dailybasic, index_classify, index_member_all,
         index_global, idx_factor_pro
  Fund:  fund_basic, fund_company, fund_manager, fund_share, fund_nav,
         fund_div, fund_portfolio, fund_factor_pro

Storage layout:
  stock_data/index/index_weight/<index_code>.csv      (time-series per index)
  stock_data/index/index_dailybasic/<index_code>.csv
  stock_data/index/index_classify.csv                 (static)
  stock_data/index/index_member_all/<index_code>.csv  (static per index)
  stock_data/index/index_global/<ts_code>.csv         (time-series per global index)
  stock_data/index/idx_factor_pro/<index_code>.csv    (time-series per index)
  stock_data/fund/fund_basic.csv                      (static)
  stock_data/fund/fund_company.csv                    (static)
  stock_data/fund/fund_manager/<ts_code>.csv          (static per fund)
  stock_data/fund/fund_share/<ts_code>.csv            (time-series per fund)
  stock_data/fund/fund_nav/<ts_code>.csv              (time-series per fund)
  stock_data/fund/fund_div/<ts_code>.csv              (event-based per fund)
  stock_data/fund/fund_portfolio/<ts_code>.csv        (quarterly per fund)
  stock_data/fund/fund_factor_pro/<ts_code>.csv       (time-series per fund)

Incremental updates:
  - Time-series files: reads max date from existing CSV, fetches only newer data.
  - Static/slow-changing files: skipped if file is younger than STATIC_REFRESH_DAYS.
  - Force refresh available via force=True.

Usage:
    from api.get_data import run

    run()                          # All datasets in order
    run('index_weight')            # One dataset only
    run('fund_nav', batch=200)     # Fund endpoints with batch limit
    run(force=True)                # Force refresh static data too
"""

import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'

DATA_DIR  = Path('./stock_data')
INDEX_DIR = DATA_DIR / 'index'
FUND_DIR  = DATA_DIR / 'fund'

MAX_RETRIES         = 3
RETRY_DELAY         = 2     # base seconds for exponential backoff
START_DATE          = '20170101'
STATIC_REFRESH_DAYS = 7     # re-download static files after this many days

# Concurrency: number of parallel API threads and global call rate.
# Increase CALLS_PER_SEC if your Tushare tier allows higher throughput;
# decrease if you see frequent rate-limit errors.
WORKERS       = 16    # concurrent API threads for per-fund endpoints
                      # Rule of thumb: WORKERS ≥ CALLS_PER_SEC × avg_latency_s
                      # With ~1.5s avg latency and 8/s target → need ≥12 workers
CALLS_PER_SEC = 8.0   # global token-bucket ceiling across all threads

# Chinese domestic indices
CHINESE_INDICES = [
    '000001.SH',  # SSE Composite
    '000010.SH',  # SSE 180
    '000016.SH',  # SSE 50
    '000300.SH',  # CSI 300
    '000688.SH',  # STAR Market 50
    '000852.SH',  # CSI 1000
    '000904.SH',  # CSI 200
    '000905.SH',  # CSI 500
    '399001.SZ',  # SZSE Component
    '399006.SZ',  # ChiNext
    '399016.SZ',  # ChiNext 50
    '399300.SZ',  # CSI 300 (SZ-listed)
    '399905.SZ',  # CSI 500 (SZ-listed)
]

# Global indices
GLOBAL_INDICES = [
    'XIN9',   # MSCI China A50
    'DJI',    # Dow Jones Industrial Average
    'SPX',    # S&P 500
    'IXIC',   # NASDAQ Composite
    'N225',   # Nikkei 225
    'HSI',    # Hang Seng Index
    'FTSE',   # FTSE 100
    'GDAXI',  # DAX
    'FCHI',   # CAC 40
    'KS11',   # KOSPI
]

# ─── Rate limiter (shared across all threads) ─────────────────────────────────

class _RateLimiter:
    """
    Token-bucket rate limiter — enforces at most `rate` calls/second globally
    across all threads.  Each call to acquire() blocks until a token is available.

    Critical: sleep OUTSIDE the lock so waiting threads don't block each other.
    The old pattern (sleep inside `with self._lock`) serialised all worker sleeps
    through a single mutex, cutting effective throughput to ~1/WORKERS of target.
    """
    def __init__(self, rate: float):
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
                    return          # token granted — exit immediately
            # Sleep *outside* the lock so other threads can check concurrently
            time.sleep(max(0.001, wait))


_limiter = _RateLimiter(CALLS_PER_SEC)

# ─── Core helpers ─────────────────────────────────────────────────────────────

_pro      = None
_pro_lock = threading.Lock()


def _get_pro():
    global _pro
    with _pro_lock:
        if _pro is None:
            ts.set_token(TUSHARE_TOKEN)
            _pro = ts.pro_api(TUSHARE_TOKEN)
        return _pro


def _today():
    return datetime.now().strftime('%Y%m%d')


def _next_day(date_str):
    """Return the next calendar day as YYYYMMDD string."""
    d = datetime.strptime(str(int(float(date_str))), '%Y%m%d') + timedelta(days=1)
    return d.strftime('%Y%m%d')


def _fetch(func, *args, **kwargs):
    """
    Rate-limit then call func; retry on transient errors.
    The rate limiter is acquired once per attempt so long sleeps on rate-limit
    errors don't double-charge the token bucket.
    """
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            # Check rate-limit BEFORE permission: the Tushare per-minute rate-limit
            # message embeds "权限" in its documentation URL, which would otherwise
            # be misclassified as a permanent permission error.
            if any(k in err for k in ('exceed', 'limit', '频率', 'too many', '每分钟', '每天最多')):
                wait = 60 * (attempt + 1)
                print(f"    [rate limit] sleeping {wait}s ...")
                time.sleep(wait)
            elif any(k in err.lower() for k in ('permission', 'not subscribed')) or \
                 ('权限' in err and '每分钟' not in err and '每天' not in err):
                print(f"    [permission denied] {err[:120]}")
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


def _last_date(filepath, date_col):
    """Return max date (YYYYMMDD str) from an existing CSV, or None."""
    fp = Path(filepath)
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, usecols=[date_col])
        if df.empty:
            return None
        return str(int(float(df[date_col].max())))
    except Exception:
        return None


def _is_fresh(filepath, days=STATIC_REFRESH_DAYS):
    """True if file exists and is younger than `days` days."""
    fp = Path(filepath)
    if not fp.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(fp.stat().st_mtime)).days
    return age < days


def _normalize_date_col(series):
    """Coerce a date column to uniform YYYYMMDD strings, regardless of
    whether it was stored as int (20260414) or string ('20260414').
    If the column is already non-numeric (e.g. ts_code), returns it unchanged."""
    if pd.api.types.is_numeric_dtype(series):
        return series.apply(lambda x: str(int(x)) if pd.notna(x) else x)
    return series  # already string — no conversion needed


def _upsert(df, filepath, key_cols, sort_col=None):
    """
    Merge df into the existing CSV (dedup by key_cols) and save.
    Creates parent directories as needed.
    """
    if df is None or df.empty:
        return 0
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fp.exists():
        try:
            existing = pd.read_csv(fp)
            # Normalize key/sort columns to string so concat doesn't mix
            # str and int — pandas reads YYYYMMDD numeric dates as int64.
            cols_to_norm = set(key_cols) | ({sort_col} if sort_col else set())
            for col in cols_to_norm:
                if col and col in existing.columns:
                    existing[col] = _normalize_date_col(existing[col])
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass  # corrupt file — overwrite with new data
    # Normalize in the combined frame before dedup/sort too
    cols_to_norm = set(key_cols) | ({sort_col} if sort_col else set())
    for col in cols_to_norm:
        if col and col in df.columns:
            df[col] = _normalize_date_col(df[col])
    df = df.drop_duplicates(subset=key_cols, keep='last')
    if sort_col:
        df = df.sort_values(sort_col).reset_index(drop=True)
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    return len(df)


def _save(df, filepath):
    """Overwrite a CSV file (creates parent dirs)."""
    if df is None or df.empty:
        return
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp, index=False, encoding='utf-8-sig')


def _load_ckpt(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'completed': [], 'failed': []}


def _save_ckpt(path, ckpt):
    Path(path).write_text(json.dumps(ckpt, indent=2), encoding='utf-8')


def _load_no_data(path):
    """Load the set of fund codes known to have no data from Tushare."""
    p = Path(path)
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding='utf-8')))
        except Exception:
            pass
    return set()


def _save_no_data(path, no_data_set):
    Path(path).write_text(json.dumps(sorted(no_data_set), indent=2), encoding='utf-8')


def _auto_init_no_data(fund_codes, out_dir, no_data_path):
    """
    One-time bootstrap: mark every fund code that has no existing CSV file as
    no_data.  Safe to call when the directory already has files from prior runs
    (any fund that had data would already have been saved).  Skips init if:
      - _no_data.json already exists (already initialised)
      - out_dir has zero CSV files (fresh start — nothing to infer from)
    Returns the newly-built no_data set (empty set if init was skipped).
    """
    if no_data_path.exists() or not out_dir.exists():
        return set()
    existing = {f.name for f in out_dir.glob('*.csv') if not f.name.startswith('_')}
    if not existing:
        return set()   # No prior data — can't infer anything

    no_data = {code for code in fund_codes
               if f"{_safe_code(code)}.csv" not in existing}
    if no_data:
        _save_no_data(no_data_path, no_data)
        print(f"  Auto-initialized _no_data.json: "
              f"{len(no_data)}/{len(fund_codes)} funds with no existing files "
              f"will be skipped (run with force=True to re-check)")
    return no_data


def _get_fund_codes(exchange_only=False):
    """
    Read fund ts_codes from fund_basic.csv.
    If exchange_only=True, return only exchange-listed (market='E') funds.
    """
    fp = FUND_DIR / 'fund_basic.csv'
    if not fp.exists():
        return []
    try:
        cols = ['ts_code', 'market'] if exchange_only else ['ts_code']
        df = pd.read_csv(fp, usecols=cols)
        if exchange_only and 'market' in df.columns:
            df = df[df['market'] == 'E']
        return df['ts_code'].dropna().tolist()
    except Exception:
        return []


def _safe_code(code):
    """Convert ts_code to a safe filename stem (replace '.' with '_')."""
    return code.replace('.', '_')


def _pre_scan_dates(codes, out_dir, date_col):
    """
    Read last dates for all codes in parallel (I/O-bound; uses threads).
    Returns dict {code: last_date_str_or_None}.
    """
    results = {}

    def _read_one(code):
        fp = out_dir / f"{_safe_code(code)}.csv"
        return code, _last_date(fp, date_col)

    with ThreadPoolExecutor(max_workers=min(16, len(codes))) as ex:
        for code, last in ex.map(_read_one, codes):
            results[code] = last
    return results


# ─── 1. index_weight ──────────────────────────────────────────────────────────

def fetch_index_weight(force=False):
    out_dir = INDEX_DIR / 'index_weight'
    out_dir.mkdir(parents=True, exist_ok=True)
    pro   = _get_pro()
    today = _today()

    print(f"\n[index_weight] {len(CHINESE_INDICES)} indices ...")
    for code in CHINESE_INDICES:
        fp   = out_dir / f"{_safe_code(code)}.csv"
        last = _last_date(fp, 'trade_date')
        start = _next_day(last) if (last and not force) else START_DATE

        if start > today:
            print(f"  {code}: up-to-date (last={last})")
            continue

        print(f"  {code}: {start} → {today} ...", end=' ', flush=True)
        df = _fetch(pro.index_weight, index_code=code, start_date=start, end_date=today)
        if df is not None and not df.empty:
            n = _upsert(df, fp, key_cols=['trade_date', 'con_code'], sort_col='trade_date')
            print(f"{len(df)} new rows (total {n})")
        elif df is not None:
            print("no data (empty response)")
        else:
            print("error — skipped (see log above)")


# ─── 2. index_dailybasic ──────────────────────────────────────────────────────

def fetch_index_dailybasic(force=False):
    out_dir = INDEX_DIR / 'index_dailybasic'
    out_dir.mkdir(parents=True, exist_ok=True)
    pro   = _get_pro()
    today = _today()

    print(f"\n[index_dailybasic] {len(CHINESE_INDICES)} indices ...")
    for code in CHINESE_INDICES:
        fp   = out_dir / f"{_safe_code(code)}.csv"
        last = _last_date(fp, 'trade_date')
        start = _next_day(last) if (last and not force) else START_DATE

        if start > today:
            print(f"  {code}: up-to-date (last={last})")
            continue

        print(f"  {code}: {start} → {today} ...", end=' ', flush=True)
        df = _fetch(pro.index_dailybasic, ts_code=code, start_date=start, end_date=today)
        if df is not None and not df.empty:
            n = _upsert(df, fp, key_cols=['ts_code', 'trade_date'], sort_col='trade_date')
            print(f"{len(df)} new rows (total {n})")
        elif df is not None:
            print("no data (empty response)")
        else:
            print("error — skipped (see log above)")


# ─── 3. index_classify ────────────────────────────────────────────────────────

def fetch_index_classify(force=False):
    fp = INDEX_DIR / 'index_classify.csv'
    if not force and _is_fresh(fp):
        print(f"\n[index_classify] Fresh — skip.")
        return

    pro = _get_pro()
    print(f"\n[index_classify] Downloading ...")

    frames = []
    for src in ('SW2021', 'SW', 'CITICS', 'CNI', 'MSCI'):
        for level in ('L1', 'L2', 'L3'):
            df = _fetch(pro.index_classify, level=level, src=src)
            if df is not None and not df.empty:
                df['_level'] = level
                df['_src'] = src
                frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        _save(combined, fp)
        print(f"  {len(combined)} records → {fp}")
    else:
        print("  No data retrieved")


# ─── 4. index_member_all ──────────────────────────────────────────────────────

def fetch_index_member_all(force=False):
    out_dir = INDEX_DIR / 'index_member_all'
    out_dir.mkdir(parents=True, exist_ok=True)
    pro = _get_pro()

    print(f"\n[index_member_all] {len(CHINESE_INDICES)} indices ...")
    for code in CHINESE_INDICES:
        fp = out_dir / f"{_safe_code(code)}.csv"
        if not force and _is_fresh(fp):
            print(f"  {code}: fresh — skip")
            continue

        print(f"  {code}: fetching ...", end=' ', flush=True)
        df = _fetch(pro.index_member_all, index_code=code)
        if df is not None and not df.empty:
            _save(df, fp)
            print(f"{len(df)} rows")
        else:
            print("no data / error")


# ─── 5. index_global ──────────────────────────────────────────────────────────

def fetch_index_global(force=False):
    out_dir = INDEX_DIR / 'index_global'
    out_dir.mkdir(parents=True, exist_ok=True)
    pro   = _get_pro()
    today = _today()

    print(f"\n[index_global] {len(GLOBAL_INDICES)} global indices ...")
    for code in GLOBAL_INDICES:
        fp   = out_dir / f"{code}.csv"
        last = _last_date(fp, 'trade_date')
        start = _next_day(last) if (last and not force) else START_DATE

        if start > today:
            print(f"  {code}: up-to-date (last={last})")
            continue

        print(f"  {code}: {start} → {today} ...", end=' ', flush=True)
        df = _fetch(pro.index_global, ts_code=code, start_date=start, end_date=today)
        if df is not None and not df.empty:
            n = _upsert(df, fp, key_cols=['ts_code', 'trade_date'], sort_col='trade_date')
            print(f"{len(df)} new rows (total {n})")
        elif df is not None:
            print("no data (empty response)")
        else:
            print("error — skipped (see log above)")


# ─── 6. idx_factor_pro ────────────────────────────────────────────────────────

def fetch_idx_factor_pro(force=False):
    out_dir = INDEX_DIR / 'idx_factor_pro'
    out_dir.mkdir(parents=True, exist_ok=True)
    pro   = _get_pro()
    today = _today()

    print(f"\n[idx_factor_pro] {len(CHINESE_INDICES)} indices ...")
    for code in CHINESE_INDICES:
        fp   = out_dir / f"{_safe_code(code)}.csv"
        last = _last_date(fp, 'trade_date')
        start = _next_day(last) if (last and not force) else START_DATE

        if start > today:
            print(f"  {code}: up-to-date (last={last})")
            continue

        print(f"  {code}: {start} → {today} ...", end=' ', flush=True)
        df = _fetch(pro.idx_factor_pro, ts_code=code, start_date=start, end_date=today)
        if df is not None and not df.empty:
            n = _upsert(df, fp, key_cols=['ts_code', 'trade_date'], sort_col='trade_date')
            print(f"{len(df)} new rows (total {n})")
        elif df is not None:
            print("no data (empty response)")
        else:
            print("error — skipped (see log above)")


# ─── 7. fund_basic ────────────────────────────────────────────────────────────

def fetch_fund_basic(force=False):
    fp = FUND_DIR / 'fund_basic.csv'
    if not force and _is_fresh(fp):
        print(f"\n[fund_basic] Fresh — skip.")
        return

    pro = _get_pro()
    print(f"\n[fund_basic] Downloading ...")
    frames = []
    for market in ('E', 'O'):
        for status in ('L', 'DE', 'P'):
            df = _fetch(pro.fund_basic, market=market, status=status)
            if df is not None and not df.empty:
                frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['ts_code'], keep='last')
        _save(combined, fp)
        print(f"  {len(combined)} funds → {fp}")
    else:
        print("  No data retrieved")


# ─── 8. fund_company ──────────────────────────────────────────────────────────

def fetch_fund_company(force=False):
    fp = FUND_DIR / 'fund_company.csv'
    if not force and _is_fresh(fp):
        print(f"\n[fund_company] Fresh — skip.")
        return

    pro = _get_pro()
    print(f"\n[fund_company] Downloading ...")
    frames = []
    for market in ('E', 'O'):
        df = _fetch(pro.fund_company, market=market)
        if df is not None and not df.empty:
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        _save(combined, fp)
        print(f"  {len(combined)} companies → {fp}")
    else:
        print("  No data retrieved")


# ─── Concurrent per-fund worker ───────────────────────────────────────────────

def _run_fund_workers(
    name:        str,
    fund_codes:  list,
    worker_fn,          # callable(code) → 'ok' | 'skip' | 'fail'
    ckpt_path=None,
    report_every: int = 500,
):
    """
    Execute worker_fn(code) for each fund code using a thread pool.

    worker_fn must be thread-safe (each code writes to its own file).
    Checkpoint saving (if ckpt_path given) is protected by a lock.
    """
    import time as _time
    total    = len(fund_codes)
    counters = {'ok': 0, 'fail': 0, 'skip': 0, 'done': 0}
    lock     = threading.Lock()
    ckpt     = _load_ckpt(ckpt_path) if ckpt_path else None
    t0       = _time.monotonic()

    def _wrapped(code):
        result = worker_fn(code)
        with lock:
            counters['done'] += 1
            if result == 'ok':
                counters['ok'] += 1
                if ckpt is not None:
                    ckpt['completed'].append(code)
            elif result == 'skip':
                counters['skip'] += 1
            else:
                counters['fail'] += 1
                if ckpt is not None:
                    ckpt['failed'].append(code)
            n = counters['done']
            if n % report_every == 0:
                if ckpt_path and ckpt is not None:
                    _save_ckpt(ckpt_path, ckpt)
                elapsed = _time.monotonic() - t0
                rate    = n / elapsed if elapsed > 0 else 0
                eta     = (total - n) / rate if rate > 0 else 0
                print(f"  {n}/{total}  ok={counters['ok']} "
                      f"fail={counters['fail']} skip={counters['skip']}  "
                      f"{rate:.1f}/s  ETA {eta:.0f}s")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(_wrapped, fund_codes))

    if ckpt_path and ckpt is not None:
        _save_ckpt(ckpt_path, ckpt)
    elapsed = _time.monotonic() - t0
    print(f"  Done in {elapsed:.0f}s — "
          f"ok={counters['ok']} fail={counters['fail']} skip={counters['skip']}")


# ─── 9. fund_manager ──────────────────────────────────────────────────────────

def fetch_fund_manager(force=False, batch=None):
    out_dir = FUND_DIR / 'fund_manager'
    out_dir.mkdir(parents=True, exist_ok=True)
    fund_codes = _get_fund_codes()
    if not fund_codes:
        print(f"\n[fund_manager] fund_basic.csv not found — run fund_basic first.")
        return
    if batch:
        fund_codes = fund_codes[:batch]

    pro        = _get_pro()
    ckpt_path  = out_dir / '_checkpoint.json'
    completed  = set(_load_ckpt(ckpt_path)['completed'])
    print(f"\n[fund_manager] {len(fund_codes)} funds  (workers={WORKERS}) ...")

    def _worker(code):
        fp = out_dir / f"{_safe_code(code)}.csv"
        if code in completed and not force and _is_fresh(fp):
            return 'skip'
        df = _fetch(pro.fund_manager, ts_code=code)
        if df is not None and not df.empty:
            _save(df, fp)
            return 'ok'
        return 'fail'

    _run_fund_workers('fund_manager', fund_codes, _worker, ckpt_path)


# ─── 10. fund_share ───────────────────────────────────────────────────────────

def fetch_fund_share(force=False, batch=None):
    out_dir      = FUND_DIR / 'fund_share'
    out_dir.mkdir(parents=True, exist_ok=True)
    no_data_path = out_dir / '_no_data.json'
    no_data      = _load_no_data(no_data_path) if not force else set()

    fund_codes = _get_fund_codes()
    if not fund_codes:
        print(f"\n[fund_share] fund_basic.csv not found — run fund_basic first.")
        return
    if batch:
        fund_codes = fund_codes[:batch]

    # First-run bootstrap: build _no_data.json from existing files so we don't
    # waste API calls on funds that previous runs already found to be empty.
    if not no_data and not force:
        no_data = _auto_init_no_data(fund_codes, out_dir, no_data_path)

    todo = [c for c in fund_codes if c not in no_data]
    print(f"\n[fund_share] {len(fund_codes)} funds  (workers={WORKERS}) ...")
    if no_data:
        print(f"  Skipping {len(fund_codes) - len(todo)} known-empty funds "
              f"(pass force=True to re-check) ...")
    print(f"  Pre-scanning existing dates ...")
    last_dates = _pre_scan_dates(todo, out_dir, 'trade_date')

    pro          = _get_pro()
    today        = _today()
    # Use yesterday as the fetch ceiling — today's data isn't published yet,
    # and it lets us skip funds already current without making an API call.
    yesterday    = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    no_data_lock = threading.Lock()
    no_data_new  = set()

    def _worker(code):
        fp    = out_dir / f"{_safe_code(code)}.csv"
        last  = last_dates.get(code)
        start = _next_day(last) if (last and not force) else START_DATE
        # Skip without API call if already current through yesterday
        if start > yesterday:
            return 'skip'
        df = _fetch(pro.fund_share, ts_code=code, start_date=start, end_date=today)
        if df is not None and not df.empty:
            _upsert(df, fp, key_cols=['ts_code', 'trade_date'], sort_col='trade_date')
            return 'ok'
        if df is not None:          # empty DataFrame
            if not fp.exists():     # Only permanent no_data if fund never had any file
                with no_data_lock:
                    no_data_new.add(code)
            return 'skip'
        return 'fail'               # None = API error, retry next run

    _run_fund_workers('fund_share', todo, _worker)
    _save_no_data(no_data_path, no_data | no_data_new)


# ─── 11. fund_nav ─────────────────────────────────────────────────────────────

def fetch_fund_nav(force=False, batch=None):
    out_dir      = FUND_DIR / 'fund_nav'
    out_dir.mkdir(parents=True, exist_ok=True)
    no_data_path = out_dir / '_no_data.json'
    no_data      = _load_no_data(no_data_path) if not force else set()

    fund_codes = _get_fund_codes()
    if not fund_codes:
        print(f"\n[fund_nav] fund_basic.csv not found — run fund_basic first.")
        return
    if batch:
        fund_codes = fund_codes[:batch]

    if not no_data and not force:
        no_data = _auto_init_no_data(fund_codes, out_dir, no_data_path)

    todo = [c for c in fund_codes if c not in no_data]
    print(f"\n[fund_nav] {len(fund_codes)} funds  (workers={WORKERS}) ...")
    if no_data:
        print(f"  Skipping {len(fund_codes) - len(todo)} known-empty funds "
              f"(pass force=True to re-check) ...")
    print(f"  Pre-scanning existing dates ...")
    last_dates = _pre_scan_dates(todo, out_dir, 'nav_date')

    pro          = _get_pro()
    today        = _today()
    yesterday    = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    today_year   = int(today[:4])
    no_data_lock = threading.Lock()
    no_data_new  = set()

    def _worker(code):
        fp         = out_dir / f"{_safe_code(code)}.csv"
        last       = last_dates.get(code)
        start      = _next_day(last) if (last and not force) else START_DATE
        start_year = int(start[:4])
        if start > yesterday:
            return 'skip'

        # Collect all year-chunks into one DataFrame, then do a single _upsert.
        frames     = []
        had_error  = False
        for year in range(start_year, today_year + 1):
            yr_start = max(start, f'{year}0101')
            yr_end   = min(today, f'{year}1231')
            if yr_start > yr_end:
                continue
            df = _fetch(pro.fund_nav, ts_code=code, start_date=yr_start, end_date=yr_end)
            if df is not None and not df.empty:
                frames.append(df)
            elif df is None:
                had_error = True    # permission / hard error

        if frames:
            new_data = pd.concat(frames, ignore_index=True)
            _upsert(new_data, fp, key_cols=['ts_code', 'nav_date'], sort_col='nav_date')
            return 'ok'
        if had_error:
            return 'fail'           # real error — retry next run
        if not fp.exists():         # Only permanent no_data if fund never had any file
            with no_data_lock:
                no_data_new.add(code)
        return 'skip'

    _run_fund_workers('fund_nav', todo, _worker)
    _save_no_data(no_data_path, no_data | no_data_new)


# ─── 12. fund_div ─────────────────────────────────────────────────────────────

def fetch_fund_div(force=False, batch=None):
    out_dir = FUND_DIR / 'fund_div'
    out_dir.mkdir(parents=True, exist_ok=True)
    fund_codes = _get_fund_codes()
    if not fund_codes:
        print(f"\n[fund_div] fund_basic.csv not found — run fund_basic first.")
        return
    if batch:
        fund_codes = fund_codes[:batch]

    pro       = _get_pro()
    ckpt_path = out_dir / '_checkpoint.json'
    completed = set(_load_ckpt(ckpt_path)['completed'])
    print(f"\n[fund_div] {len(fund_codes)} funds  (workers={WORKERS}) ...")

    def _worker(code):
        fp = out_dir / f"{_safe_code(code)}.csv"
        if code in completed and not force and _is_fresh(fp):
            return 'skip'
        df = _fetch(pro.fund_div, ts_code=code)
        if df is not None:
            if not df.empty:
                _save(df, fp)
            return 'ok'   # empty result is valid (fund never distributed)
        return 'fail'

    _run_fund_workers('fund_div', fund_codes, _worker, ckpt_path)


# ─── 13. fund_portfolio ───────────────────────────────────────────────────────

def fetch_fund_portfolio(force=False, batch=None):
    out_dir = FUND_DIR / 'fund_portfolio'
    out_dir.mkdir(parents=True, exist_ok=True)
    fund_codes = _get_fund_codes()
    if not fund_codes:
        print(f"\n[fund_portfolio] fund_basic.csv not found — run fund_basic first.")
        return
    if batch:
        fund_codes = fund_codes[:batch]

    pro       = _get_pro()
    ckpt_path = out_dir / '_checkpoint.json'
    completed = set(_load_ckpt(ckpt_path)['completed'])
    print(f"\n[fund_portfolio] {len(fund_codes)} funds  (workers={WORKERS}) ...")

    def _worker(code):
        fp = out_dir / f"{_safe_code(code)}.csv"
        if code in completed and not force and _is_fresh(fp):
            return 'skip'
        df = _fetch(pro.fund_portfolio, ts_code=code)
        if df is not None:
            if not df.empty:
                _save(df, fp)
            return 'ok'
        return 'fail'

    _run_fund_workers('fund_portfolio', fund_codes, _worker, ckpt_path)


# ─── 14. fund_factor_pro ──────────────────────────────────────────────────────

def fetch_fund_factor_pro(force=False, batch=None):
    out_dir      = FUND_DIR / 'fund_factor_pro'
    out_dir.mkdir(parents=True, exist_ok=True)
    no_data_path = out_dir / '_no_data.json'
    no_data      = _load_no_data(no_data_path) if not force else set()

    fund_codes = _get_fund_codes()
    if not fund_codes:
        print(f"\n[fund_factor_pro] fund_basic.csv not found — run fund_basic first.")
        return
    if batch:
        fund_codes = fund_codes[:batch]

    if not no_data and not force:
        no_data = _auto_init_no_data(fund_codes, out_dir, no_data_path)

    todo = [c for c in fund_codes if c not in no_data]
    print(f"\n[fund_factor_pro] {len(fund_codes)} funds  (workers={WORKERS}) ...")
    if no_data:
        print(f"  Skipping {len(fund_codes) - len(todo)} known-empty funds "
              f"(pass force=True to re-check) ...")
    print(f"  Pre-scanning existing dates ...")
    last_dates = _pre_scan_dates(todo, out_dir, 'trade_date')

    pro          = _get_pro()
    today        = _today()
    yesterday    = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    no_data_lock = threading.Lock()
    no_data_new  = set()

    def _worker(code):
        fp    = out_dir / f"{_safe_code(code)}.csv"
        last  = last_dates.get(code)
        start = _next_day(last) if (last and not force) else START_DATE
        if start > yesterday:
            return 'skip'
        df = _fetch(pro.fund_factor_pro, ts_code=code, start_date=start, end_date=today)
        if df is not None and not df.empty:
            _upsert(df, fp, key_cols=['ts_code', 'trade_date'], sort_col='trade_date')
            return 'ok'
        if df is not None:          # empty DataFrame
            if not fp.exists():     # Only permanent no_data if fund never had any file
                with no_data_lock:
                    no_data_new.add(code)
            return 'skip'
        return 'fail'               # None = API error, retry next run

    _run_fund_workers('fund_factor_pro', todo, _worker)
    _save_no_data(no_data_path, no_data | no_data_new)


# ─── Ordered dataset registry ─────────────────────────────────────────────────

DATASETS = [
    ('index_weight',     fetch_index_weight),
    ('index_dailybasic', fetch_index_dailybasic),
    ('index_classify',   fetch_index_classify),
    ('index_member_all', fetch_index_member_all),
    ('index_global',     fetch_index_global),
    ('idx_factor_pro',   fetch_idx_factor_pro),
    ('fund_basic',       fetch_fund_basic),
    ('fund_company',     fetch_fund_company),
    ('fund_manager',     fetch_fund_manager),
    ('fund_share',       fetch_fund_share),
    ('fund_nav',         fetch_fund_nav),
    ('fund_div',         fetch_fund_div),
    ('fund_portfolio',   fetch_fund_portfolio),
    ('fund_factor_pro',  fetch_fund_factor_pro),
]

_DATASET_MAP = {name: fn for name, fn in DATASETS}


# ─── Public entry point ───────────────────────────────────────────────────────

def run(dataset=None, force=False, batch=None):
    """
    Download Tushare Pro index and fund data.

    Args:
        dataset (str | None):
            Name of a specific dataset to run, or None to run all.
        force (bool):
            Force re-download of static / fresh files (default False).
        batch (int | None):
            Maximum number of funds to process (fund endpoints only).

    Examples:
        from api.get_data import run

        run()                        # All datasets in order
        run('index_weight')          # Single dataset
        run('fund_nav', batch=200)   # Fund NAV, 200 funds at a time
        run(force=True)              # Force-refresh all static data
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    FUND_DIR.mkdir(parents=True, exist_ok=True)

    today = _today()
    print(f"\n{'='*60}")
    print(f"Tushare Data Acquisition  [{today}]  workers={WORKERS}  rate={CALLS_PER_SEC}/s")
    print(f"{'='*60}")

    if dataset:
        if dataset not in _DATASET_MAP:
            names = ', '.join(_DATASET_MAP)
            print(f"Unknown dataset '{dataset}'.  Available: {names}")
            return
        to_run = [(dataset, _DATASET_MAP[dataset])]
    else:
        to_run = DATASETS

    for name, fn in to_run:
        sig    = inspect.signature(fn)
        kwargs = {'force': force}
        if 'batch' in sig.parameters:
            kwargs['batch'] = batch
        try:
            fn(**kwargs)
        except Exception as e:
            print(f"\n[{name}] Unexpected error: {e}")

    print(f"\n{'='*60}")
    print(f"All done  [{_today()}]")
    print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Tushare Pro index & fund data (incremental).'
    )
    parser.add_argument('--dataset', default=None,
                        help='Specific dataset name (default: all in order)')
    parser.add_argument('--force', action='store_true',
                        help='Force refresh even for fresh/static files')
    parser.add_argument('--batch', type=int, default=None,
                        help='Max funds to process per run (fund endpoints only)')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'Override WORKERS (default {WORKERS})')
    parser.add_argument('--rate', type=float, default=None,
                        help=f'Override CALLS_PER_SEC (default {CALLS_PER_SEC})')
    args, _ = parser.parse_known_args()

    if args.workers is not None:
        WORKERS = args.workers
    if args.rate is not None:
        CALLS_PER_SEC = args.rate
        _limiter = _RateLimiter(CALLS_PER_SEC)

    run(dataset=args.dataset, force=args.force, batch=args.batch)
