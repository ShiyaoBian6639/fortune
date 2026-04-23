"""
Financial Indicator Data Acquisition from Tushare Pro
- Fetches quarterly financial ratios (ROE, profit margin, debt ratio, etc.)
- Saves one file per stock: stock_data/fina_indicator/{ts_code}.csv
- Incremental at data level: only fetches announcements newer than last ann_date
- Incremental at job level: checkpoint tracks completed/failed stocks so reruns
  skip already-processed stocks and only retry failures

Usage:
    from api.fina_indicator import run

    run()                          # update all stocks (skips already done)
    run('update')                  # same as above
    run('retry')                   # retry only failed stocks from last run
    run('update', batch=500)       # process up to 500 remaining stocks
    run('stock', ts_code='000001.SZ')  # single stock
    run('status')                  # show coverage + checkpoint summary
    run('reset')                   # clear checkpoint to reprocess everything
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN   = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR        = Path('./stock_data/fina_indicator')
CKPT_FILE       = DATA_DIR / '_checkpoint.json'
STOCK_LIST_FILE = Path('./stock_data/stock_list.csv')
START_DATE      = '20170101'

# Financial-statement APIs have per-minute quotas. For 8000-pt accounts,
# 12 workers × 6 calls/s keeps fina_indicator inside its ~500/min bucket
# (each call fetches one stock's full quarterly history).
WORKERS       = 12    # WORKERS ≥ CALLS_PER_SEC × avg_latency_s (~1.5s → need ≥9)
CALLS_PER_SEC = 6.0   # raised from 4.0 for 8000-pt tier
MAX_RETRIES   = 3
RETRY_DELAY   = 2

FIELDS = [
    'ts_code', 'ann_date', 'end_date', 'eps', 'eps_yoy',
    'roe', 'roe_yoy', 'roa', 'roa_yoy',
    'grossprofit_margin', 'netprofit_margin',
    'current_ratio', 'quick_ratio', 'debt_to_assets',
    'revenue_yoy', 'profit_yoy',
    'assets_yoy', 'equity_yoy',
    'op_yoy', 'ebt_yoy',
]

# ─── Rate limiter ─────────────────────────────────────────────────────────────

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
                print(f"    [permission denied] {err[:100]}")
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _load_ckpt():
    if CKPT_FILE.exists():
        try:
            data = json.loads(CKPT_FILE.read_text(encoding='utf-8'))
            return set(data.get('completed', [])), set(data.get('failed', []))
        except Exception:
            pass
    return set(), set()


def _save_ckpt(completed, failed):
    CKPT_FILE.write_text(
        json.dumps({'completed': sorted(completed), 'failed': sorted(failed)},
                   indent=2),
        encoding='utf-8'
    )


# ─── Per-stock helpers ────────────────────────────────────────────────────────

def _setup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _last_ann_date(ts_code):
    fp = DATA_DIR / f"{ts_code.replace('.', '_')}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, usecols=['ann_date'])
        if df.empty:
            return None
        return str(int(float(df['ann_date'].max())))
    except Exception:
        return None


def _load_stock_list():
    if STOCK_LIST_FILE.exists():
        df = pd.read_csv(STOCK_LIST_FILE)
        for col in ('ts_code', 'code', 'symbol'):
            if col in df.columns:
                codes = df[col].astype(str).tolist()
                result = []
                for c in codes:
                    c = c.strip()
                    if '.' not in c:
                        c = c + ('.SH' if c.startswith('6') else '.SZ')
                    result.append(c)
                return result
    # Fallback: scan sh/ and sz/ directories
    codes = []
    for d, suffix in [('./stock_data/sh', 'SH'), ('./stock_data/sz', 'SZ')]:
        p = Path(d)
        if p.exists():
            codes += [f.stem + '.' + suffix for f in p.glob('*.csv')]
    return codes


def _download_one(ts_code):
    """Download fina_indicator for a single stock (incremental). Returns (ts_code, status)."""
    fp    = DATA_DIR / f"{ts_code.replace('.', '_')}.csv"
    last  = _last_ann_date(ts_code)
    start = last if last else START_DATE

    pro = _get_pro()
    df  = _fetch(
        pro.fina_indicator,
        ts_code=ts_code,
        start_date=start,
        fields=','.join(FIELDS),
    )

    if df is None:
        return ts_code, 'error'
    if df.empty:
        return ts_code, 'no_new_data'

    df = df.sort_values('ann_date')

    if fp.exists() and last is not None:
        existing = pd.read_csv(fp)
        # Normalize ann_date to string in case it was saved as int
        if 'ann_date' in existing.columns and pd.api.types.is_numeric_dtype(existing['ann_date']):
            existing['ann_date'] = existing['ann_date'].apply(
                lambda x: str(int(x)) if pd.notna(x) else x)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=['ts_code', 'ann_date', 'end_date'], keep='last')
        df = df.sort_values('ann_date')

    df.to_csv(fp, index=False, encoding='utf-8-sig')
    return ts_code, f'+{len(df)} rows'


# ─── Public interface ─────────────────────────────────────────────────────────

def update_all(batch=None, retry_only=False):
    """
    Update fina_indicator for all stocks.

    Args:
        batch:       process at most this many stocks in this session
        retry_only:  if True, only re-attempt stocks that failed in a prior run
    """
    _setup()
    all_stocks    = _load_stock_list()
    completed, failed = _load_ckpt()

    if retry_only:
        todo = [c for c in all_stocks if c in failed]
        print(f"Retrying {len(todo)} previously-failed stocks ...")
    else:
        # Always process all stocks — _last_ann_date() provides incremental
        # start_date so only new announcements are fetched.  The completed set
        # is intentionally NOT used to skip stocks here; it is only kept for
        # the retry workflow.
        todo = all_stocks
        print(f"fina_indicator: {len(all_stocks)} total (incremental per ann_date) ...")

    if batch:
        todo = todo[:batch]
        print(f"  (batch limit: processing first {len(todo)})")

    if not todo:
        print("Nothing to do.")
        return

    ckpt_lock  = threading.Lock()
    counters   = {'ok': 0, 'err': 0, 'done': 0}
    total      = len(todo)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_one, code): code for code in todo}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                _, st = fut.result()
                is_err = (st == 'error')
            except Exception as exc:
                st     = f'exception: {exc}'
                is_err = True

            with ckpt_lock:
                counters['done'] += 1
                if is_err:
                    counters['err'] += 1
                    failed.add(code)
                    completed.discard(code)
                else:
                    counters['ok'] += 1
                    completed.add(code)
                    failed.discard(code)
                # Save checkpoint every 50 completions
                if counters['done'] % 50 == 0:
                    _save_ckpt(completed, failed)

            if counters['done'] % 50 == 0 or is_err:
                print(f"  [{counters['done']}/{total}] {code}: {st}")

    _save_ckpt(completed, failed)
    print(f"\nDone. ok={counters['ok']} errors={counters['err']} "
          f"(checkpoint: {len(completed)} completed, {len(failed)} failed total)")
    if failed:
        sample = sorted(failed)[:5]
        print(f"  Run run('retry') to re-attempt {len(failed)} failed stocks. "
              f"Sample: {sample}")


def update_stock(ts_code):
    """Update a single stock."""
    _setup()
    code, st = _download_one(ts_code)
    print(f"{code}: {st}")


def status():
    """Print coverage and checkpoint summary."""
    _setup()
    files = list(DATA_DIR.glob('[!_]*.csv'))  # exclude _checkpoint.json
    completed, failed = _load_ckpt()
    print(f"fina_indicator/ has {len(files)} stock files")
    print(f"  Checkpoint: {len(completed)} completed, {len(failed)} failed")
    if files:
        for fp in sorted(files)[:3]:
            try:
                df = pd.read_csv(fp, usecols=['ann_date'])
                print(f"  {fp.stem}: {df['ann_date'].min()} → "
                      f"{df['ann_date'].max()} ({len(df)} rows)")
            except Exception:
                pass


def init_checkpoint():
    """
    Pre-populate the checkpoint from existing files on disk.
    Call this once after a legacy run (before checkpoint support was added)
    so that future run('update') only processes the stocks that are still missing.
    """
    _setup()
    all_stocks        = _load_stock_list()
    completed, failed = _load_ckpt()

    pre_existing = set()
    for code in all_stocks:
        fp = DATA_DIR / f"{code.replace('.', '_')}.csv"
        if fp.exists():
            pre_existing.add(code)

    new_completions = pre_existing - completed
    completed |= new_completions
    # Remove from failed if file now exists
    failed -= pre_existing
    _save_ckpt(completed, failed)

    missing = [c for c in all_stocks if c not in completed]
    print(f"Checkpoint initialised from disk:")
    print(f"  {len(pre_existing)} files found → marked as completed")
    print(f"  {len(missing)} stocks still need downloading")
    if missing:
        print(f"  Run run('update') to download the {len(missing)} missing stocks.")


def reset_checkpoint():
    """Clear the checkpoint so all stocks are reprocessed from scratch."""
    _save_ckpt(set(), set())
    print("Checkpoint cleared.")


def run(action='update', **kwargs):
    """
    Entry point for fina_indicator operations.

    Actions:
        'update'  – incremental update, skip already-completed stocks (default)
        'retry'   – re-attempt only stocks that failed in a prior run
        'stock'   – update a single stock
        'status'  – show coverage + checkpoint summary
        'reset'   – clear checkpoint to reprocess everything

    Keyword args:
        ts_code  (str)  – stock code for 'stock' action
        batch    (int)  – max stocks to process this session
    """
    if action in ('update', 'download'):
        update_all(batch=kwargs.get('batch'), retry_only=False)

    elif action == 'retry':
        update_all(batch=kwargs.get('batch'), retry_only=True)

    elif action == 'stock':
        ts_code = kwargs.get('ts_code')
        if not ts_code:
            print("Error: ts_code required for 'stock' action")
        else:
            update_stock(ts_code)

    elif action == 'status':
        status()

    elif action == 'reset':
        reset_checkpoint()

    elif action == 'init':
        init_checkpoint()

    else:
        print(f"Unknown action: {action!r}. Valid: update | retry | init | stock | status | reset")


if __name__ == '__main__':
    run('status')
    run('update')
