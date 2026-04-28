"""
Chip Distribution Data Acquisition from Tushare Pro
- cyq_perf: Chip distribution performance (筹码分布绩效)
- cyq_chips: Chip distribution data (筹码分布数据)

Note: These APIs require higher Tushare access levels (typically 5000+ points).

Storage:
  - cyq_perf: stock_data/chip_data/cyq_perf/{ts_code}.csv
  - cyq_chips: stock_data/chip_data/cyq_chips/{ts_code}.csv

Usage:
    from api.chip_data import run

    run()                    # download all chip data
    run('cyq_perf')          # download performance only
    run('cyq_chips')         # download chips only
    run('status')            # show coverage summary
"""

import json
import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN   = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR        = Path('./stock_data/chip_data')
STOCK_LIST_FILE = Path('./stock_data/stock_list.csv')
START_DATE      = '20200101'  # Chip data often has limited history

# Data types configuration
DATA_TYPES = {
    'cyq_perf': {
        'method': 'cyq_perf',
        'date_col': 'trade_date',
        'fields': [
            'ts_code', 'trade_date', 'his_low', 'his_high', 'cost_5pct',
            'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct',
            'weight_avg', 'winner_rate',
        ],
    },
    'cyq_chips': {
        'method': 'cyq_chips',
        'date_col': 'trade_date',
        'fields': [
            'ts_code', 'trade_date', 'price', 'percent',
        ],
    },
}

WORKERS       = 4
CALLS_PER_SEC = 2.0  # Chip APIs may have stricter limits
MAX_RETRIES   = 3
RETRY_DELAY   = 2


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
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _load_ckpt(data_type: str):
    ckpt_file = DATA_DIR / data_type / '_checkpoint.json'
    if ckpt_file.exists():
        try:
            data = json.loads(ckpt_file.read_text(encoding='utf-8'))
            return set(data.get('completed', [])), set(data.get('failed', []))
        except Exception:
            pass
    return set(), set()


def _save_ckpt(data_type: str, completed, failed):
    ckpt_file = DATA_DIR / data_type / '_checkpoint.json'
    ckpt_file.write_text(
        json.dumps({'completed': sorted(completed), 'failed': sorted(failed)}, indent=2),
        encoding='utf-8'
    )


# ─── Per-stock helpers ────────────────────────────────────────────────────────

def _load_stock_list() -> List[str]:
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
    codes = []
    for d, suffix in [('./stock_data/sh', 'SH'), ('./stock_data/sz', 'SZ')]:
        p = Path(d)
        if p.exists():
            codes += [f.stem + '.' + suffix for f in p.glob('*.csv')]
    return codes


def _last_date(data_type: str, ts_code: str) -> Optional[str]:
    fp = DATA_DIR / data_type / f"{ts_code.replace('.', '_')}.csv"
    if not fp.exists():
        return None
    try:
        cfg = DATA_TYPES[data_type]
        df = pd.read_csv(fp, usecols=[cfg['date_col']])
        if df.empty:
            return None
        return str(int(float(df[cfg['date_col']].max())))
    except Exception:
        return None


def _download_one(data_type: str, ts_code: str) -> tuple:
    """Download chip data for a single stock. Returns (ts_code, status)."""
    cfg = DATA_TYPES[data_type]
    out_dir = DATA_DIR / data_type
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{ts_code.replace('.', '_')}.csv"

    last = _last_date(data_type, ts_code)
    start = last if last else START_DATE
    end = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    pro = _get_pro()
    method = getattr(pro, cfg['method'], None)
    if method is None:
        return ts_code, 'method_not_found'

    df = _fetch(method, ts_code=ts_code, start_date=start, end_date=end)

    if df is None:
        return ts_code, 'error'
    if df.empty:
        return ts_code, 'no_data'

    # Filter to available columns
    available = [c for c in cfg['fields'] if c in df.columns]
    df = df[available].copy()

    if fp.exists() and last:
        existing = pd.read_csv(fp)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=['ts_code', cfg['date_col']], keep='last')

    # Normalise date column to str — existing CSVs have it as int64 (pandas
    # inferred), tushare returns str. After concat the column is `object`
    # dtype with mixed int + str values, which breaks sort_values's `<`
    # comparison. Cast unconditionally.
    if cfg['date_col'] in df.columns:
        df[cfg['date_col']] = df[cfg['date_col']].astype(str)
    df = df.sort_values(cfg['date_col'])
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    return ts_code, f'+{len(df)} rows'


# ─── Update functions ─────────────────────────────────────────────────────────

def update_data_type(data_type: str, batch: Optional[int] = None, retry_only: bool = False):
    """Update a single data type for all stocks."""
    if data_type not in DATA_TYPES:
        print(f"Unknown data type: {data_type}")
        return

    (DATA_DIR / data_type).mkdir(parents=True, exist_ok=True)
    all_stocks = _load_stock_list()
    completed, failed = _load_ckpt(data_type)

    if retry_only:
        todo = [c for c in all_stocks if c in failed]
        print(f"[{data_type}] Retrying {len(todo)} previously-failed stocks ...")
    else:
        todo = all_stocks
        print(f"[{data_type}] {len(all_stocks)} total stocks ...")

    if batch:
        todo = todo[:batch]
        print(f"  (batch limit: processing first {len(todo)})")

    if not todo:
        print("Nothing to do.")
        return

    ckpt_lock = threading.Lock()
    counters = {'ok': 0, 'err': 0, 'done': 0}
    total = len(todo)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_one, data_type, code): code for code in todo}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                _, st = fut.result()
                is_err = st in ('error', 'method_not_found')
            except Exception as exc:
                st = f'exception: {exc}'
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
                if counters['done'] % 50 == 0:
                    _save_ckpt(data_type, completed, failed)

            if counters['done'] % 100 == 0 or is_err:
                print(f"  [{counters['done']}/{total}] {code}: {st}")

    _save_ckpt(data_type, completed, failed)
    print(f"\n[{data_type}] Done. ok={counters['ok']} errors={counters['err']}")


def update_all(batch: Optional[int] = None):
    """Update all chip data types for all stocks."""
    for data_type in DATA_TYPES:
        update_data_type(data_type, batch=batch)


def status():
    """Print coverage summary."""
    print("\n[Chip Data Status]")
    for data_type in DATA_TYPES:
        out_dir = DATA_DIR / data_type
        if not out_dir.exists():
            print(f"  {data_type}: Not downloaded")
            continue
        files = list(out_dir.glob('[!_]*.csv'))
        completed, failed = _load_ckpt(data_type)
        print(f"  {data_type}: {len(files)} stock files")
        print(f"    Checkpoint: {len(completed)} completed, {len(failed)} failed")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(action: str = 'update', **kwargs):
    """
    Entry point for chip data operations.

    Actions:
        'update'    - update all chip data types
        'cyq_perf'  - chip distribution performance
        'cyq_chips' - chip distribution data
        'retry'     - retry failed stocks
        'status'    - show coverage summary

    Keyword args:
        batch (int) - max stocks to process
    """
    if action == 'update':
        update_all(batch=kwargs.get('batch'))

    elif action in DATA_TYPES:
        update_data_type(action, batch=kwargs.get('batch'))

    elif action == 'retry':
        for data_type in DATA_TYPES:
            update_data_type(data_type, batch=kwargs.get('batch'), retry_only=True)

    elif action == 'status':
        status()

    else:
        print(f"Unknown action: {action}")


if __name__ == '__main__':
    run('status')
