"""
Financial Extras Data Acquisition from Tushare Pro
- forecast: Earnings forecast
- express: Express earnings report (preliminary quick report)
- dividend: Dividend distribution
- fina_audit: Audit opinions
- fina_mainbz: Main business composition
- disclosure_date: Financial report disclosure dates

Storage:
  - Per-stock data: stock_data/fina_extras/{data_type}/{ts_code}.csv
  - disclosure_date is per-stock annual schedule

Usage:
    from api.fina_extras import run

    run()                              # update all types
    run('update')                      # same as above
    run('forecast')                    # update forecast only
    run('dividend')                    # update dividend only
    run('status')                      # show coverage summary
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN   = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR        = Path('./stock_data/fina_extras')
STOCK_LIST_FILE = Path('./stock_data/stock_list.csv')
START_DATE      = '20170101'

# Data types and their configurations
DATA_TYPES = {
    'forecast': {
        'method': 'forecast',
        'dedup_cols': ['ts_code', 'ann_date', 'end_date'],
        'date_col': 'ann_date',
        'fields': [
            'ts_code', 'ann_date', 'end_date', 'type', 'p_change_min', 'p_change_max',
            'net_profit_min', 'net_profit_max', 'last_parent_net', 'first_ann_date',
            'summary', 'change_reason',
        ],
    },
    'express': {
        'method': 'express',
        'dedup_cols': ['ts_code', 'ann_date', 'end_date'],
        'date_col': 'ann_date',
        'fields': [
            'ts_code', 'ann_date', 'end_date', 'revenue', 'operate_profit', 'total_profit',
            'n_income', 'total_assets', 'total_hldr_eqy_exc_min_int', 'diluted_eps',
            'diluted_roe', 'yoy_net_profit', 'bps', 'yoy_sales', 'yoy_op', 'yoy_tp',
            'yoy_dedu_np', 'yoy_eps', 'yoy_roe', 'growth_assets', 'yoy_equity',
            'growth_bps', 'or_last_year', 'op_last_year', 'tp_last_year', 'np_last_year',
            'eps_last_year', 'open_net_assets', 'open_bps', 'perf_summary', 'is_audit',
            'remark',
        ],
    },
    'dividend': {
        'method': 'dividend',
        'dedup_cols': ['ts_code', 'end_date', 'div_proc'],
        'date_col': 'ann_date',
        'fields': [
            'ts_code', 'end_date', 'ann_date', 'div_proc', 'stk_div', 'stk_bo_rate',
            'stk_co_rate', 'cash_div', 'cash_div_tax', 'record_date', 'ex_date',
            'pay_date', 'div_listdate', 'imp_ann_date', 'base_date', 'base_share',
        ],
    },
    'fina_audit': {
        'method': 'fina_audit',
        'dedup_cols': ['ts_code', 'ann_date', 'end_date'],
        'date_col': 'ann_date',
        'fields': [
            'ts_code', 'ann_date', 'end_date', 'audit_result', 'audit_fees',
            'audit_agency', 'audit_sign',
        ],
    },
    'fina_mainbz': {
        'method': 'fina_mainbz',
        'dedup_cols': ['ts_code', 'end_date', 'bz_item'],
        'date_col': 'end_date',
        'fields': [
            'ts_code', 'end_date', 'bz_item', 'bz_sales', 'bz_profit', 'bz_cost',
            'curr_type', 'update_flag',
        ],
    },
    'disclosure_date': {
        'method': 'disclosure_date',
        'dedup_cols': ['ts_code', 'end_date'],
        'date_col': 'ann_date',
        'fields': [
            'ts_code', 'ann_date', 'end_date', 'pre_date', 'actual_date', 'modify_date',
        ],
    },
}

WORKERS       = 4
CALLS_PER_SEC = 2.0
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
                print(f"    [permission denied] {err[:100]}")
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

def _setup(data_type: str):
    (DATA_DIR / data_type).mkdir(parents=True, exist_ok=True)


def _last_date(data_type: str, ts_code: str, col: str) -> Optional[str]:
    fp = DATA_DIR / data_type / f"{ts_code.replace('.', '_')}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, usecols=[col])
        if df.empty:
            return None
        return str(int(float(df[col].max())))
    except Exception:
        return None


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


def _download_one(data_type: str, ts_code: str) -> tuple:
    """Download data for a single stock. Returns (ts_code, status)."""
    cfg      = DATA_TYPES[data_type]
    fp       = DATA_DIR / data_type / f"{ts_code.replace('.', '_')}.csv"
    last     = _last_date(data_type, ts_code, cfg['date_col'])

    pro = _get_pro()
    method = getattr(pro, cfg['method'])

    # Call API
    kwargs = {'ts_code': ts_code}
    if last and data_type not in ('disclosure_date',):
        kwargs['start_date'] = last

    df = _fetch(method, **kwargs)

    if df is None:
        return ts_code, 'error'
    if df.empty:
        return ts_code, 'no_data'

    # Filter to available columns
    available_cols = [c for c in cfg['fields'] if c in df.columns]
    df = df[available_cols].copy()

    if fp.exists() and last is not None:
        existing = pd.read_csv(fp)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=cfg['dedup_cols'], keep='last')

    df = df.sort_values(cfg['date_col'] if cfg['date_col'] in df.columns else df.columns[0])
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    return ts_code, f'+{len(df)} rows'


# ─── Update functions ─────────────────────────────────────────────────────────

def update_data_type(data_type: str, batch: Optional[int] = None, retry_only: bool = False):
    """Update a single data type for all stocks."""
    _setup(data_type)
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

    ckpt_lock  = threading.Lock()
    counters   = {'ok': 0, 'err': 0, 'done': 0}
    total      = len(todo)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_one, data_type, code): code for code in todo}
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
                if counters['done'] % 50 == 0:
                    _save_ckpt(data_type, completed, failed)

            if counters['done'] % 100 == 0 or is_err:
                print(f"  [{counters['done']}/{total}] {code}: {st}")

    _save_ckpt(data_type, completed, failed)
    print(f"\n[{data_type}] Done. ok={counters['ok']} errors={counters['err']}")


def update_all(batch: Optional[int] = None):
    """Update all data types for all stocks."""
    for data_type in DATA_TYPES:
        update_data_type(data_type, batch=batch)


def status():
    """Print coverage summary for all data types."""
    for data_type in DATA_TYPES:
        data_dir = DATA_DIR / data_type
        if not data_dir.exists():
            print(f"[{data_type}] Not downloaded yet")
            continue
        files = list(data_dir.glob('[!_]*.csv'))
        completed, failed = _load_ckpt(data_type)
        print(f"[{data_type}] {len(files)} stock files")
        print(f"  Checkpoint: {len(completed)} completed, {len(failed)} failed")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(action: str = 'update', **kwargs):
    """
    Entry point for financial extras operations.

    Actions:
        'update'          - update all data types
        'forecast'        - earnings forecast
        'express'         - express earnings
        'dividend'        - dividend distribution
        'fina_audit'      - audit opinions
        'fina_mainbz'     - main business composition
        'disclosure_date' - disclosure schedule
        'retry'           - retry failed stocks
        'status'          - show coverage summary

    Keyword args:
        ts_code  (str) - stock code for single stock
        batch    (int) - max stocks to process
    """
    if action == 'update':
        update_all(batch=kwargs.get('batch'))

    elif action in DATA_TYPES:
        update_data_type(action, batch=kwargs.get('batch'))

    elif action == 'retry':
        for data_type in DATA_TYPES:
            update_data_type(data_type, batch=kwargs.get('batch'), retry_only=True)

    elif action == 'stock':
        ts_code = kwargs.get('ts_code')
        if not ts_code:
            print("Error: ts_code required")
        else:
            for data_type in DATA_TYPES:
                _setup(data_type)
                code, st = _download_one(data_type, ts_code)
                print(f"[{data_type}] {code}: {st}")

    elif action == 'status':
        status()

    else:
        print(f"Unknown action: {action!r}")


if __name__ == '__main__':
    run('status')
