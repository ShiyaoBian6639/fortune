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

Update strategy
---------------
Tushare's fina-extras endpoints all support a period-style bulk parameter
(``period`` for forecast/express/fina_audit/fina_mainbz, ``end_date`` for
disclosure_date and dividend).  Calling them per-stock — 5201 stocks × 6
endpoints = 31,206 calls — was the source of the multi-hour update times.
The bulk path makes one call per (endpoint, period) instead, then fans the
response out to per-stock CSVs.

For 9 years × 4 quarters × 6 endpoints ≈ 216 calls vs 31,206 — roughly
145× fewer requests.

Usage:
    from api.fina_extras import run

    run()                              # bulk update everything (default)
    run('forecast')                    # bulk update one type
    run('stock', ts_code='000001.SZ')  # per-stock update for one code
    run('status')                      # coverage summary
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN   = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR        = Path('./stock_data/fina_extras')
STOCK_LIST_FILE = Path('./stock_data/stock_list.csv')
START_DATE      = '20170101'

# Per-data-type config:
#   method      : Tushare API method name on the Pro client
#   bulk_kwarg  : period-style parameter accepted by the bulk endpoint
#   dedup_cols  : keys for dedup when merging into existing per-stock CSV
#   date_col    : primary date column (used for legacy per-stock incremental)
#   fields      : kept response columns
DATA_TYPES: Dict[str, dict] = {
    'forecast': {
        'method':     'forecast',
        'bulk_kwarg': 'period',
        'dedup_cols': ['ts_code', 'ann_date', 'end_date'],
        'date_col':   'ann_date',
        'fields': [
            'ts_code', 'ann_date', 'end_date', 'type', 'p_change_min', 'p_change_max',
            'net_profit_min', 'net_profit_max', 'last_parent_net', 'first_ann_date',
            'summary', 'change_reason',
        ],
    },
    'express': {
        'method':     'express',
        'bulk_kwarg': 'period',
        'dedup_cols': ['ts_code', 'ann_date', 'end_date'],
        'date_col':   'ann_date',
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
        'method':     'dividend',
        # ``dividend`` accepts ``end_date`` for period-style bulk (year-end of
        # the dividend year).  Dividends are usually annual so pulling 0331,
        # 0630, 0930, 1231 covers interim and final distributions.
        'bulk_kwarg': 'end_date',
        'dedup_cols': ['ts_code', 'end_date', 'div_proc'],
        'date_col':   'ann_date',
        'fields': [
            'ts_code', 'end_date', 'ann_date', 'div_proc', 'stk_div', 'stk_bo_rate',
            'stk_co_rate', 'cash_div', 'cash_div_tax', 'record_date', 'ex_date',
            'pay_date', 'div_listdate', 'imp_ann_date', 'base_date', 'base_share',
        ],
    },
    'fina_audit': {
        'method':     'fina_audit',
        'bulk_kwarg': 'period',
        'dedup_cols': ['ts_code', 'ann_date', 'end_date'],
        'date_col':   'ann_date',
        'fields': [
            'ts_code', 'ann_date', 'end_date', 'audit_result', 'audit_fees',
            'audit_agency', 'audit_sign',
        ],
    },
    'fina_mainbz': {
        'method':     'fina_mainbz',
        'bulk_kwarg': 'period',
        'dedup_cols': ['ts_code', 'end_date', 'bz_item'],
        'date_col':   'end_date',
        'fields': [
            'ts_code', 'end_date', 'bz_item', 'bz_sales', 'bz_profit', 'bz_cost',
            'curr_type', 'update_flag',
        ],
    },
    'disclosure_date': {
        'method':     'disclosure_date',
        'bulk_kwarg': 'end_date',
        'dedup_cols': ['ts_code', 'end_date'],
        'date_col':   'ann_date',
        'fields': [
            'ts_code', 'ann_date', 'end_date', 'pre_date', 'actual_date', 'modify_date',
        ],
    },
}

CALLS_PER_SEC = 5.0    # bumped from the original 2.0 — easy to tune down if needed
MAX_RETRIES   = 3
RETRY_DELAY   = 2
PAGE_SIZE     = 6000


# ─── Rate limiter ─────────────────────────────────────────────────────────────

class _RateLimiter:
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


# ─── Period helpers ──────────────────────────────────────────────────────────

def _generate_periods(start_year: int = 2017, end_year: Optional[int] = None) -> List[str]:
    if end_year is None:
        end_year = datetime.now().year
    out: List[str] = []
    for y in range(start_year, end_year + 1):
        for q in ('0331', '0630', '0930', '1231'):
            out.append(f"{y}{q}")
    return out


def _fetch_period_bulk(data_type: str, period: str) -> Optional[pd.DataFrame]:
    """Bulk-fetch one period across all stocks, paginating via offset."""
    cfg    = DATA_TYPES[data_type]
    pro    = _get_pro()
    method = getattr(pro, cfg['method'])
    bulk_k = cfg['bulk_kwarg']

    pages: List[pd.DataFrame] = []
    offset = 0
    while True:
        kwargs = {bulk_k: period, 'offset': offset, 'limit': PAGE_SIZE}
        df = _fetch(method, **kwargs)
        if df is None:
            return None
        if df.empty:
            break
        pages.append(df)
        if len(df) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not pages:
        return pd.DataFrame()
    return pd.concat(pages, ignore_index=True)


# ─── Per-stock CSV helpers ───────────────────────────────────────────────────

def _setup(data_type: str):
    (DATA_DIR / data_type).mkdir(parents=True, exist_ok=True)


def _stock_csv(data_type: str, ts_code: str) -> Path:
    return DATA_DIR / data_type / f"{ts_code.replace('.', '_')}.csv"


def _last_date(data_type: str, ts_code: str, col: str) -> Optional[str]:
    fp = _stock_csv(data_type, ts_code)
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, usecols=[col])
        if df.empty:
            return None
        return str(int(float(df[col].max())))
    except Exception:
        return None


def _merge_into_csv(data_type: str, ts_code: str, new_rows: pd.DataFrame) -> int:
    """Upsert ``new_rows`` into the per-stock CSV using configured dedup keys."""
    cfg            = DATA_TYPES[data_type]
    available_cols = [c for c in cfg['fields'] if c in new_rows.columns]
    new_rows       = new_rows[available_cols].copy()

    fp = _stock_csv(data_type, ts_code)
    str_cols = set(cfg['dedup_cols']) | {cfg['date_col']}

    if fp.exists():
        existing = pd.read_csv(fp)
        for col in str_cols:
            if col in existing.columns:
                existing[col] = existing[col].astype(str)
            if col in new_rows.columns:
                new_rows[col] = new_rows[col].astype(str)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        for col in str_cols:
            if col in new_rows.columns:
                new_rows[col] = new_rows[col].astype(str)
        combined = new_rows

    dedup_cols = [c for c in cfg['dedup_cols'] if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep='last')

    sort_col = cfg['date_col'] if cfg['date_col'] in combined.columns else (
        'end_date' if 'end_date' in combined.columns else None
    )
    if sort_col:
        combined[sort_col] = combined[sort_col].astype(str)
        combined = combined.sort_values(sort_col)

    combined.to_csv(fp, index=False, encoding='utf-8-sig')
    return len(combined)


# ─── Bulk update ─────────────────────────────────────────────────────────────

def update_data_type_bulk(data_type: str, since_year: int = 2017,
                          end_year: Optional[int] = None) -> None:
    """Period-bulk update for one data type."""
    _setup(data_type)
    periods = _generate_periods(since_year, end_year)
    print(f"[{data_type}] bulk fetch over {len(periods)} periods "
          f"({periods[0]} → {periods[-1]}) ...")

    accum: Dict[str, List[pd.DataFrame]] = {}
    fetched_periods   = 0
    fetched_rows      = 0
    failed_periods: List[str] = []

    for p in periods:
        df = _fetch_period_bulk(data_type, p)
        if df is None:
            failed_periods.append(p)
            print(f"  [{data_type}] {p}: FAILED")
            continue
        fetched_periods += 1
        if df.empty:
            continue
        fetched_rows += len(df)
        if 'ts_code' not in df.columns:
            print(f"  [{data_type}] {p}: response missing ts_code column — skipped")
            continue
        for ts_code, grp in df.groupby('ts_code', sort=False):
            accum.setdefault(str(ts_code), []).append(grp)

    print(f"[{data_type}] fetched {fetched_periods}/{len(periods)} periods, "
          f"{fetched_rows:,} rows across {len(accum):,} stocks")
    if failed_periods:
        print(f"  WARNING: {len(failed_periods)} periods failed: {failed_periods[:5]}"
              + (" ..." if len(failed_periods) > 5 else ""))

    merged = 0
    for ts_code, frames in accum.items():
        new_rows = pd.concat(frames, ignore_index=True)
        try:
            _merge_into_csv(data_type, ts_code, new_rows)
            merged += 1
        except Exception as e:
            print(f"  [{data_type}] merge {ts_code} failed: {type(e).__name__}: {e}")

    print(f"[{data_type}] wrote {merged:,} stock CSVs.")


def update_all_bulk(since_year: int = 2017, end_year: Optional[int] = None) -> None:
    for data_type in DATA_TYPES:
        update_data_type_bulk(data_type, since_year=since_year, end_year=end_year)


# ─── Per-stock fallback (single 'stock' action) ──────────────────────────────

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
    """Per-stock fetch (kept for single-stock retry)."""
    cfg    = DATA_TYPES[data_type]
    pro    = _get_pro()
    method = getattr(pro, cfg['method'])
    last   = _last_date(data_type, ts_code, cfg['date_col'])

    kwargs = {'ts_code': ts_code}
    if last and data_type not in ('disclosure_date',):
        kwargs['start_date'] = last

    df = _fetch(method, **kwargs)
    if df is None:
        return ts_code, 'error'
    if df.empty:
        return ts_code, 'no_data'

    n = _merge_into_csv(data_type, ts_code, df)
    return ts_code, f'+{n} rows'


# ─── Status ──────────────────────────────────────────────────────────────────

def status():
    for data_type in DATA_TYPES:
        data_dir = DATA_DIR / data_type
        if not data_dir.exists():
            print(f"[{data_type}] Not downloaded yet")
            continue
        files = list(data_dir.glob('[!_]*.csv'))
        print(f"[{data_type}] {len(files)} stock files")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(action: str = 'update', **kwargs):
    """
    Actions:
        'update'          - bulk update all data types
        'forecast'        - bulk update earnings forecast only
        'express'         - bulk update express earnings only
        'dividend'        - bulk update dividend distribution only
        'fina_audit'      - bulk update audit opinions only
        'fina_mainbz'     - bulk update main business composition only
        'disclosure_date' - bulk update disclosure schedule only
        'stock'           - per-stock update across all types (requires ts_code)
        'status'          - coverage summary

    Keyword args:
        ts_code     (str) - stock code for 'stock' action
        since_year  (int) - first year to fetch (default 2017)
        end_year    (int) - last year to fetch (default current year)
    """
    since = int(kwargs.get('since_year', 2017))
    end_y = kwargs.get('end_year')
    end_y = int(end_y) if end_y is not None else None

    if action == 'update':
        update_all_bulk(since_year=since, end_year=end_y)

    elif action in DATA_TYPES:
        update_data_type_bulk(action, since_year=since, end_year=end_y)

    elif action == 'stock':
        ts_code = kwargs.get('ts_code')
        if not ts_code:
            print("Error: ts_code required for 'stock' action")
            return
        for data_type in DATA_TYPES:
            _setup(data_type)
            code, st = _download_one(data_type, ts_code)
            print(f"[{data_type}] {code}: {st}")

    elif action == 'status':
        status()

    else:
        valid = 'update | ' + ' | '.join(DATA_TYPES) + ' | stock | status'
        print(f"Unknown action: {action!r}.  Valid: {valid}")


if __name__ == '__main__':
    run('status')
