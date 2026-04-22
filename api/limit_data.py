"""
Limit and Dragon-Tiger List Data Acquisition from Tushare Pro
- limit_list_d: Daily limit statistics (涨跌停统计)
- limit_step: Consecutive limit steps (连板统计)
- top_list: Dragon-tiger list (龙虎榜)
- top_inst: Institutional trading on dragon-tiger (龙虎榜机构明细)
- limit_list_ths: Tonghuashun limit list
- limit_cpt_list: Concept limit statistics (概念涨停统计)

Storage:
  - Per-date files: stock_data/limit_data/{data_type}/{data_type}_YYYYMMDD.csv
  - Per-stock aggregated: stock_data/limit_data/{data_type}_by_stock/{ts_code}.csv

Usage:
    from api.limit_data import run

    run()                          # download all types
    run('limit_list_d')            # download daily limit stats
    run('top_list')                # download dragon-tiger list
    run('status')                  # show coverage summary
"""

import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Set

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN   = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR        = Path('./stock_data/limit_data')
START_DATE      = '20170101'

# Data types and their API configurations
DATA_TYPES = {
    'limit_list_d': {
        'method': 'limit_list_d',
        'date_param': 'trade_date',
        'fields': [
            'trade_date', 'ts_code', 'industry', 'name', 'close', 'pct_chg', 'amount',
            'limit_amount', 'float_mv', 'total_mv', 'turnover_ratio', 'fd_amount',
            'first_time', 'last_time', 'open_times', 'up_stat', 'limit_times',
            'limit',  # U=涨停, D=跌停
        ],
    },
    'limit_step': {
        'method': 'stk_limit',  # Note: limit_step might not be a direct API; use stk_limit
        'date_param': 'trade_date',
        'fields': [
            'trade_date', 'ts_code', 'name', 'close', 'pct_chg', 'limit_times',
            'up_stat', 'limit', 'strth',
        ],
    },
    'top_list': {
        'method': 'top_list',
        'date_param': 'trade_date',
        'fields': [
            'trade_date', 'ts_code', 'name', 'close', 'pct_change', 'turnover_rate',
            'amount', 'l_sell', 'l_buy', 'l_amount', 'net_amount', 'net_rate', 'reason',
        ],
    },
    'top_inst': {
        'method': 'top_inst',
        'date_param': 'trade_date',
        'fields': [
            'trade_date', 'ts_code', 'exalter', 'buy', 'buy_rate', 'sell', 'sell_rate',
            'net_buy', 'side', 'reason',
        ],
    },
}

# These require higher API access levels
PREMIUM_DATA_TYPES = {
    'limit_list_ths': {
        'method': 'ths_hot',  # Placeholder - actual API varies
        'date_param': 'trade_date',
        'fields': ['trade_date', 'ts_code', 'name', 'limit_up_stat'],
    },
    'limit_cpt_list': {
        'method': 'limit_list',  # Concept-based limits
        'date_param': 'trade_date',
        'fields': ['trade_date', 'concept', 'limit_up_count', 'limit_down_count'],
    },
}

WORKERS       = 4
CALLS_PER_SEC = 4.0
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
                # Permission issues - return None but don't retry
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


# ─── Trading calendar ─────────────────────────────────────────────────────────

_TRADE_CAL_CACHE = {}


def _get_trading_dates(start: str, end: str) -> List[str]:
    """Get list of trading dates from Tushare trade_cal API."""
    cache_key = (start, end)
    if cache_key in _TRADE_CAL_CACHE:
        return _TRADE_CAL_CACHE[cache_key]

    pro = _get_pro()
    for attempt in range(3):
        try:
            df = pro.trade_cal(exchange='SSE', start_date=start, end_date=end, is_open='1')
            if df is not None and not df.empty:
                dates = sorted(df['cal_date'].astype(str).tolist())
                _TRADE_CAL_CACHE[cache_key] = dates
                return dates
        except Exception:
            time.sleep(1)
    return []


def _get_existing_dates(data_type: str) -> Set[str]:
    """Get set of already downloaded dates for a data type."""
    data_dir = DATA_DIR / data_type
    if not data_dir.exists():
        return set()
    existing = set()
    for f in data_dir.glob(f'{data_type}_*.csv'):
        date_str = f.stem.replace(f'{data_type}_', '')
        if len(date_str) == 8 and date_str.isdigit():
            existing.add(date_str)
    return existing


# ─── Download functions ───────────────────────────────────────────────────────

def _download_date(data_type: str, trade_date: str) -> tuple:
    """Download data for a single date. Returns (date, status, count)."""
    cfg = DATA_TYPES.get(data_type)
    if cfg is None:
        cfg = PREMIUM_DATA_TYPES.get(data_type)
    if cfg is None:
        return trade_date, 'unknown_type', 0

    data_dir = DATA_DIR / data_type
    data_dir.mkdir(parents=True, exist_ok=True)
    fp = data_dir / f"{data_type}_{trade_date}.csv"

    if fp.exists():
        return trade_date, 'exists', 0

    pro = _get_pro()
    method = getattr(pro, cfg['method'], None)
    if method is None:
        return trade_date, 'method_not_found', 0

    kwargs = {cfg['date_param']: trade_date}
    df = _fetch(method, **kwargs)

    if df is None:
        return trade_date, 'error', 0
    if df.empty:
        # Create empty file to mark date as processed
        pd.DataFrame(columns=cfg['fields']).to_csv(fp, index=False)
        return trade_date, 'no_data', 0

    # Filter to available columns
    available = [c for c in cfg['fields'] if c in df.columns]
    df = df[available].copy()
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    return trade_date, 'ok', len(df)


def update_data_type(
    data_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force: bool = False,
):
    """Update a single data type for all missing dates."""
    if data_type not in DATA_TYPES and data_type not in PREMIUM_DATA_TYPES:
        print(f"Unknown data type: {data_type}")
        return

    start = start_date or START_DATE
    end   = end_date or (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    print(f"[{data_type}] Fetching trading calendar {start} -> {end}")
    all_dates = _get_trading_dates(start, end)
    if not all_dates:
        print(f"  No trading dates found")
        return

    existing = _get_existing_dates(data_type) if not force else set()
    todo = [d for d in all_dates if d not in existing]

    if not todo:
        print(f"  All {len(all_dates)} dates already downloaded")
        return

    print(f"  Downloading {len(todo)}/{len(all_dates)} dates...")

    counters = {'ok': 0, 'err': 0, 'empty': 0}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_date, data_type, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures)):
            try:
                date, status, count = fut.result()
                if status == 'ok':
                    counters['ok'] += 1
                elif status in ('error', 'method_not_found'):
                    counters['err'] += 1
                else:
                    counters['empty'] += 1
            except Exception:
                counters['err'] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(todo)}] ok={counters['ok']} err={counters['err']}")

    print(f"\n[{data_type}] Done. ok={counters['ok']} errors={counters['err']} empty={counters['empty']}")


def update_all(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Update all data types."""
    for data_type in DATA_TYPES:
        update_data_type(data_type, start_date, end_date)


def aggregate_by_stock(data_type: str):
    """Aggregate per-date files into per-stock files for easier lookup."""
    data_dir = DATA_DIR / data_type
    out_dir  = DATA_DIR / f"{data_type}_by_stock"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"[{data_type}] No data to aggregate")
        return

    print(f"[{data_type}] Aggregating to per-stock files...")

    all_rows = []
    files = list(data_dir.glob(f'{data_type}_*.csv'))
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                all_rows.append(df)
        except Exception:
            continue

    if not all_rows:
        print(f"  No data to aggregate")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"  Combined {len(combined)} rows from {len(files)} files")

    if 'ts_code' not in combined.columns:
        print(f"  No ts_code column, skipping aggregation")
        return

    for ts_code, grp in combined.groupby('ts_code'):
        bare = str(ts_code).split('.')[0]
        out_path = out_dir / f"{bare}.csv"
        grp.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"  Saved {combined['ts_code'].nunique()} stock files")


def status():
    """Print coverage summary."""
    print("\n[Limit Data Status]")
    for data_type in list(DATA_TYPES.keys()) + list(PREMIUM_DATA_TYPES.keys()):
        data_dir = DATA_DIR / data_type
        if not data_dir.exists():
            print(f"  {data_type}: Not downloaded")
            continue
        files = list(data_dir.glob(f'{data_type}_*.csv'))
        print(f"  {data_type}: {len(files)} date files")
        if files:
            dates = sorted([f.stem.replace(f'{data_type}_', '') for f in files])
            if dates:
                print(f"    Range: {dates[0]} -> {dates[-1]}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(action: str = 'update', **kwargs):
    """
    Entry point for limit data operations.

    Actions:
        'update'        - update all data types
        'limit_list_d'  - daily limit statistics
        'limit_step'    - consecutive limit steps
        'top_list'      - dragon-tiger list
        'top_inst'      - institutional dragon-tiger
        'aggregate'     - aggregate to per-stock files
        'status'        - show coverage summary

    Keyword args:
        start_date (str) - start date YYYYMMDD
        end_date   (str) - end date YYYYMMDD
        force      (bool) - redownload existing dates
    """
    if action == 'update':
        update_all(
            start_date=kwargs.get('start_date'),
            end_date=kwargs.get('end_date'),
        )

    elif action in DATA_TYPES or action in PREMIUM_DATA_TYPES:
        update_data_type(
            action,
            start_date=kwargs.get('start_date'),
            end_date=kwargs.get('end_date'),
            force=kwargs.get('force', False),
        )

    elif action == 'aggregate':
        for data_type in DATA_TYPES:
            aggregate_by_stock(data_type)

    elif action == 'status':
        status()

    else:
        print(f"Unknown action: {action}")


if __name__ == '__main__':
    run('status')
