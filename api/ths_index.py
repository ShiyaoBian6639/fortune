"""
Tonghuashun (THS) and Dongcai (DC) Index Data Acquisition from Tushare Pro
- ths_index: THS index list (同花顺概念/行业指数列表)
- ths_daily: THS index daily data (同花顺指数日线)
- ths_member: THS index members (同花顺指数成分)
- dc_index: Dongcai index list (东财概念指数列表) - not available in basic API
- dc_member: Dongcai index members (东财概念成分) - not available in basic API

Storage:
  - Index list: stock_data/ths_index/ths_index.csv
  - Daily data: stock_data/ths_index/ths_daily/{ts_code}.csv
  - Members: stock_data/ths_index/ths_member/{ts_code}.csv

Usage:
    from api.ths_index import run

    run()                    # download all THS data
    run('ths_index')         # download index list only
    run('ths_daily')         # download daily data
    run('ths_member')        # download member data
    run('status')            # show coverage summary
"""

import json
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
DATA_DIR        = Path('./stock_data/ths_index')
START_DATE      = '20170101'

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
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


# ─── Download functions ───────────────────────────────────────────────────────

def download_ths_index():
    """Download THS index list."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / 'ths_index.csv'

    print("[ths_index] Downloading index list...")
    pro = _get_pro()

    # ths_index returns all THS indices
    # exchange: A-A股, HK-港股, US-美股
    # type: N-概念指数, I-行业指数, S-特色指数
    all_indices = []

    for exchange in ['A']:
        for idx_type in ['N', 'I', 'S']:
            df = _fetch(pro.ths_index, exchange=exchange, type=idx_type)
            if df is not None and not df.empty:
                df['exchange'] = exchange
                df['type'] = idx_type
                all_indices.append(df)
                print(f"  {exchange}-{idx_type}: {len(df)} indices")
            time.sleep(0.5)

    if not all_indices:
        print("  No indices found or permission denied")
        return

    combined = pd.concat(all_indices, ignore_index=True)
    combined.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f"  Saved {len(combined)} indices to {fp}")


def download_ths_daily(force: bool = False):
    """Download THS index daily data."""
    index_fp = DATA_DIR / 'ths_index.csv'
    if not index_fp.exists():
        print("[ths_daily] No index list found, downloading first...")
        download_ths_index()
        if not index_fp.exists():
            return

    out_dir = DATA_DIR / 'ths_daily'
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = pd.read_csv(index_fp)
    if 'ts_code' not in indices.columns:
        print("  No ts_code column in index list")
        return

    ts_codes = indices['ts_code'].dropna().unique().tolist()
    print(f"[ths_daily] Processing {len(ts_codes)} indices...")

    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    counters = {'ok': 0, 'err': 0}

    def _download_one(ts_code: str) -> tuple:
        fp = out_dir / f"{ts_code.replace('.', '_')}.csv"

        # Get last date if exists
        start = START_DATE
        if fp.exists() and not force:
            try:
                existing = pd.read_csv(fp, usecols=['trade_date'])
                if not existing.empty:
                    last = str(int(existing['trade_date'].max()))
                    if last >= end_date:
                        return ts_code, 'up_to_date', 0
                    start = last
            except Exception:
                pass

        pro = _get_pro()
        df = _fetch(pro.ths_daily, ts_code=ts_code, start_date=start, end_date=end_date)

        if df is None:
            return ts_code, 'error', 0
        if df.empty:
            return ts_code, 'no_data', 0

        if fp.exists() and not force:
            existing = pd.read_csv(fp)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=['trade_date'], keep='last')

        df = df.sort_values('trade_date')
        df.to_csv(fp, index=False, encoding='utf-8-sig')
        return ts_code, 'ok', len(df)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_one, code): code for code in ts_codes}
        for i, fut in enumerate(as_completed(futures)):
            try:
                code, status, count = fut.result()
                if status == 'ok':
                    counters['ok'] += 1
                elif status == 'error':
                    counters['err'] += 1
            except Exception:
                counters['err'] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(ts_codes)}] ok={counters['ok']} err={counters['err']}")

    print(f"\n[ths_daily] Done. ok={counters['ok']} errors={counters['err']}")


def download_ths_member():
    """Download THS index member constituents."""
    index_fp = DATA_DIR / 'ths_index.csv'
    if not index_fp.exists():
        print("[ths_member] No index list found, downloading first...")
        download_ths_index()
        if not index_fp.exists():
            return

    out_dir = DATA_DIR / 'ths_member'
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = pd.read_csv(index_fp)
    ts_codes = indices['ts_code'].dropna().unique().tolist()
    print(f"[ths_member] Processing {len(ts_codes)} indices...")

    counters = {'ok': 0, 'err': 0}

    def _download_one(ts_code: str) -> tuple:
        fp = out_dir / f"{ts_code.replace('.', '_')}.csv"

        # Skip if recent file exists (members don't change daily)
        if fp.exists():
            mtime = datetime.fromtimestamp(fp.stat().st_mtime)
            if (datetime.now() - mtime).days < 7:
                return ts_code, 'cached', 0

        pro = _get_pro()
        df = _fetch(pro.ths_member, ts_code=ts_code)

        if df is None:
            return ts_code, 'error', 0
        if df.empty:
            return ts_code, 'no_data', 0

        df.to_csv(fp, index=False, encoding='utf-8-sig')
        return ts_code, 'ok', len(df)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_one, code): code for code in ts_codes}
        for i, fut in enumerate(as_completed(futures)):
            try:
                code, status, count = fut.result()
                if status == 'ok':
                    counters['ok'] += 1
                elif status == 'error':
                    counters['err'] += 1
            except Exception:
                counters['err'] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(ts_codes)}] ok={counters['ok']} err={counters['err']}")

    print(f"\n[ths_member] Done. ok={counters['ok']} errors={counters['err']}")


def download_dc_index():
    """Download Dongcai (DC) concept index data."""
    # Note: dc_index is not available in basic Tushare API
    # This is a placeholder for when/if it becomes available
    dc_dir = DATA_DIR.parent / 'dc_index'
    dc_dir.mkdir(parents=True, exist_ok=True)

    print("[dc_index] Attempting to download Dongcai indices...")
    pro = _get_pro()

    # Try to get DC index list (may not be available)
    df = _fetch(getattr(pro, 'concept', lambda: None))
    if df is None or (hasattr(df, 'empty') and df.empty):
        print("  dc_index API not available or permission denied")
        print("  Creating placeholder with available concept data...")

        # Alternative: use concept() API if available
        try:
            df = pro.concept()
            if df is not None and not df.empty:
                fp = dc_dir / 'dc_concept.csv'
                df.to_csv(fp, index=False, encoding='utf-8-sig')
                print(f"  Saved {len(df)} concepts to {fp}")
        except Exception as e:
            print(f"  concept() also unavailable: {e}")
        return

    fp = dc_dir / 'dc_index.csv'
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    print(f"  Saved {len(df)} DC indices to {fp}")


def download_dc_member():
    """Download Dongcai concept members."""
    dc_dir = DATA_DIR.parent / 'dc_index'
    index_fp = dc_dir / 'dc_concept.csv'

    if not index_fp.exists():
        print("[dc_member] No DC concept list found, downloading first...")
        download_dc_index()
        if not index_fp.exists():
            return

    out_dir = dc_dir / 'dc_member'
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = pd.read_csv(index_fp)
    if 'code' not in indices.columns:
        print("  No code column in DC concept list")
        return

    codes = indices['code'].dropna().unique().tolist()
    print(f"[dc_member] Processing {len(codes)} concepts...")

    pro = _get_pro()
    counters = {'ok': 0, 'err': 0}

    for code in codes:
        try:
            fp = out_dir / f"{code}.csv"
            if fp.exists():
                mtime = datetime.fromtimestamp(fp.stat().st_mtime)
                if (datetime.now() - mtime).days < 7:
                    continue

            df = _fetch(pro.concept_detail, id=code)
            if df is not None and not df.empty:
                df.to_csv(fp, index=False, encoding='utf-8-sig')
                counters['ok'] += 1
            else:
                counters['err'] += 1
        except Exception:
            counters['err'] += 1

    print(f"\n[dc_member] Done. ok={counters['ok']} errors={counters['err']}")


def status():
    """Print coverage summary."""
    print("\n[THS/DC Index Status]")

    # THS index list
    index_fp = DATA_DIR / 'ths_index.csv'
    if index_fp.exists():
        df = pd.read_csv(index_fp)
        print(f"  ths_index: {len(df)} indices")
    else:
        print("  ths_index: Not downloaded")

    # THS daily
    daily_dir = DATA_DIR / 'ths_daily'
    if daily_dir.exists():
        files = list(daily_dir.glob('*.csv'))
        print(f"  ths_daily: {len(files)} index files")
    else:
        print("  ths_daily: Not downloaded")

    # THS members
    member_dir = DATA_DIR / 'ths_member'
    if member_dir.exists():
        files = list(member_dir.glob('*.csv'))
        print(f"  ths_member: {len(files)} index files")
    else:
        print("  ths_member: Not downloaded")

    # DC data
    dc_dir = DATA_DIR.parent / 'dc_index'
    if dc_dir.exists():
        concept_fp = dc_dir / 'dc_concept.csv'
        if concept_fp.exists():
            df = pd.read_csv(concept_fp)
            print(f"  dc_index: {len(df)} concepts")
        member_dir = dc_dir / 'dc_member'
        if member_dir.exists():
            files = list(member_dir.glob('*.csv'))
            print(f"  dc_member: {len(files)} concept files")
    else:
        print("  dc_index: Not downloaded")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(action: str = 'update', **kwargs):
    """
    Entry point for THS/DC index operations.

    Actions:
        'update'     - download all THS and DC data
        'ths_index'  - THS index list only
        'ths_daily'  - THS daily data only
        'ths_member' - THS members only
        'dc_index'   - DC concept list only
        'dc_member'  - DC members only
        'status'     - show coverage summary

    Keyword args:
        force (bool) - redownload existing data
    """
    if action == 'update':
        download_ths_index()
        download_ths_daily(force=kwargs.get('force', False))
        download_ths_member()
        download_dc_index()
        download_dc_member()

    elif action == 'ths_index':
        download_ths_index()

    elif action == 'ths_daily':
        download_ths_daily(force=kwargs.get('force', False))

    elif action == 'ths_member':
        download_ths_member()

    elif action == 'dc_index':
        download_dc_index()

    elif action == 'dc_member':
        download_dc_member()

    elif action == 'status':
        status()

    else:
        print(f"Unknown action: {action}")


if __name__ == '__main__':
    run('status')
