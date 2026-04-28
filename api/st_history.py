"""
Download authoritative ST/*ST history from tushare's namechange API.

For every ts_code in our universe, pull the full name-change history. A stock
is "ST at date d" iff its current name during [start_date, end_date] is one of:
   - starts with "*ST"   (special-treatment, severe)
   - starts with "ST"    (special-treatment, mild)
   - starts with "S*ST"  (legacy share-reform marker, treat same as *ST)

Output: stock_data/st_history.csv with columns
    ts_code, st_kind, start_date, end_date, name, reason
where end_date is empty for the currently-active row.

A helper `is_st_at(roster, ts_code, date)` returns 1 if ts_code is ST on `date`,
else 0. Loaded once and queried millions of times during backtest replay.

Usage:
    ./venv/Scripts/python -m api.st_history --download
    ./venv/Scripts/python -m api.st_history --check 000609.SZ 20210601
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import tushare as ts

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'
OUT_CSV  = DATA_DIR / 'st_history.csv'
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
WORKERS  = 12     # tushare namechange tolerates moderate parallelism
RATE_DELAY = 0.05  # gentle pacing per worker
MAX_RETRIES = 4


def _init_pro():
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api(TUSHARE_TOKEN)


def _classify_st(name: str) -> Optional[str]:
    """Return 'st_kind' label or None if the name does NOT indicate ST."""
    if not isinstance(name, str):
        return None
    n = name.strip()
    if n.startswith('*ST') or n.startswith('S*ST'):
        return '*ST'
    if n.startswith('ST'):
        return 'ST'
    return None


def _list_universe() -> list:
    """All A-share ts_codes we have local data for (SH + SZ)."""
    codes = []
    for sub, suffix in [('sh', 'SH'), ('sz', 'SZ')]:
        d = DATA_DIR / sub
        if not d.exists():
            continue
        codes += [f.stem + '.' + suffix for f in d.glob('*.csv')]
    return sorted(set(codes))


def _fetch_one(pro, ts_code: str) -> pd.DataFrame:
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.namechange(
                ts_code=ts_code,
                fields='ts_code,name,start_date,end_date,ann_date,change_reason',
            )
            time.sleep(RATE_DELAY)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            if 'limit' in str(e).lower() or 'frequency' in str(e).lower():
                wait = 30 + 30 * attempt
            time.sleep(wait)
    print(f"  [skip] {ts_code}: {last_exc}", flush=True)
    return pd.DataFrame()


def download() -> pd.DataFrame:
    pro = _init_pro()
    codes = _list_universe()
    print(f"[st_history] universe: {len(codes):,} ts_codes")

    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_fetch_one, pro, c): c for c in codes}
        done = 0
        for fut in as_completed(futures):
            ts_code = futures[fut]
            df = fut.result()
            if not df.empty:
                df = df.copy()
                df['st_kind'] = df['name'].map(_classify_st)
                df = df[df['st_kind'].notna()]
                if not df.empty:
                    rows.append(df)
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(codes)} done, {len(rows)} ST stocks so far", flush=True)

    if not rows:
        print("[st_history] no ST entries found")
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    # Dedup — tushare often returns duplicate rows
    out = out.drop_duplicates(
        subset=['ts_code', 'st_kind', 'start_date', 'end_date', 'change_reason'],
        keep='first',
    )
    # Order columns
    out = out[['ts_code', 'st_kind', 'start_date', 'end_date',
               'ann_date', 'name', 'change_reason']]
    out = out.sort_values(['ts_code', 'start_date']).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"[st_history] wrote {OUT_CSV}  ({len(out):,} rows, "
          f"{out['ts_code'].nunique():,} stocks ever flagged ST)")
    return out


def load_roster() -> pd.DataFrame:
    if not OUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {OUT_CSV}. Run `python -m api.st_history --download` first.")
    df = pd.read_csv(OUT_CSV, dtype={'start_date': str, 'end_date': str, 'ann_date': str})
    df['start_date'] = df['start_date'].fillna('')
    df['end_date']   = df['end_date'].fillna('')
    return df


def is_st_at(roster: pd.DataFrame, ts_code: str, date: str) -> bool:
    """True if `ts_code` was ST/*ST on `date` (YYYYMMDD)."""
    sub = roster[roster['ts_code'] == ts_code]
    if sub.empty:
        return False
    for _, r in sub.iterrows():
        if r['start_date'] and r['start_date'] <= date and \
           (not r['end_date'] or r['end_date'] >= date):
            return True
    return False


def build_daily_index(roster: pd.DataFrame) -> dict:
    """Per-stock list of (start_date, end_date, st_kind) intervals.
    Backtest hot path uses this to avoid pandas filtering inside the inner loop.
    """
    idx = {}
    for ts_code, grp in roster.groupby('ts_code'):
        intervals = []
        for _, r in grp.iterrows():
            if not r['start_date']:
                continue
            intervals.append((r['start_date'], r['end_date'] or '99999999', r['st_kind']))
        idx[ts_code] = intervals
    return idx


def is_st_fast(intervals: list, date: str) -> Optional[str]:
    """Returns the ST kind ('ST' / '*ST') if active on date else None."""
    for st_start, st_end, kind in intervals:
        if st_start <= date <= st_end:
            return kind
    return None


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true',
                   help='Refresh stock_data/st_history.csv from tushare')
    p.add_argument('--check', nargs=2, metavar=('TS_CODE', 'DATE'),
                   help='Sanity check: report ST status of TS_CODE on DATE (YYYYMMDD)')
    p.add_argument('--summary', action='store_true',
                   help='Print summary statistics about the ST roster')
    args = p.parse_args()
    if args.download:
        download()
    elif args.check:
        ts_code, date = args.check
        roster = load_roster()
        ans = is_st_at(roster, ts_code, date)
        print(f"{ts_code} on {date}: {'ST' if ans else 'normal'}")
    elif args.summary:
        roster = load_roster()
        print(f"total ST stocks: {roster['ts_code'].nunique():,}")
        print(f"total ST events: {len(roster):,}")
        print(f"ever-*ST count:  {roster[roster['st_kind']=='*ST']['ts_code'].nunique():,}")
        print(f"ever-ST count:   {roster[roster['st_kind']=='ST']['ts_code'].nunique():,}")
    else:
        p.print_help()


if __name__ == '__main__':
    _cli()
