"""
Fix the survivorship-bias hole: download OHLCV for every delisted A-share that
delisted after our backtest start (2016-01-01) and is currently MISSING from
stock_data/{sh,sz}/. After running this, the modeling panel will include the
true point-in-time universe.

Source: tushare pro.stock_basic(list_status='D') for the delisted roster +
pro.daily(ts_code=...) for each stock's OHLCV history. The download is
incremental — already-downloaded stocks are skipped.

Run:
    ./venv/Scripts/python -m api.delisted_ohlcv

Outputs: stock_data/sh/{code}.csv, stock_data/sz/{code}.csv
         stock_data/delisted_manifest.csv  (audit trail of what was added)
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
RATE_DELAY    = 0.05   # per call in worker
WORKERS       = 8
MAX_RETRIES   = 4
START_DATE    = '20160101'


def _init_pro():
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api(TUSHARE_TOKEN)


def _local_universe() -> set:
    codes = set()
    for sub, suffix in [('sh', 'SH'), ('sz', 'SZ')]:
        d = DATA_DIR / sub
        if d.exists():
            codes |= {f.stem + '.' + suffix for f in d.glob('*.csv')}
    return codes


def _fetch_ohlcv(pro, ts_code: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for one stock; columns match existing per-stock CSVs."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
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


def _save_one(pro, row) -> dict:
    ts_code = row['ts_code']
    code, suffix = ts_code.split('.')
    sub = 'sh' if suffix.upper() == 'SH' else 'sz'
    fp = DATA_DIR / sub / f'{code}.csv'
    if fp.exists():
        return {'ts_code': ts_code, 'status': 'skip-exists', 'rows': 0}
    delist_date = str(row.get('delist_date') or '20260101')
    if delist_date == 'nan':
        delist_date = '20260101'
    list_date   = str(row.get('list_date')   or START_DATE)
    if list_date == 'nan':
        list_date = START_DATE
    start = max(list_date, START_DATE)
    df = _fetch_ohlcv(pro, ts_code, start, delist_date)
    if df.empty:
        return {'ts_code': ts_code, 'status': 'empty', 'rows': 0,
                'delist_date': delist_date}
    # Conform to existing per-stock CSV schema
    expected = ['ts_code','trade_date','open','high','low','close','pre_close',
                'change','pct_chg','vol','amount']
    for c in expected:
        if c not in df.columns:
            df[c] = None
    df = df[expected].sort_values('trade_date').reset_index(drop=True)
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp, index=False, encoding='utf-8-sig')
    return {'ts_code': ts_code, 'status': 'ok', 'rows': len(df),
            'first': str(df['trade_date'].min()),
            'last':  str(df['trade_date'].max()),
            'delist_date': delist_date}


def main():
    pro = _init_pro()
    print('[delisted_ohlcv] fetching delisted-stock roster from tushare ...')
    df = pro.stock_basic(exchange='', list_status='D',
                          fields='ts_code,name,list_date,delist_date')
    df['delist_date_str'] = df['delist_date'].astype(str)
    df = df[df['delist_date_str'] >= START_DATE].reset_index(drop=True)
    print(f'[delisted_ohlcv] {len(df):,} delisted stocks since {START_DATE}')

    local = _local_universe()
    df['present_locally'] = df['ts_code'].isin(local)
    missing = df[~df['present_locally']].copy()
    have = int(df['present_locally'].sum())
    print(f'[delisted_ohlcv] already have CSVs for {have}; '
          f'NEED to download {len(missing):,}')

    if missing.empty:
        print('[delisted_ohlcv] nothing to download.')
        return

    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut_map = {ex.submit(_save_one, pro, r): r['ts_code'] for _, r in missing.iterrows()}
        done = 0
        for fut in as_completed(fut_map):
            r = fut.result()
            rows.append(r)
            done += 1
            if done % 25 == 0:
                ok = sum(1 for x in rows if x['status'] == 'ok')
                print(f'  {done}/{len(missing)} processed, {ok} OK so far', flush=True)

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / 'delisted_manifest.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\n[delisted_ohlcv] complete:')
    print(out['status'].value_counts().to_string())
    print(f'  manifest → {out_path}')

    # Final: delisted stocks now in local universe
    local_after = _local_universe()
    print(f'\nlocal universe: before={len(local):,}  after={len(local_after):,}  '
          f'delta={len(local_after) - len(local):+d}')


if __name__ == '__main__':
    main()
