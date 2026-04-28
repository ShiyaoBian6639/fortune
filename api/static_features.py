"""
Download static + slow-changing features from tushare for the data audit.

Sources (all parallelisable, rate-limited):
  - stock_company   → province, industry, setup_date, reg_capital
  - stk_managers    → n_managers, avg_age, avg_education, chairman_tenure
  - stk_holdernumber→ shareholder count (quarterly)
  - top10_holders   → ownership concentration (HHI), institutional pct (quarterly)
  - index_member    → CSI300/500/SSE50/CSI1000 membership flags

Outputs:
  stock_data/static_features/stock_company.csv
  stock_data/static_features/stk_managers_summary.csv
  stock_data/static_features/stk_holdernumber.csv         (long form)
  stock_data/static_features/top10_holders_summary.csv    (long form, quarterly)
  stock_data/static_features/index_member_flags.csv

Run:
  ./venv/Scripts/python -m api.static_features --all
  ./venv/Scripts/python -m api.static_features --source stock_company
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'
OUT_DIR  = DATA_DIR / 'static_features'
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'

WORKERS    = 12
RATE_DELAY = 0.06
MAX_RETRIES = 4

EDU_RANK = {'高中': 1, '专科': 2, '大专': 2, '本科': 3, '硕士': 4, 'MBA': 4,
            'EMBA': 4, '博士': 5, '院士': 6}


def _init_pro():
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api(TUSHARE_TOKEN)


def _retry(fn, *args, label='', **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            r = fn(*args, **kwargs)
            time.sleep(RATE_DELAY)
            return r
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            if 'limit' in str(e).lower() or 'frequency' in str(e).lower():
                wait = 30 + 30 * attempt
            time.sleep(wait)
    print(f"  [skip] {label}: {last_exc}", flush=True)
    return None


def _list_universe() -> list:
    codes = []
    for sub, suf in [('sh', 'SH'), ('sz', 'SZ')]:
        d = DATA_DIR / sub
        if d.exists():
            codes += [f.stem + '.' + suf for f in d.glob('*.csv')]
    return sorted(set(codes))


# ─── Source 1: stock_company ────────────────────────────────────────────────
def fetch_stock_company():
    pro = _init_pro()
    print('[stock_company] pulling all stocks ...')
    rows = []
    # tushare allows bulk listing via stock_basic; stock_company per-stock
    for exchange in ['SSE', 'SZSE']:
        df = _retry(pro.stock_company, exchange=exchange,
                     fields='ts_code,com_name,com_id,exchange,chairman,manager,'
                            'reg_capital,setup_date,province,industry,city',
                     label=f'stock_company({exchange})')
        if df is not None and not df.empty:
            rows.append(df)
    if not rows:
        print('[stock_company] empty — skipped')
        return
    out = pd.concat(rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / 'stock_company.csv', index=False, encoding='utf-8-sig')
    print(f'[stock_company] wrote {len(out):,} rows')


# ─── Source 2: stk_managers ─────────────────────────────────────────────────
def _stk_managers_one(pro, ts_code: str) -> dict:
    df = _retry(pro.stk_managers, ts_code=ts_code,
                 fields='ts_code,name,gender,birthday,edu,title,begin_date,end_date',
                 label=f'stk_managers({ts_code})')
    if df is None or df.empty:
        return None
    df = df.copy()
    df['edu_rank'] = df['edu'].map(EDU_RANK).fillna(0)
    # current managers (end_date null or future)
    today = int(time.strftime('%Y%m%d'))
    end_int = pd.to_numeric(df['end_date'].astype(str).str[:8],
                              errors='coerce')
    cur = df[(end_int.isna()) | (end_int >= today)]
    n_mgr = len(cur)
    # current chairman tenure
    chair = cur[cur['title'].astype(str).str.contains('董事长', na=False)]
    if len(chair):
        bd = chair['begin_date'].astype(str).str[:8]
        bd_int = pd.to_numeric(bd, errors='coerce').dropna().astype(int)
        chair_tenure = int(today - int(bd_int.min())) if len(bd_int) else 0
    else:
        chair_tenure = 0
    avg_age, avg_edu = 0.0, 0.0
    if len(cur):
        bday = pd.to_numeric(cur['birthday'].astype(str).str[:4],
                              errors='coerce').dropna()
        if len(bday):
            avg_age = float((today // 10000) - bday.mean())
        avg_edu = float(cur['edu_rank'].mean()) if len(cur) else 0.0
    return {'ts_code': ts_code, 'n_managers': n_mgr,
            'avg_manager_age': avg_age, 'avg_education_rank': avg_edu,
            'chairman_tenure_days': chair_tenure}


def fetch_stk_managers():
    pro = _init_pro()
    codes = _list_universe()
    print(f'[stk_managers] processing {len(codes):,} stocks (workers={WORKERS}) ...')
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_stk_managers_one, pro, c): c for c in codes}
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r: rows.append(r)
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(codes)}  collected {len(rows)}', flush=True)
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / 'stk_managers_summary.csv', index=False, encoding='utf-8-sig')
    print(f'[stk_managers] wrote {len(out):,} rows')


# ─── Source 3: stk_holdernumber ─────────────────────────────────────────────
def _holdernum_one(pro, ts_code: str) -> pd.DataFrame:
    df = _retry(pro.stk_holdernumber, ts_code=ts_code,
                 start_date='20160101',
                 fields='ts_code,ann_date,end_date,holder_num',
                 label=f'stk_holdernumber({ts_code})')
    return df


def fetch_stk_holdernumber():
    pro = _init_pro()
    codes = _list_universe()
    print(f'[stk_holdernumber] processing {len(codes):,} stocks ...')
    frames = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_holdernum_one, pro, c): c for c in codes}
        for i, f in enumerate(as_completed(futs)):
            df = f.result()
            if df is not None and not df.empty: frames.append(df)
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(codes)}  frames={len(frames)}', flush=True)
    if not frames: return
    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / 'stk_holdernumber.csv', index=False, encoding='utf-8-sig')
    print(f'[stk_holdernumber] wrote {len(out):,} rows over {out["ts_code"].nunique()} stocks')


# ─── Source 4: top10_holders ────────────────────────────────────────────────
def _top10_one(pro, ts_code: str) -> pd.DataFrame:
    """Compute summary stats per (ts_code, ann_date)."""
    df = _retry(pro.top10_holders, ts_code=ts_code,
                 start_date='20160101',
                 fields='ts_code,ann_date,end_date,holder_name,hold_amount,hold_ratio',
                 label=f'top10_holders({ts_code})')
    if df is None or df.empty: return None
    out = []
    for (ts, end), grp in df.groupby(['ts_code', 'end_date']):
        ratios = grp['hold_ratio'].fillna(0).values / 100.0  # tushare returns %
        hhi = float(np.sum(ratios ** 2))
        n_funds = int(grp['holder_name'].astype(str).str.contains(
            '基金|资管|信托|保险|养老', na=False).sum())
        ann = grp['ann_date'].astype(str).min()
        out.append({
            'ts_code': ts, 'ann_date': ann, 'end_date': end,
            'top10_pct':       float(ratios.sum()),
            'top10_hhi':       hhi,
            'top10_pct_top1':  float(ratios.max()) if len(ratios) else 0.0,
            'n_funds_in_top10': n_funds,
        })
    return pd.DataFrame(out)


def fetch_top10_holders():
    pro = _init_pro()
    codes = _list_universe()
    print(f'[top10_holders] processing {len(codes):,} stocks ...')
    frames = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_top10_one, pro, c): c for c in codes}
        for i, f in enumerate(as_completed(futs)):
            df = f.result()
            if df is not None and not df.empty: frames.append(df)
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(codes)}  frames={len(frames)}', flush=True)
    if not frames: return
    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / 'top10_holders_summary.csv', index=False, encoding='utf-8-sig')
    print(f'[top10_holders] wrote {len(out):,} rows')


# ─── Source 5: index_member flags ───────────────────────────────────────────
def fetch_index_member_flags():
    pro = _init_pro()
    indexes = {
        '000300.SH': 'in_csi300',
        '000905.SH': 'in_csi500',
        '000016.SH': 'in_sse50',
        '000852.SH': 'in_csi1000',
    }
    flags = {idx_col: set() for idx_col in indexes.values()}
    for idx_code, col in indexes.items():
        df = _retry(pro.index_weight, index_code=idx_code, start_date='20260101',
                     fields='index_code,con_code,trade_date',
                     label=f'index_weight({idx_code})')
        if df is None or df.empty: continue
        latest = df['trade_date'].max()
        df = df[df['trade_date'] == latest]
        flags[col] = set(df['con_code'])
        print(f'  {idx_code}: {len(flags[col])} members on {latest}')
    codes = _list_universe()
    rows = []
    for ts in codes:
        rows.append({'ts_code': ts, **{c: int(ts in flags[c]) for c in flags}})
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / 'index_member_flags.csv', index=False, encoding='utf-8-sig')
    print(f'[index_member] wrote {len(out):,} rows')


SOURCES = {
    'stock_company':       fetch_stock_company,
    'stk_managers':        fetch_stk_managers,
    'stk_holdernumber':    fetch_stk_holdernumber,
    'top10_holders':       fetch_top10_holders,
    'index_member':        fetch_index_member_flags,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=list(SOURCES.keys()), default=None)
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        for k, fn in SOURCES.items():
            print(f'\n========== {k} ==========')
            fn()
    elif args.source:
        SOURCES[args.source]()
    else:
        p.print_help()


if __name__ == '__main__':
    main()
