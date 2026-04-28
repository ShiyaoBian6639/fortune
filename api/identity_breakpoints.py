"""
Build skeleton identity_breakpoints.csv from namechange data.

Auto-detects candidate breakpoints whose change_reason indicates a possible
business-identity change (借壳, 重大资产重组, 主营变更). Marks them as
`status=detected`. Operator manually reviews and changes `detected` →
`confirmed` or `rejected`. Only `confirmed` rows trigger pre-event row-drop
in data_loader.

Source: stock_data/st_history.csv has the namechange data we already pulled.

Run:
    ./venv/Scripts/python -m api.identity_breakpoints --build
    ./venv/Scripts/python -m api.identity_breakpoints --check 600234.SH
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'
OUT_PATH = DATA_DIR / 'identity_breakpoints.csv'
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'

# Reasons in `change_reason` that suggest a business-identity change
SUSPICIOUS_PATTERNS = [
    '重大事项', '重大资产重组', '资产重组', '业务调整',
    '主营变更', '主业变更', '主要业务变更',
    '股权转让', '终止上市', '重新上市',
]


def _init_pro():
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api(TUSHARE_TOKEN)


def _list_universe() -> list:
    codes = []
    for sub, suf in [('sh', 'SH'), ('sz', 'SZ')]:
        d = DATA_DIR / sub
        if d.exists():
            codes += [f.stem + '.' + suf for f in d.glob('*.csv')]
    return sorted(set(codes))


def _suspicious(reason: str) -> bool:
    if not isinstance(reason, str):
        return False
    return any(pat in reason for pat in SUSPICIOUS_PATTERNS)


def build():
    """Pull namechange for every stock, flag suspicious entries."""
    pro = _init_pro()
    codes = _list_universe()
    print(f'[identity_breakpoints] scanning {len(codes):,} stocks ...')
    rows = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    def _fetch(ts_code):
        for attempt in range(4):
            try:
                df = pro.namechange(
                    ts_code=ts_code,
                    fields='ts_code,name,start_date,end_date,ann_date,change_reason',
                )
                time.sleep(0.04)
                return ts_code, df
            except Exception as e:
                if 'limit' in str(e).lower() or 'frequency' in str(e).lower():
                    time.sleep(30 * (attempt + 1))
                else:
                    time.sleep(2 ** attempt)
        return ts_code, None

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_fetch, c) for c in codes]
        for i, f in enumerate(as_completed(futs)):
            ts_code, df = f.result()
            if df is None or df.empty: continue
            for _, r in df.iterrows():
                if _suspicious(r['change_reason']):
                    rows.append({
                        'ts_code':       ts_code,
                        'breakpoint_date': str(r['start_date']),
                        'change_reason': r['change_reason'],
                        'name_at_change': r['name'],
                        'ann_date':      str(r['ann_date']),
                        'status':        'detected',
                        'notes':         '',
                    })
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(codes)} scanned, {len(rows)} candidates flagged', flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=['ts_code','breakpoint_date','change_reason',
                                      'name_at_change','ann_date','status','notes'])
    out = out.drop_duplicates(['ts_code','breakpoint_date','change_reason'])
    out = out.sort_values(['ts_code','breakpoint_date']).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'[identity_breakpoints] wrote {len(out):,} candidate breakpoints '
          f'across {out["ts_code"].nunique()} stocks → {OUT_PATH}')
    print(f'  status mix: {out["status"].value_counts().to_dict()}')


def check(ts_code: str):
    if not OUT_PATH.exists():
        print('Run --build first.')
        return
    df = pd.read_csv(OUT_PATH, encoding='utf-8-sig')
    sub = df[df['ts_code'] == ts_code]
    if sub.empty:
        print(f'{ts_code}: no breakpoints detected')
    else:
        print(sub.to_string(index=False))


def confirmed_breakpoints() -> dict:
    """Return {ts_code: earliest_confirmed_breakpoint_date_str}.
    Used by data_loader to drop rows with trade_date < breakpoint."""
    if not OUT_PATH.exists():
        return {}
    df = pd.read_csv(OUT_PATH, encoding='utf-8-sig', dtype={'breakpoint_date': str})
    df = df[df['status'] == 'confirmed']
    if df.empty:
        return {}
    return df.groupby('ts_code')['breakpoint_date'].min().to_dict()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', action='store_true')
    p.add_argument('--check', metavar='TS_CODE')
    args = p.parse_args()
    if args.build:
        build()
    elif args.check:
        check(args.check)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
