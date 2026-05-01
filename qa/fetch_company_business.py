"""
One-shot Tushare pull for main_business + business_scope per ts_code.

Bulk-fetches via pro.stock_company per exchange (2 API calls total) and
writes a thin sidecar `stock_data/qa/company_business.csv`. The entity
index reads this and adds a "主营: ..." line to each card so bge-m3
can match concept queries like "医美 / 智能驾驶 / 国产CPU" against the
actual filed business description (which is independent of news
recency / coverage).

Tushare returns:
  main_business    主要业务及产品   ~50–200 chars, focused
  business_scope   经营范围         ~200–800 chars, legal text

We keep main_business in full and truncate business_scope to 300 chars.

Run:
    ./venv/Scripts/python -m qa.fetch_company_business
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'
QA_DIR = DATA / 'qa'

# Reuse the project's token
import sys; sys.path.insert(0, str(ROOT))
from dl.config import TUSHARE_TOKEN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(QA_DIR / 'company_business.csv'))
    p.add_argument('--scope_chars', type=int, default=300)
    args = p.parse_args()

    QA_DIR.mkdir(parents=True, exist_ok=True)

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    rows = []
    for exch in ['SSE', 'SZSE']:
        for attempt in range(3):
            try:
                df = pro.stock_company(
                    exchange=exch,
                    fields='ts_code,main_business,business_scope',
                )
                break
            except Exception as e:
                print(f"[fetch] {exch} attempt {attempt+1}: {e}")
                time.sleep(3)
        else:
            print(f"[fetch] {exch} FAILED after 3 attempts")
            continue
        if df is None or df.empty:
            print(f"[fetch] {exch} empty")
            continue
        rows.append(df)
        print(f"[fetch] {exch}: {len(df):,} rows")

    if not rows:
        print("[fetch] no data")
        return

    out = pd.concat(rows, ignore_index=True).drop_duplicates('ts_code')
    out['main_business']  = out['main_business'].fillna('').astype(str)
    out['business_scope'] = (out['business_scope'].fillna('').astype(str)
                              .str.replace(r'\s+', ' ', regex=True)
                              .str[: args.scope_chars])

    out.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f"[fetch] wrote {len(out):,} rows → {args.out}")
    print(f"[fetch] avg main_business chars: "
          f"{out['main_business'].str.len().mean():.0f}")
    print(f"[fetch] avg business_scope chars: "
          f"{out['business_scope'].str.len().mean():.0f}")

    # Spot-check
    for ts_code in ('600519.SH', '300750.SZ', '002594.SZ',
                     '300896.SZ', '300496.SZ', '688041.SH'):
        row = out[out['ts_code'] == ts_code]
        if row.empty: continue
        r = row.iloc[0]
        print(f"\n  {ts_code}")
        print(f"    main_business : {r['main_business'][:120]}")
        print(f"    business_scope: {r['business_scope'][:120]}")


if __name__ == '__main__':
    main()
