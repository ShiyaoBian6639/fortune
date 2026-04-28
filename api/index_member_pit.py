"""
Build a point-in-time index-membership table from monthly `pro.index_weight`
snapshots that have already been downloaded to `stock_data/index/index_weight/`.

The output table is consumed by `xgbmodel.data_loader._merge_static_features`
via `pd.merge_asof(direction='backward')`, so each panel row only sees the
membership flags that existed *as of its own trade_date*. This replaces the
old today-snapshot `index_member_flags.csv`, which leaked the current index
composition back into 2017 training data.

Schema produced (long form, one row per (ts_code, snapshot_date)):

    ts_code    | snapshot_date | in_csi300 | in_csi500 | in_csi1000 | in_sse50
    600519.SH  | 2017-01-26    |    1.0    |    0.0    |    0.0     |   1.0
    ...

A stock appears in the table on every snapshot_date during which it was a
member of *any* of the four tracked indices. Stocks never in any tracked
index are absent from the table — `merge_asof` will still match them
(with NaN), and the data_loader fills NaN→0.

Output:
    stock_data/static_features/index_member_pit.csv

Run:
    ./venv/Scripts/python -m api.index_member_pit
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
WEIGHT_D  = ROOT / 'stock_data' / 'index' / 'index_weight'
OUT_DIR   = ROOT / 'stock_data' / 'static_features'
OUT_PATH  = OUT_DIR / 'index_member_pit.csv'

INDEX_TO_FLAG = {
    '000300.SH': 'in_csi300',
    '000905.SH': 'in_csi500',
    '000852.SH': 'in_csi1000',
    '000016.SH': 'in_sse50',
}


def _safe_code(code: str) -> str:
    return code.replace('.', '_')


def build_pit_table() -> pd.DataFrame:
    """Read the four index_weight files and emit a long-form PIT membership table."""
    frames = []
    for idx_code, flag_col in INDEX_TO_FLAG.items():
        fp = WEIGHT_D / f'{_safe_code(idx_code)}.csv'
        if not fp.exists():
            print(f"[index_member_pit] WARNING {fp} missing — skipping {idx_code}")
            continue
        df = pd.read_csv(fp, encoding='utf-8-sig')
        if df.empty:
            print(f"[index_member_pit] WARNING {idx_code}: empty file")
            continue
        df['snapshot_date'] = pd.to_datetime(df['trade_date'].astype(str),
                                              format='%Y%m%d')
        snapshots = sorted(df['snapshot_date'].unique())
        print(f"[index_member_pit] {idx_code} → {flag_col}: "
              f"{len(snapshots)} snapshots, "
              f"{snapshots[0].date()} → {snapshots[-1].date()}, "
              f"{df['con_code'].nunique()} unique members")
        sub = df[['con_code', 'snapshot_date']].rename(columns={'con_code': 'ts_code'})
        sub[flag_col] = 1.0
        frames.append(sub)

    if not frames:
        raise RuntimeError("No index_weight files found — run api.get_data.fetch_index_weight first.")

    # Outer-merge all four flags onto a unified (ts_code, snapshot_date) grid.
    # Pivot via groupby then full outer concat.
    combined = pd.concat(frames, ignore_index=True)
    pivot = (combined
             .groupby(['ts_code', 'snapshot_date'], as_index=False)
             .max())
    for col in INDEX_TO_FLAG.values():
        if col not in pivot.columns:
            pivot[col] = 0.0
        pivot[col] = pivot[col].fillna(0.0).astype('float32')

    # Forward-fill within stock — once a stock is recorded as a member at
    # snapshot t, it remains a member at every snapshot after t until a
    # later snapshot explicitly omits it. Without forward-fill, the table
    # would only show membership on the rebalance dates themselves.
    #
    # CONSERVATIVE choice: we *do not* forward-fill. The merge_asof on the
    # consuming side uses direction='backward' which already propagates
    # the last observed snapshot forward in time on its own. Forward-fill
    # here would also propagate gaps in the snapshot grid (e.g. quarters
    # where Tushare did not publish a member list), which is wrong: the
    # truth is "we don't know between the last known snapshot and the next".
    # merge_asof('backward') gives that truth.
    pivot = (pivot
             .sort_values(['ts_code', 'snapshot_date'])
             .reset_index(drop=True))

    print(f"\n[index_member_pit] final table: "
          f"{len(pivot):,} rows  ×  {len(pivot.columns)} cols")
    print(f"  unique stocks: {pivot['ts_code'].nunique():,}")
    print(f"  date span:     {pivot['snapshot_date'].min().date()} → "
          f"{pivot['snapshot_date'].max().date()}")
    print(f"  membership totals (any-snapshot):")
    for col in INDEX_TO_FLAG.values():
        n = (pivot.groupby('ts_code')[col].max() == 1.0).sum()
        print(f"    {col:14}  {n:>5} unique stocks ever-in")
    return pivot


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pit = build_pit_table()
    # Write with snapshot_date as YYYY-MM-DD so pandas reloads cleanly
    pit_out = pit.copy()
    pit_out['snapshot_date'] = pit_out['snapshot_date'].dt.strftime('%Y-%m-%d')
    pit_out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[index_member_pit] wrote {OUT_PATH}  "
          f"({OUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == '__main__':
    main()
