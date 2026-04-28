"""
Filter the A-share universe to a high-quality subset for the predictor.

Filters applied (cumulative — each AND'd):
  1. Listing age      ≥ 5 years (eliminates IPOs and thin-history names)
  2. Never ST/*ST     since 2017 (no special-treatment risk)
  3. Limit-down count ≤ 3/year on average over the last 5 years
                      (low tail-risk; stocks that hit -10% rarely)
  4. 20-day ADV       ≥ 50M CNY median (liquid enough for top-K trading)
  5. 250-day SMA slope > 0 over the last 250 days (uptrending names)
  6. Not a delisted stock (still actively listed)

Outputs `stock_data/stock_subset.csv` with columns:
  ts_code, name, listed_years, limit_down_per_year, median_adv_50m,
  sma_slope_250d, in_subset

Run:
    ./venv/Scripts/python -m backtest.select_subset
    ./venv/Scripts/python -m backtest.select_subset --min_adv_cny 30000000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'


def _log(msg): print(f"[subset] {msg}", flush=True)


def load_stock_basic() -> pd.DataFrame:
    """Listing dates + active status from stock_basic_status.csv."""
    fp = DATA / 'stock_basic_status.csv'
    if not fp.exists():
        # fallback: scan sh/ + sz/ for codes
        codes = []
        for sub in ('sh', 'sz'):
            for fp2 in (DATA / sub).glob('*.csv'):
                codes.append(fp2.stem)
        return pd.DataFrame({'ts_code': [c + '.SH' if c.startswith('6') else c + '.SZ'
                                          for c in codes],
                             'list_date': '20100101', 'list_status': 'L'})
    df = pd.read_csv(fp, dtype=str, encoding='utf-8-sig')
    return df


def load_st_roster() -> set:
    """Set of ts_codes ever flagged as ST/*ST since 2017."""
    fp = DATA / 'st_history.csv'
    if not fp.exists():
        return set()
    df = pd.read_csv(fp, encoding='utf-8-sig', dtype={'start_date': str, 'end_date': str})
    df = df[df['start_date'] >= '20170101']
    return set(df['ts_code'].unique())


def compute_per_stock_stats(ts_code: str) -> dict:
    """Compute filter stats for one stock from its OHLC CSV."""
    sub = 'sh' if ts_code.endswith('.SH') else 'sz'
    code = ts_code.split('.')[0]
    fp = DATA / sub / f'{code}.csv'
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp,
                          usecols=['trade_date','close','pct_chg','amount','pre_close'])
    except Exception:
        return None
    if len(df) < 30:
        return None
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['amount_cny'] = df['amount'].astype(float) * 1000.0   # tushare 千元 → 元

    last_5y_cutoff = df['trade_date'].max() - pd.Timedelta(days=365 * 5)
    last_250 = df.tail(250)
    last_5y  = df[df['trade_date'] >= last_5y_cutoff]

    # 1. Listing age (rough — first trade_date)
    listed_years = (df['trade_date'].max() - df['trade_date'].min()).days / 365.25

    # 3. Limit-down count: pct_chg <= -9.7 (also covers ST 4.85% via threshold)
    if len(last_5y) > 0:
        ld_count    = int((last_5y['pct_chg'] <= -9.7).sum())
        years_in_5y = max((last_5y['trade_date'].max()
                           - last_5y['trade_date'].min()).days / 365.25, 0.5)
        ld_per_year = ld_count / years_in_5y
    else:
        ld_per_year = 99.0

    # 4. 20-day ADV (median over last 5y)
    adv_20 = df['amount_cny'].rolling(20, min_periods=10).mean()
    median_adv = float(adv_20.tail(min(len(adv_20), 1250)).median())   # ~5y of trading days

    # 5. 250-day SMA slope (over last 250 days)
    if len(last_250) >= 250:
        x = np.arange(len(last_250))
        y = last_250['close'].values
        slope, _ = np.polyfit(x, y, 1)
        sma_slope_pct = slope / y.mean() * 100   # daily slope as % of mean price
    else:
        sma_slope_pct = float('nan')

    return {
        'ts_code':              ts_code,
        'listed_years':         round(listed_years, 2),
        'limit_down_per_year':  round(ld_per_year, 2),
        'median_adv_cny':       round(median_adv, 0),
        'sma_slope_pct':        round(sma_slope_pct, 4)
                                if np.isfinite(sma_slope_pct) else None,
        'last_close':           round(float(df['close'].iloc[-1]), 2),
        'first_date':           df['trade_date'].iloc[0].strftime('%Y-%m-%d'),
        'last_date':            df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_listed_years', type=float, default=5.0)
    p.add_argument('--max_limit_down_per_year', type=float, default=3.0)
    p.add_argument('--min_adv_cny',  type=float, default=5e7,
                   help='minimum 20d ADV (CNY) median over last 5y. Default 50M.')
    p.add_argument('--min_slope_pct', type=float, default=0.0,
                   help='minimum 250d SMA slope as %% of mean price per day. '
                        '0 = uptrending or flat.')
    p.add_argument('--require_active', action='store_true', default=True,
                   help='exclude delisted stocks.')
    p.add_argument('--out', default='stock_data/stock_subset.csv')
    args = p.parse_args()

    sb = load_stock_basic()
    st_set = load_st_roster()
    _log(f"stock_basic: {len(sb):,} stocks  ST roster: {len(st_set):,}")

    # Active set — exclude delisted
    if args.require_active:
        sb = sb[sb.get('list_status', 'L').isin(['L'])] if 'list_status' in sb.columns else sb
    candidates = sorted(sb['ts_code'].unique())
    _log(f"computing per-stock stats for {len(candidates):,} candidates ...")

    rows = []
    for i, ts in enumerate(candidates, 1):
        if i % 500 == 0:
            _log(f"  processed {i}/{len(candidates)} ...")
        st = compute_per_stock_stats(ts)
        if st is None: continue
        st['ever_st'] = ts in st_set
        rows.append(st)

    df = pd.DataFrame(rows)
    df = df.merge(sb[['ts_code','name']] if 'name' in sb.columns else
                   pd.DataFrame({'ts_code': df['ts_code'], 'name': ''}),
                   on='ts_code', how='left')

    # Apply filters cumulatively
    df['filter_listed_years'] = df['listed_years']      >= args.min_listed_years
    df['filter_no_st']        = ~df['ever_st']
    df['filter_low_ld']       = df['limit_down_per_year'] <= args.max_limit_down_per_year
    df['filter_liquid']       = df['median_adv_cny']    >= args.min_adv_cny
    df['filter_uptrend']      = (df['sma_slope_pct'].fillna(-9999) > args.min_slope_pct)
    df['in_subset'] = (
        df['filter_listed_years']
        & df['filter_no_st']
        & df['filter_low_ld']
        & df['filter_liquid']
        & df['filter_uptrend']
    )

    out_p = Path(args.out)
    df.to_csv(out_p, index=False, encoding='utf-8-sig')

    print()
    print("=" * 72)
    print(f"SUBSET FILTER  (n={len(df):,} stocks scanned)")
    print("=" * 72)
    print(f"  pass listed≥{args.min_listed_years}y     :  {df['filter_listed_years'].sum():>5,}")
    print(f"  pass never-ST            :  {df['filter_no_st'].sum():>5,}")
    print(f"  pass ld≤{args.max_limit_down_per_year}/yr            :  {df['filter_low_ld'].sum():>5,}")
    print(f"  pass ADV≥{args.min_adv_cny/1e6:.0f}M CNY     :  {df['filter_liquid'].sum():>5,}")
    print(f"  pass SMA slope > {args.min_slope_pct:.2f}    :  {df['filter_uptrend'].sum():>5,}")
    print(f"  pass ALL (in subset)     :  {df['in_subset'].sum():>5,}  ⭐")
    print()
    print(f"  saved to {out_p}")
    print()
    if df['in_subset'].sum() > 0:
        sub = df[df['in_subset']].nlargest(10, 'median_adv_cny')
        print("Top-10 by ADV in subset:")
        for _, r in sub.iterrows():
            print(f"    {r['ts_code']}  {r.get('name','') or '':6}  "
                  f"ADV ¥{r['median_adv_cny']/1e8:.1f}亿  "
                  f"slope {r['sma_slope_pct']:+.4f}%/d  "
                  f"ld/yr {r['limit_down_per_year']:.1f}")


if __name__ == '__main__':
    main()
