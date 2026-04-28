"""
ST stock outcome analysis: of every ST/*ST stock that the strategy ever traded,
classify its current status and report aggregate statistics for the dashboard.

Outcome buckets:
  delisted   — current list_status='D' (退市)
  paused     — current list_status='P' (停牌/暂停上市)
  recovered  — currently listed with no active ST/*ST flag (摘帽)
  still_st   — currently listed and current name still starts with ST/*ST
  unknown    — ts_code not found in stock_basic or st_history (data gap)

The analysis answers: "If the strategy entered an ST stock, what eventually
happened to it? Did it get delisted (worst case for our P&L assumption) or
get the ST tag removed (best case)?"

Output: a dict embedded into the combined dashboard payload.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'


def _log(msg: str) -> None:
    print(f"[st_outcomes] {msg}", flush=True)


def _load_st_roster() -> pd.DataFrame:
    p = DATA_DIR / 'st_history.csv'
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run `python -m api.st_history --download` first.")
    df = pd.read_csv(p, dtype={'start_date': str, 'end_date': str, 'ann_date': str})
    df['start_date'] = df['start_date'].fillna('')
    df['end_date']   = df['end_date'].fillna('')
    return df


def _load_stock_basic() -> pd.DataFrame:
    """Authoritative current-status table: ts_code, name, list_status.

    Caches at stock_data/stock_basic_status.csv. Refresh by deleting that file
    and rerunning. Pulls L/P/D rosters from tushare and merges them.
    """
    cache = DATA_DIR / 'stock_basic_status.csv'
    if cache.exists():
        df = pd.read_csv(cache, encoding='utf-8-sig')
        return df[['ts_code', 'name', 'list_status']]

    _log("fetching stock_basic from tushare for L/P/D status (one-time, cached) ...")
    import tushare as ts
    TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
    ts.set_token(TOKEN)
    pro = ts.pro_api(TOKEN)
    rows = []
    for status in ('L', 'P', 'D'):
        try:
            df = pro.stock_basic(exchange='', list_status=status,
                                  fields='ts_code,name,list_status,delist_date')
            if df is not None and not df.empty:
                rows.append(df)
                _log(f"  status={status}: {len(df):,} stocks")
        except Exception as e:
            _log(f"  status={status} failed: {e}")
    if not rows:
        return pd.DataFrame(columns=['ts_code', 'name', 'list_status'])
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(cache, index=False, encoding='utf-8-sig')
    _log(f"cached {cache}  ({len(out):,} rows)")
    return out[['ts_code', 'name', 'list_status']]


def _is_st_now(name: str) -> bool:
    if not isinstance(name, str):
        return False
    n = name.strip()
    return n.startswith('*ST') or n.startswith('ST') or n.startswith('S*ST')


def _classify_outcome(ts_code: str, basic_lookup: dict, roster: pd.DataFrame,
                      as_of: str) -> dict:
    """Determine the current status of `ts_code` (was-ST or recovered/delisted).

    Returns: {'ts_code', 'outcome', 'current_name', 'list_status',
              'first_st_date', 'last_st_end'}
    """
    info = basic_lookup.get(ts_code)
    sub  = roster[roster['ts_code'] == ts_code]
    first_st = sub['start_date'].min() if not sub.empty else ''
    # Latest interval — find row with empty end_date or latest end_date
    latest_end = ''
    if not sub.empty:
        active = sub[sub['end_date'] == '']
        if not active.empty:
            latest_end = ''   # currently ST
        else:
            latest_end = sub['end_date'].max()

    if info is None:
        return {'ts_code': ts_code, 'outcome': 'unknown',
                'current_name': '', 'list_status': '?',
                'first_st_date': first_st, 'last_st_end': latest_end}

    name   = info.get('name', '') or ''
    status = info.get('list_status', 'L')

    if status == 'D':
        outcome = 'delisted'
    elif status == 'P':
        outcome = 'paused'
    elif _is_st_now(name):
        # Currently listed but still starts with ST/*ST
        outcome = 'still_st'
    else:
        # Listed and name no longer ST — successful "摘帽"
        outcome = 'recovered'

    return {'ts_code': ts_code, 'outcome': outcome,
            'current_name': name, 'list_status': status,
            'first_st_date': first_st, 'last_st_end': latest_end}


def analyze(trades: pd.DataFrame, as_of: Optional[str] = None) -> dict:
    """Build outcome statistics for ST stocks the strategy ever entered.

    Parameters
    ----------
    trades : pd.DataFrame
        Per-trade records from the backtest (must have ts_code, entry_date).
    as_of : str
        YYYYMMDD; defaults to today.
    """
    roster = _load_st_roster()
    basic  = _load_stock_basic()
    basic_lookup = basic.set_index('ts_code').to_dict('index')

    # Build interval index for fast ST-at-trade-time test
    interval_index = {}
    for ts_code, g in roster.groupby('ts_code'):
        intervals = []
        for _, r in g.iterrows():
            if r['start_date']:
                intervals.append((r['start_date'], r['end_date'] or '99999999', r['st_kind']))
        interval_index[ts_code] = intervals

    def _was_st_at(ts_code: str, date_str: str) -> bool:
        for s, e, _ in interval_index.get(ts_code, ()):
            if s <= date_str <= e:
                return True
        return False

    # Tag every trade with ST-at-trade-time flag
    tr = trades.copy()
    tr['entry_str'] = pd.to_datetime(tr['entry_date']).dt.strftime('%Y%m%d')
    tr['was_st']   = [_was_st_at(t, d) for t, d in zip(tr['ts_code'], tr['entry_str'])]

    # Aggregate at the stock level (each unique stock the strategy entered as ST)
    st_trades = tr[tr['was_st']].copy()
    st_stocks = st_trades.groupby('ts_code').agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        wins=('pnl', lambda x: int((x > 0).sum())),
        first_entry=('entry_date', 'min'),
        last_entry=('entry_date', 'max'),
    ).reset_index()
    st_stocks['hit_rate'] = st_stocks['wins'] / st_stocks['trades']

    # Classify each
    if as_of is None:
        as_of = pd.Timestamp.now().strftime('%Y%m%d')

    rows = []
    for _, r in st_stocks.iterrows():
        out = _classify_outcome(r['ts_code'], basic_lookup, roster, as_of)
        rows.append({**r.to_dict(),
                     'outcome': out['outcome'],
                     'current_name': out['current_name'],
                     'list_status': out['list_status'],
                     'first_st_date': out['first_st_date'],
                     'last_st_end': out['last_st_end']})
    detail = pd.DataFrame(rows)

    # Aggregate counts + pnl by outcome
    summary = (detail.groupby('outcome')
                     .agg(stocks=('ts_code', 'nunique'),
                          trades=('trades', 'sum'),
                          total_pnl=('total_pnl', 'sum'))
                     .reset_index().sort_values('stocks', ascending=False))

    # Headline numbers
    headline = {
        'st_stocks_traded':  int(detail['ts_code'].nunique()),
        'st_trades':         int(st_trades.shape[0]),
        'st_pnl':            float(st_trades['pnl'].sum()),
        'all_trades':        int(len(tr)),
        'all_pnl':           float(tr['pnl'].sum()),
        'pct_st_trades':     float(st_trades.shape[0] / max(len(tr), 1)),
        'pct_st_pnl':        float(st_trades['pnl'].sum() / max(tr['pnl'].sum(), 1e-9)),
    }
    _log(f"ST stocks ever traded: {headline['st_stocks_traded']:,}  "
         f"({headline['pct_st_trades']*100:.1f}% of trades, "
         f"{headline['pct_st_pnl']*100:.1f}% of P&L)")
    for _, row in summary.iterrows():
        _log(f"  {row['outcome']:10s} : {int(row['stocks']):4d} stocks, "
             f"{int(row['trades']):5d} trades, P&L = {row['total_pnl']:>14,.0f}")

    return {
        'headline': headline,
        'as_of':    as_of,
        'by_outcome': [
            {'outcome': r['outcome'],
             'stocks': int(r['stocks']),
             'trades': int(r['trades']),
             'total_pnl': float(r['total_pnl'])}
            for _, r in summary.iterrows()
        ],
        # Top contributors by P&L for the dashboard table
        'top_contributors': [
            {'ts_code': r['ts_code'], 'current_name': r['current_name'],
             'outcome': r['outcome'], 'list_status': r['list_status'],
             'trades': int(r['trades']), 'hit_rate': float(r['hit_rate']),
             'total_pnl': float(r['total_pnl']),
             'first_entry': pd.Timestamp(r['first_entry']).strftime('%Y-%m-%d'),
             'last_entry':  pd.Timestamp(r['last_entry']).strftime('%Y-%m-%d'),
             'first_st_date': r['first_st_date'],
             'last_st_end':   r['last_st_end']}
            for _, r in detail.sort_values('total_pnl', ascending=False).head(50).iterrows()
        ],
    }


if __name__ == '__main__':
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument('--trades', default='plots/backtest_xgb_markowitz/trades_qp.csv')
    args = p.parse_args()
    trades = pd.read_csv(args.trades, parse_dates=['entry_date', 'exit_date'])
    out = analyze(trades)
    print(json.dumps(out['headline'], ensure_ascii=False, indent=2))
    print(json.dumps(out['by_outcome'], ensure_ascii=False, indent=2))
