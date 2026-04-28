"""
Build the Markowitz long-only backtest dashboard.

Reads:
  - plots/backtest_xgb_markowitz/equity_qp.csv   : daily NAV / cash / invested
  - plots/backtest_xgb_markowitz/trades_qp.csv   : per-trade buy/sell records
  - plots/backtest_xgb_markowitz/metrics_qp.txt  : human-readable metrics dump
  - stock_data/index/idx_factor_pro/000300_SH.csv : CSI300 benchmark
  - stock_sectors.csv                            : ts_code → name (Chinese)

Writes:
  - dashboard/backtest_data.json
  - dashboard/backtest.html  (single-file page, Plotly via CDN)

Run:
  ./venv/Scripts/python -m dashboard.build_backtest
Then:
  ./venv/Scripts/python -m http.server -d dashboard 8000
and open http://localhost:8000/backtest.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT / 'plots' / 'backtest_xgb_markowitz'
CSI300 = ROOT / 'stock_data' / 'index' / 'idx_factor_pro' / '000300_SH.csv'


def _log(msg: str) -> None:
    print(f"[backtest_dashboard] {msg}", flush=True)


def load_run(tag: str) -> dict:
    suffix = f"_{tag}" if tag else ""
    eq = pd.read_csv(BACKTEST_DIR / f'equity{suffix}.csv', parse_dates=['trade_date'])
    tr = pd.read_csv(BACKTEST_DIR / f'trades{suffix}.csv',
                     parse_dates=['entry_date', 'exit_date'])
    metrics_path = BACKTEST_DIR / f'metrics{suffix}.txt'
    if metrics_path.exists():
        with open(metrics_path, 'rb') as fh:
            metrics_text = fh.read().decode('utf-8', errors='replace')
    else:
        metrics_text = ""
    return {'equity': eq, 'trades': tr, 'metrics_text': metrics_text, 'tag': tag}


def load_csi300(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(CSI300, encoding='utf-8-sig', usecols=['trade_date', 'close'])
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df[(df['trade_date'] >= start - pd.Timedelta(days=10)) &
            (df['trade_date'] <= end + pd.Timedelta(days=10))]
    return df.sort_values('trade_date').reset_index(drop=True)


def load_names() -> dict:
    p = ROOT / 'stock_sectors.csv'
    if not p.exists():
        return {}
    try:
        with open(p, 'rb') as f:
            raw = f.read().decode('utf-8', errors='replace')
        import io
        df = pd.read_csv(io.StringIO(raw), usecols=['ts_code', 'name'])
        return dict(zip(df['ts_code'], df['name']))
    except Exception as e:
        _log(f"warning: could not load names ({e})")
        return {}


def compute_metrics(equity: pd.DataFrame, csi: pd.DataFrame, initial: float) -> dict:
    nav = equity['nav'].to_numpy(dtype=np.float64)
    idx = equity['trade_date']

    csi_a = csi.set_index('trade_date').reindex(idx, method='ffill')
    csi_norm = csi_a['close'] / csi_a['close'].iloc[0] * initial

    ret  = pd.Series(nav, index=idx).pct_change().fillna(0.0)
    bret = csi_a['close'].pct_change().fillna(0.0)

    years = (idx.iloc[-1] - idx.iloc[0]).days / 365.25
    total = nav[-1] / initial - 1.0
    cagr  = (nav[-1] / initial) ** (1.0 / max(years, 1e-6)) - 1.0
    vol   = ret.std(ddof=1) * np.sqrt(252)
    sharpe = (ret.mean() * 252) / (vol + 1e-12)

    roll_max = pd.Series(nav, index=idx).cummax()
    dd = (pd.Series(nav, index=idx) / roll_max - 1.0)
    mdd = float(dd.min())
    mdd_date = dd.idxmin()
    peak = roll_max.loc[:mdd_date].idxmax()

    b_total = float(csi_a['close'].iloc[-1] / csi_a['close'].iloc[0] - 1.0)
    b_cagr  = (csi_a['close'].iloc[-1] / csi_a['close'].iloc[0]) ** (1.0 / max(years, 1e-6)) - 1.0
    b_vol   = bret.std(ddof=1) * np.sqrt(252)
    b_sharpe = (bret.mean() * 252) / (b_vol + 1e-12)
    b_dd = (csi_a['close'] / csi_a['close'].cummax() - 1.0)
    b_mdd = float(b_dd.min())

    cov = np.cov(ret.values, bret.values, ddof=1)
    beta = float(cov[0, 1] / (cov[1, 1] + 1e-12))
    alpha_ann = float((ret.mean() - beta * bret.mean()) * 252)

    active = ret - bret
    ir = float((active.mean() * 252) / (active.std(ddof=1) * np.sqrt(252) + 1e-12))

    return {
        'years':           float(years),
        'total_return':    float(total),
        'cagr':            float(cagr),
        'vol_ann':         float(vol),
        'sharpe':          float(sharpe),
        'max_drawdown':    mdd,
        'mdd_peak':        peak.strftime('%Y-%m-%d'),
        'mdd_trough':      mdd_date.strftime('%Y-%m-%d'),
        'calmar':          float(cagr / abs(mdd)) if mdd < 0 else None,
        'bench_total':     b_total,
        'bench_cagr':      float(b_cagr),
        'bench_vol':       float(b_vol),
        'bench_sharpe':    float(b_sharpe),
        'bench_mdd':       b_mdd,
        'alpha_ann':       alpha_ann,
        'beta':            beta,
        'info_ratio':      ir,
        'final_nav':       float(nav[-1]),
        'initial':         float(initial),
        'date_start':      idx.iloc[0].strftime('%Y-%m-%d'),
        'date_end':        idx.iloc[-1].strftime('%Y-%m-%d'),
    }


def trade_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    wins   = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]
    return {
        'n_trades':         int(len(trades)),
        'hit_rate':         float(len(wins) / len(trades)),
        'avg_win_pct':      float(wins['ret'].mean() * 100) if len(wins)   else 0.0,
        'avg_loss_pct':     float(losses['ret'].mean() * 100) if len(losses) else 0.0,
        'avg_ret_pct':      float(trades['ret'].mean() * 100),
        'median_hold_days': float(trades['held_days'].median()),
        'pct_tp':           float((trades['reason'] == 'take_profit').mean()),
        'pct_sl':           float((trades['reason'] == 'stop_loss').mean()),
        'pct_horizon':      float((trades['reason'] == 'horizon').mean()),
        'profit_factor':    float(wins['pnl'].sum() / abs(losses['pnl'].sum()))
                                if len(losses) and abs(losses['pnl'].sum()) > 0 else float('inf'),
    }


def per_stock_stats(trades: pd.DataFrame, names: dict) -> list:
    g = trades.groupby('ts_code')
    out = g.agg(
        trades=('pnl', 'count'),
        wins=('pnl', lambda x: int((x > 0).sum())),
        total_pnl=('pnl', 'sum'),
        avg_ret=('ret', 'mean'),
        first=('entry_date', 'min'),
        last=('exit_date', 'max'),
    ).reset_index()
    out['hit_rate'] = out['wins'] / out['trades']
    out = out.sort_values('total_pnl', ascending=False)
    return [{
        'ts_code':   r['ts_code'],
        'name':      names.get(r['ts_code'], ''),
        'trades':    int(r['trades']),
        'wins':      int(r['wins']),
        'total_pnl': float(r['total_pnl']),
        'avg_ret':   float(r['avg_ret'] * 100),
        'hit_rate':  float(r['hit_rate']),
        'first':     r['first'].strftime('%Y-%m-%d'),
        'last':      r['last'].strftime('%Y-%m-%d'),
    } for _, r in out.iterrows()]


def top_stock_timelines(trades: pd.DataFrame, names: dict,
                         top_n: int = 20,
                         price_start: str = '2017-01-01') -> list:
    """For each of the top-N stocks (by total P&L), build a long-horizon
    close-price series and the list of buy/sell events with reasons.

    Output (per stock):
        { ts_code, name, total_pnl, n_trades, hit_rate,
          dates: [...], close: [...],
          events: [{date, type: 'buy'|'sell', price, reason, ret_pct, pnl,
                    held_days, paired_date}] }
    """
    g = trades.groupby('ts_code').agg(
        total_pnl=('pnl', 'sum'),
        n=('pnl', 'count'),
        wins=('pnl', lambda x: int((x > 0).sum())),
    ).reset_index()
    g['abs_pnl'] = g['total_pnl'].abs()
    top = g.sort_values('total_pnl', ascending=False).head(top_n)

    start_ts = pd.Timestamp(price_start)
    end_ts = max(trades['exit_date'].max(), pd.Timestamp.today())

    out = []
    for _, r in top.iterrows():
        ts_code = r['ts_code']
        code, suffix = ts_code.split('.')
        sub = 'sh' if suffix.upper() == 'SH' else 'sz'
        fp = ROOT / 'stock_data' / sub / f'{code}.csv'
        if not fp.exists():
            _log(f"  skip {ts_code}: price file missing")
            continue
        try:
            df = pd.read_csv(fp, usecols=['trade_date', 'close'])
        except Exception as e:
            _log(f"  skip {ts_code}: read error {e}")
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df = df[(df['trade_date'] >= start_ts) & (df['trade_date'] <= end_ts)]
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 10:
            continue

        sub_trades = trades[trades['ts_code'] == ts_code].sort_values('entry_date')
        events = []
        for _, tr in sub_trades.iterrows():
            events.append({
                'date':       tr['entry_date'].strftime('%Y-%m-%d'),
                'type':       'buy',
                'price':      float(tr['entry_price']),
                'reason':     'open',
                'ret_pct':    float(tr['ret'] * 100),
                'pnl':        float(tr['pnl']),
                'held_days':  int(tr['held_days']),
                'paired':     tr['exit_date'].strftime('%Y-%m-%d'),
            })
            events.append({
                'date':       tr['exit_date'].strftime('%Y-%m-%d'),
                'type':       'sell',
                'price':      float(tr['exit_price']),
                'reason':     tr['reason'],
                'ret_pct':    float(tr['ret'] * 100),
                'pnl':        float(tr['pnl']),
                'held_days':  int(tr['held_days']),
                'paired':     tr['entry_date'].strftime('%Y-%m-%d'),
            })

        out.append({
            'ts_code':    ts_code,
            'name':       names.get(ts_code, ''),
            'total_pnl':  float(r['total_pnl']),
            'n_trades':   int(r['n']),
            'wins':       int(r['wins']),
            'hit_rate':   float(r['wins'] / r['n']),
            'dates':      [d.strftime('%Y-%m-%d') for d in df['trade_date']],
            'close':      df['close'].astype(float).tolist(),
            'events':     events,
        })
    _log(f"top-{top_n} timelines: {len(out)} stocks "
         f"({sum(len(s['dates']) for s in out):,} price points, "
         f"{sum(len(s['events']) for s in out):,} events)")
    return out


def hold_distribution(trades: pd.DataFrame) -> dict:
    bins = list(range(0, int(trades['held_days'].max()) + 2))
    h, _ = np.histogram(trades['held_days'], bins=bins)
    return {'bins': bins[:-1], 'counts': h.tolist()}


def ret_histogram(trades: pd.DataFrame) -> dict:
    rets = (trades['ret'] * 100).clip(-10, 10).values
    bins = np.linspace(-10, 10, 41)
    h, edges = np.histogram(rets, bins=bins)
    return {
        'bin_edges': [float(e) for e in edges],
        'counts':    h.tolist(),
    }


def reason_by_pred_decile(trades: pd.DataFrame) -> dict:
    if trades.empty or 'pred' not in trades.columns:
        return {}
    df = trades.copy()
    df['pred_decile'] = pd.qcut(df['pred'], 10, labels=False, duplicates='drop') + 1
    pivot = df.pivot_table(index='pred_decile', columns='reason',
                           values='pnl', aggfunc='count', fill_value=0)
    avg_ret = df.groupby('pred_decile')['ret'].mean() * 100
    out = {
        'deciles': pivot.index.tolist(),
        'reasons': pivot.columns.tolist(),
        'counts':  pivot.values.tolist(),
        'avg_ret_pct': avg_ret.tolist(),
    }
    return out


def trade_records(trades: pd.DataFrame, names: dict) -> list:
    df = trades.copy()
    df = df.sort_values(['entry_date', 'ts_code']).reset_index(drop=True)
    out = []
    for _, r in df.iterrows():
        out.append({
            'ts_code':     r['ts_code'],
            'name':        names.get(r['ts_code'], ''),
            'entry_date':  r['entry_date'].strftime('%Y-%m-%d'),
            'exit_date':   r['exit_date'].strftime('%Y-%m-%d'),
            'entry_price': float(r['entry_price']),
            'exit_price':  float(r['exit_price']),
            'shares':      int(r['shares']),
            'cost_basis':  float(r['cost_basis']),
            'proceeds':    float(r['proceeds']),
            'pnl':         float(r['pnl']),
            'ret_pct':     float(r['ret'] * 100),
            'held_days':   int(r['held_days']),
            'reason':      r['reason'],
            'pred':        float(r['pred']),
            'sigma':       float(r['sigma']),
            'weight':      float(r['weight']),
        })
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='qp', help='backtest run tag (default: qp)')
    p.add_argument('--initial', type=float, default=1_000_000.0)
    args = p.parse_args()

    run = load_run(args.tag)
    eq, tr = run['equity'], run['trades']
    _log(f"loaded {len(eq):,} equity rows, {len(tr):,} trades from tag='{args.tag}'")

    csi = load_csi300(eq['trade_date'].min(), eq['trade_date'].max())
    names = load_names()

    metrics = compute_metrics(eq, csi, initial=args.initial)
    tsum    = trade_summary(tr)

    csi_aligned = csi.set_index('trade_date').reindex(eq['trade_date'], method='ffill')
    csi_norm = (csi_aligned['close'] / csi_aligned['close'].iloc[0] * args.initial).tolist()

    payload = {
        'tag':       args.tag,
        'metrics':   metrics,
        'trade_summary': tsum,
        'equity': {
            'dates':    [d.strftime('%Y-%m-%d') for d in eq['trade_date']],
            'nav':      eq['nav'].tolist(),
            'cash':     eq['cash'].tolist(),
            'invested': eq['invested'].tolist(),
            'n_pos':    eq['n_pos'].tolist(),
            'pnl_realized': eq['pnl_realized_day'].tolist(),
            'csi300':   csi_norm,
        },
        'reasons': {
            'take_profit': int((tr['reason'] == 'take_profit').sum()),
            'stop_loss':   int((tr['reason'] == 'stop_loss').sum()),
            'horizon':     int((tr['reason'] == 'horizon').sum()),
        },
        'ret_hist':       ret_histogram(tr),
        'hold_hist':      hold_distribution(tr),
        'pred_deciles':   reason_by_pred_decile(tr),
        'per_stock':      per_stock_stats(tr, names),
        'top_timelines':  top_stock_timelines(tr, names, top_n=20,
                                              price_start='2017-01-01'),
        'trades':         trade_records(tr, names),
        'metrics_text':   run['metrics_text'],
    }

    out_path = OUT / 'backtest_data.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
    _log(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")

    # Self-contained single-file build for Netlify drag-drop.
    # Injects the JSON into backtest.html as window.BACKTEST_DATA, so the page
    # works without a separate fetch (the dev pair backtest.html + .json still
    # works the normal way alongside it).
    src_html = (OUT / 'backtest.html').read_text(encoding='utf-8')
    embedded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    inject = (
        '<script id="backtest-data-embed">\n'
        f'window.BACKTEST_DATA = {embedded};\n'
        '</script>\n'
    )
    needle = '</head>'
    if needle not in src_html:
        raise RuntimeError("backtest.html has no </head> — cannot inject data")
    bundled = src_html.replace(needle, inject + needle, 1)
    bundled_path = OUT / 'index_backtest.html'
    bundled_path.write_text(bundled, encoding='utf-8')
    _log(f"wrote {bundled_path}  ({bundled_path.stat().st_size / 1e6:.2f} MB, single-file Netlify drop)")


if __name__ == '__main__':
    main()
