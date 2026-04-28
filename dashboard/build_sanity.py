"""
Build the data-sanity dashboard.

This is the auditor's view: data freshness, leakage guarantees, prediction
archive diff, ST roster summary, and structural invariants. The intent is
"if anything in the pipeline silently goes stale or breaks, the sanity
dashboard should make it obvious."

Reads:
  - stock_data/* (all data sources)
  - stock_predictions_xgb_features_*.csv (consecutive-day prediction archives)
  - stock_data/st_history.csv
  - dashboard/feature_catalog.py (coverage check)
  - stock_data/models/xgb_pct_chg.{features,meta}.json
  - plots/backtest_xgb_markowitz/equity_qp.csv + trades_qp.csv

Writes:
  - dashboard/sanity_data.json
  - dashboard/index_sanity.html (single-file)

Run:
  ./venv/Scripts/python -m dashboard.build_sanity
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent

from dashboard.feature_catalog  import CATALOG, GROUPS, coverage
from dashboard.leakage_audit    import build_audit


def _log(msg: str) -> None:
    print(f"[sanity] {msg}", flush=True)


def archive_diff() -> dict:
    """Compare consecutive prediction archives to verify the model isn't stuck."""
    archives = sorted(glob.glob(str(ROOT / 'stock_predictions_xgb_features_*.csv')))
    rows = []
    for i in range(1, len(archives)):
        a_p, b_p = archives[i-1], archives[i]
        try:
            a = pd.read_csv(a_p, usecols=['ts_code', 'pred_pct_chg_next']).set_index('ts_code')
            b = pd.read_csv(b_p, usecols=['ts_code', 'pred_pct_chg_next']).set_index('ts_code')
        except Exception:
            continue
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        diff = (a.loc[sorted(common), 'pred_pct_chg_next']
                - b.loc[sorted(common), 'pred_pct_chg_next']).abs()
        identical = bool((diff < 1e-9).all())
        rows.append({
            'prev_file':  os.path.basename(a_p),
            'next_file':  os.path.basename(b_p),
            'common':     int(len(common)),
            'differ':     int((diff > 1e-6).sum()),
            'mean_abs':   float(diff.mean()),
            'max_abs':    float(diff.max()),
            'identical':  identical,
            'identical_warning': identical,
        })
    return {'archives': [os.path.basename(a) for a in archives], 'pairs': rows}


def st_roster_summary() -> dict:
    p = ROOT / 'stock_data' / 'st_history.csv'
    if not p.exists():
        return {'present': False}
    df = pd.read_csv(p, dtype={'start_date': str, 'end_date': str})
    out = {
        'present':            True,
        'rows':               int(len(df)),
        'unique_stocks':      int(df['ts_code'].nunique()),
        'ever_starST':        int(df[df['st_kind'] == '*ST']['ts_code'].nunique()),
        'ever_ST':            int(df[df['st_kind'] == 'ST']['ts_code'].nunique()),
        'currently_active':   int((df['end_date'].fillna('') == '').sum()),
        'oldest_event':       df['start_date'].min(),
        'newest_event':       df['start_date'].max(),
    }
    return out


def feature_coverage_check() -> dict:
    feats_p = ROOT / 'stock_data' / 'models' / 'xgb_pct_chg.features.json'
    if not feats_p.exists():
        return {'present': False}
    with open(feats_p, 'r', encoding='utf-8') as f:
        feats = json.load(f)
    cov, missing = coverage(feats)
    return {
        'present':       True,
        'model_n':       len(feats),
        'catalog_total': len(CATALOG),
        'covered':       cov,
        'missing':       missing,
        'group_count':   len(GROUPS),
    }


def backtest_invariants() -> dict:
    """Check that the historical backtest's recorded trades are internally
    consistent with the equity curve."""
    eq_p = ROOT / 'plots' / 'backtest_xgb_markowitz' / 'equity_qp.csv'
    tr_p = ROOT / 'plots' / 'backtest_xgb_markowitz' / 'trades_qp.csv'
    out = {'present': eq_p.exists() and tr_p.exists()}
    if not out['present']:
        return out
    eq = pd.read_csv(eq_p, parse_dates=['trade_date'])
    tr = pd.read_csv(tr_p, parse_dates=['entry_date', 'exit_date'])
    out['n_trading_days']    = int(len(eq))
    out['n_trades']          = int(len(tr))
    out['unique_stocks']     = int(tr['ts_code'].nunique())
    out['date_start']        = eq['trade_date'].min().strftime('%Y-%m-%d')
    out['date_end']          = eq['trade_date'].max().strftime('%Y-%m-%d')
    out['initial_nav']       = float(eq['nav'].iloc[0])
    out['final_nav']         = float(eq['nav'].iloc[-1])
    # Sum of realised P&L should equal final − initial − unrealised; cash drag
    # makes a strict equality difficult, but realised P&L should be in same OOM
    realized = float(eq['pnl_realized_day'].sum())
    out['sum_realized_pnl']  = realized
    out['nav_change']        = float(eq['nav'].iloc[-1] - eq['nav'].iloc[0])
    out['realised_pct_of_nav_change'] = realized / max(1.0, eq['nav'].iloc[-1] - eq['nav'].iloc[0])
    # No trade should have entry_date > exit_date
    out['n_invalid_dates'] = int((tr['entry_date'] > tr['exit_date']).sum())
    # Cash never goes negative
    out['min_cash']  = float(eq['cash'].min())
    out['cash_negative_days'] = int((eq['cash'] < 0).sum())
    return out


def prediction_freshness() -> dict:
    """Compare prediction's feature_date against the latest data on disk."""
    pred_p = ROOT / 'stock_predictions_xgb.csv'
    if not pred_p.exists():
        return {'present': False}
    df = pd.read_csv(pred_p)
    pred_date = pd.to_datetime(df['trade_date'].astype(str)).max()
    # latest stock OHLCV date (sample 30 stocks)
    latest = []
    for sub in ('sh', 'sz'):
        for fp in list((ROOT / 'stock_data' / sub).glob('*.csv'))[:30]:
            try:
                d = pd.read_csv(fp, usecols=['trade_date'])
                latest.append(int(str(d['trade_date'].astype(str).max())))
            except Exception:
                continue
    disk_max = pd.Timestamp(str(max(latest))) if latest else None
    return {
        'present':       True,
        'pred_feature_date':   pred_date.strftime('%Y-%m-%d'),
        'disk_latest_date':    disk_max.strftime('%Y-%m-%d') if disk_max is not None else 'unknown',
        'pred_n_stocks':       int(len(df)),
        'is_stale':            bool(disk_max is not None and disk_max > pred_date),
        'staleness_days':      int((disk_max - pred_date).days) if disk_max is not None else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    audit       = build_audit()
    archives    = archive_diff()
    st_summary  = st_roster_summary()
    feat_cov    = feature_coverage_check()
    bt_inv      = backtest_invariants()
    fresh       = prediction_freshness()

    payload = {
        'leakage_audit':       audit,
        'archive_diffs':       archives,
        'st_roster':           st_summary,
        'feature_coverage':    feat_cov,
        'backtest_invariants': bt_inv,
        'prediction_freshness': fresh,
    }

    out_json = OUT / 'sanity_data.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
    _log(f"wrote {out_json}  ({out_json.stat().st_size / 1024:.1f} KB)")

    # Single-file embed
    src = OUT / 'sanity.html'
    if not src.exists():
        _log(f"WARNING {src} missing")
        return
    html = src.read_text(encoding='utf-8')
    embedded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    inject = f'<script id="sanity-data-embed">\nwindow.SANITY_DATA = {embedded};\n</script>\n'
    bundled = html.replace('</head>', inject + '</head>', 1)
    bundled_p = OUT / 'index_sanity.html'
    bundled_p.write_text(bundled, encoding='utf-8')
    _log(f"wrote {bundled_p}  ({bundled_p.stat().st_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
