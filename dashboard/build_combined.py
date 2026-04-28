"""
Build the unified XGB prediction + Markowitz backtest dashboard.

Reads:
  - dashboard/data.json                 : prediction OOS metrics (built by dashboard.build)
  - dashboard/backtest_data.json        : backtest output (built by dashboard.build_backtest)
  - stock_data/models/xgb_pct_chg.meta.json
  - stock_data/models/xgb_pct_chg.features.json
  - dashboard/feature_catalog.py        : feature → (group, meaning_zh, source)
  - plots/backtest_xgb_markowitz/equity_qp.csv + trades_qp.csv  (for Barra)

Writes:
  - dashboard/combined_data.json
  - dashboard/index_combined.html       : single-file (data inlined, Netlify-ready)

Run:
  ./venv/Scripts/python -m dashboard.build_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent

from dashboard.feature_catalog import CATALOG, GROUPS, coverage, lookup
from dashboard.barra_attribution import run_barra, FACTOR_NAMES, FACTOR_LABEL
from dashboard.leakage_audit import build_audit
from dashboard.live_prediction import build_live_payload
from dashboard.st_outcomes import analyze as analyze_st_outcomes


def _log(msg: str) -> None:
    print(f"[combined] {msg}", flush=True)


def load_predictions_json() -> dict:
    p = OUT / 'data.json'
    if not p.exists():
        raise SystemExit(f'Missing {p} — run `dashboard.build` first.')
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_backtest_json() -> dict:
    p = OUT / 'backtest_data.json'
    if not p.exists():
        raise SystemExit(f'Missing {p} — run `dashboard.build_backtest --tag qp` first.')
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_feature_list() -> list:
    p = ROOT / 'stock_data' / 'models' / 'xgb_pct_chg.features.json'
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_full_meta() -> dict:
    p = ROOT / 'stock_data' / 'models' / 'xgb_pct_chg.meta.json'
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_feature_catalog_payload() -> dict:
    feats = load_feature_list()
    cov, missing = coverage(feats)
    if missing:
        _log(f"WARNING: {len(missing)} features missing from catalog: {missing[:5]}...")
    rows = [lookup(f) for f in feats]
    by_group = {g[0]: [] for g in GROUPS}
    for r in rows:
        by_group.setdefault(r['group'], []).append(r['name'])
    return {
        'total':    len(feats),
        'features': rows,
        'groups':   [{'key': g[0], 'name_zh': g[1], 'desc': g[2],
                      'count': len(by_group.get(g[0], []))} for g in GROUPS],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-barra', action='store_true',
                    help='Skip the Barra attribution step (faster reruns)')
    ap.add_argument('--cached-barra', default=None,
                    help='Path to a cached Barra JSON to load instead of recomputing')
    ap.add_argument('--live_preds_csv', default=None,
                    help='Live predictions CSV (default: stock_predictions_xgb.csv)')
    ap.add_argument('--skip-live', action='store_true',
                    help='Skip the live-prediction + portfolio block')
    args = ap.parse_args()

    pred = load_predictions_json()
    bt   = load_backtest_json()
    full_meta = load_full_meta()
    catalog   = build_feature_catalog_payload()

    # Barra attribution
    if args.cached_barra and Path(args.cached_barra).exists():
        with open(args.cached_barra, 'r', encoding='utf-8') as f:
            barra = json.load(f)
        _log(f"loaded Barra attribution from cache: {args.cached_barra}")
    elif args.skip_barra:
        barra = None
        _log("skipping Barra attribution")
    else:
        eq = pd.read_csv(ROOT / 'plots' / 'backtest_xgb_markowitz' / 'equity_qp.csv',
                         parse_dates=['trade_date'])
        tr = pd.read_csv(ROOT / 'plots' / 'backtest_xgb_markowitz' / 'trades_qp.csv',
                         parse_dates=['entry_date', 'exit_date'])
        barra = run_barra(eq, tr)
        # Cache it so future builds can `--cached-barra`
        cache_p = OUT / 'barra_cache.json'
        with open(cache_p, 'w', encoding='utf-8') as f:
            json.dump(barra, f, ensure_ascii=False, separators=(',', ':'))
        _log(f"cached Barra → {cache_p}")

    # Leakage attestation + data freshness inventory
    audit = build_audit()
    _log(f"audit: {len(audit['guarantees'])} guarantees, "
         f"{len(audit['data_freshness']['sources'])} data sources")

    # ST outcome analysis on the historical backtest trades
    st_outcomes = None
    try:
        eq_path = ROOT / 'plots' / 'backtest_xgb_markowitz' / 'trades_qp.csv'
        if eq_path.exists():
            tr_df = pd.read_csv(eq_path, parse_dates=['entry_date', 'exit_date'])
            st_outcomes = analyze_st_outcomes(tr_df)
            _log(f"st_outcomes: {st_outcomes['headline']['st_stocks_traded']:,} ST stocks traded "
                 f"({st_outcomes['headline']['pct_st_pnl']*100:.1f}% of P&L)")
    except FileNotFoundError as e:
        _log(f"WARNING: ST outcome analysis skipped — {e}")
    except Exception as e:
        _log(f"WARNING: ST outcome analysis failed: {e}")

    # Capped-ST backtest comparison (only loaded if it has been run)
    capped_backtest = None
    capped_eq_p = ROOT / 'plots' / 'backtest_xgb_markowitz' / 'equity_qp_st4.csv'
    capped_tr_p = ROOT / 'plots' / 'backtest_xgb_markowitz' / 'trades_qp_st4.csv'
    if capped_eq_p.exists() and capped_tr_p.exists():
        try:
            ce = pd.read_csv(capped_eq_p, parse_dates=['trade_date'])
            ct = pd.read_csv(capped_tr_p, parse_dates=['entry_date', 'exit_date'])
            from dashboard.build_backtest import compute_metrics, load_csi300, trade_summary
            csi = load_csi300(ce['trade_date'].min(), ce['trade_date'].max())
            cmet = compute_metrics(ce, csi, initial=1_000_000.0)
            csum = trade_summary(ct)
            capped_backtest = {
                'tag':         'qp_st4 (max 4 ST per day)',
                'metrics':     cmet,
                'summary':     csum,
                'equity_dates': [d.strftime('%Y-%m-%d') for d in ce['trade_date']],
                'equity_nav':   ce['nav'].tolist(),
                'n_trades':     int(len(ct)),
            }
            _log(f"loaded capped backtest: CAGR={cmet['cagr']*100:+.2f}%  "
                 f"Sharpe={cmet['sharpe']:.2f}  MDD={cmet['max_drawdown']*100:+.2f}%")
        except Exception as e:
            _log(f"WARNING: could not load capped-ST backtest: {e}")

    # Live next-day prediction + Markowitz portfolio.
    # We compute THREE variants:
    #   live           — backtest-faithful (pool=40, exactly mirrors backtest).
    #                    Default for live use. On heavy-limit-stop days this
    #                    correctly recommends cash (matches what the backtest
    #                    would have done — zero entries that day).
    #   live_with_st   — no ST cap (also pool=40), matches the original
    #                    uncapped backtest's profile.
    #   live_expanded  — pool=200 (expanded search), forces K=10 holdings
    #                    by going deeper into the prediction list. Diverges
    #                    from backtest fidelity but useful as a diagnostic
    #                    "what does the model see beyond the limit-stops?"
    live = None
    live_with_st = None
    live_expanded = None
    if not args.skip_live:
        preds_csv = Path(args.live_preds_csv) if args.live_preds_csv \
                    else (ROOT / 'stock_predictions_xgb.csv')
        if preds_csv.exists():
            try:
                # Backtest-faithful: pool = top_k × 4 = 40, max 4 ST/day
                live = build_live_payload(preds_csv, top_k=10,
                                           candidate_pool=40, max_st_per_day=4)
            except Exception as e:
                _log(f"WARNING: live_prediction (backtest-faithful) failed: {e}")
            try:
                # Backtest-faithful: pool = 40, no ST cap
                live_with_st = build_live_payload(preds_csv, top_k=10,
                                                   candidate_pool=40, max_st_per_day=-1)
            except Exception as e:
                _log(f"WARNING: live_prediction (uncapped) failed: {e}")
            try:
                # Expanded: pool = 200 to force K=10 even on heavy-limit days
                live_expanded = build_live_payload(preds_csv, top_k=10,
                                                    candidate_pool=200,
                                                    max_st_per_day=4)
            except Exception as e:
                _log(f"WARNING: live_prediction (expanded) failed: {e}")
        else:
            _log(f"WARNING: {preds_csv} not found — skipping live prediction")

    payload = {
        'predictions': {
            'model_meta':      pred['model_meta'],
            'overall_metrics': pred['overall_metrics'],
            'daily_rank_ic':   pred['daily_rank_ic'],
            'decile':          pred['decile_analysis'],
            'scatter_sample':  pred.get('scatter_sample'),
            'feature_importance_top30': pred.get('feature_importance_top30', []),
            'live_distribution': pred.get('live_distribution'),
            'live_topn':         pred.get('live_topn'),
            'prediction_date':   pred.get('prediction_date'),
            'n_live_stocks':     pred.get('n_live_stocks'),
        },
        'feature_importance_top50': full_meta.get('feature_importance_top50', []),
        'feature_catalog': catalog,
        'backtest':        bt,
        'barra':           barra,
        'xgb_params':      full_meta.get('xgb_params', {}),
        'fold_config':     full_meta.get('fold_config', {}),
        'metric_summary':  full_meta.get('metric_summary', {}),
        'leakage_audit':   audit,
        'live':            live,
        'live_with_st':    live_with_st,
        'live_expanded':   live_expanded,
        'st_outcomes':     st_outcomes,
        'capped_backtest': capped_backtest,
    }

    out_json = OUT / 'combined_data.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
    _log(f"wrote {out_json}  ({out_json.stat().st_size / 1e6:.2f} MB)")

    # ── Self-contained single-file build ───────────────────────────────────
    src_html_p = OUT / 'combined.html'
    if not src_html_p.exists():
        _log(f"WARNING: {src_html_p} not found — skipping single-file bundle")
        return
    src_html = src_html_p.read_text(encoding='utf-8')
    embedded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    inject = (
        '<script id="combined-data-embed">\n'
        f'window.COMBINED_DATA = {embedded};\n'
        '</script>\n'
    )
    needle = '</head>'
    if needle not in src_html:
        raise RuntimeError("combined.html has no </head> — cannot inject data")
    bundled = src_html.replace(needle, inject + needle, 1)
    bundled_p = OUT / 'index_combined.html'
    bundled_p.write_text(bundled, encoding='utf-8')
    _log(f"wrote {bundled_p}  ({bundled_p.stat().st_size / 1e6:.2f} MB, single-file Netlify drop)")


if __name__ == '__main__':
    main()
