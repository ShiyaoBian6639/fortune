"""
Run a Markowitz QP backtest for every model in `MODELS` and tag the outputs
with the engine name. Default backtest parameters are kept identical to the
canonical XGBoost run so comparison is apples-to-apples.

Outputs go to `plots/backtest_xgb_markowitz/{equity,trades,metrics}_<tag>.csv`
with `tag = qp_<engine>`. Existing canonical run uses tag=qp.

Run:
    ./venv/Scripts/python -m model_compare.run_backtests
        # Backtests every completed model with default params

    ./venv/Scripts/python -m model_compare.run_backtests --tune
        # Adds a small TP/SL/max_hold grid per model, picks best by Sharpe,
        # then runs final backtest with those params (saved as qp_<engine>_tuned)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    'xgb_default',
    'xgb_shallow',
    'xgb_deep',
    'xgb_strong_reg',
    'lightgbm',
    'catboost',
    'transformer_reg',
    'ensemble_mean',
    'ensemble_rankavg',
]


def model_preds_csv(name: str) -> Path:
    """Per-model OOF predictions path."""
    return ROOT / 'stock_data' / f'models_{name}' / 'xgb_preds' / 'test.csv'


def run_backtest(name: str, *, tag_suffix: str = '',
                  tp: float = 0.03, sl: float = 0.02, max_hold: int = 5,
                  max_st_per_day: int = 4, solver: str = 'qp',
                  start: str = '2021-04-22', end: str = '2026-04-21',
                  impl_lag: int = 1, entry_price: str = 'open',
                  extra_args: List[str] = None) -> Dict:
    """Run xgb_markowitz with the given model's predictions; return metrics.

    Defaults: impl_lag=1 + entry_price=open — the realistic deployment timing
    where pred(X) computed after close(X) is used to enter at open(X+1) via a
    market-on-open order. Pass impl_lag=0 / entry_price='close' to reproduce
    legacy zero-lag results (optimistic; used for prior pipeline runs).
    """
    preds_csv = model_preds_csv(name)
    if not preds_csv.exists():
        print(f"[skip] {name}: predictions file missing ({preds_csv})")
        return None
    tag = f"{solver}_{name}{tag_suffix}"
    cmd = [
        str(ROOT / 'venv' / 'Scripts' / 'python'),
        '-m', 'backtest.xgb_markowitz',
        '--solver',          solver,
        '--max_st_per_day',  str(max_st_per_day),
        '--preds_csv',       str(preds_csv),
        '--tag',             tag,
        '--start',           start,
        '--end',             end,
        '--tp',              str(tp),
        '--sl',              str(sl),
        '--max_hold',        str(max_hold),
        '--impl_lag',        str(impl_lag),
        '--entry_price',     entry_price,
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n[bt] {name}{tag_suffix}: tp={tp} sl={sl} max_hold={max_hold} ...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[bt] {name} FAILED: {r.stderr[-400:]}")
        return None
    print(f"[bt] {name} done in {dt:.0f}s")
    metrics = parse_metrics(ROOT / 'plots' / 'backtest_xgb_markowitz' / f'metrics_{tag}.txt')
    if metrics is None:
        return None
    metrics['name']    = name
    metrics['tag']     = tag
    metrics['tp']      = tp
    metrics['sl']      = sl
    metrics['max_hold'] = max_hold
    metrics['runtime_s'] = dt
    return metrics


def parse_metrics(metrics_path: Path) -> Dict:
    """Cheap parser of the human-readable metrics dump produced by xgb_markowitz."""
    if not metrics_path.exists():
        return None
    txt = metrics_path.read_text(encoding='utf-8', errors='replace')
    out = {}
    def _grab(key, line_marker, idx=-1, dtype=float, suffix=''):
        for line in txt.splitlines():
            if line_marker in line:
                tok = line.replace(suffix, '').split()[idx]
                try:
                    out[key] = dtype(tok.rstrip('%'))
                except Exception:
                    pass
                return
    _grab('cagr',         'CAGR ',          idx=-1)
    _grab('sharpe',       'Sharpe (rf=0)',  idx=-1)
    _grab('mdd',          'max drawdown',   idx=-1)
    _grab('vol_ann',      'vol (ann)',      idx=-1)
    _grab('alpha_ann',    'alpha (ann)',    idx=-1)
    _grab('beta',         'beta',           idx=-1)
    _grab('info_ratio',   'info ratio',     idx=-1)
    _grab('total_return', 'total return',   idx=-1)
    _grab('final_nav',    'final NAV',      idx=-1, dtype=lambda s: float(s.replace(',','')))
    _grab('n_trades',     'n_trades',       idx=-1, dtype=lambda s: int(s.replace(',','')))
    _grab('hit_rate',     'hit rate',       idx=-1)
    _grab('avg_win_pct',  'avg win',        idx=-1)
    _grab('avg_loss_pct', 'avg loss',       idx=-1)
    return out


# ─── TP/SL/max_hold grid tuning ─────────────────────────────────────────────
TUNE_GRID = [
    (tp, sl, mh)
    for tp in (0.02, 0.03, 0.04)
    for sl in (0.015, 0.02, 0.025)
    for mh in (3, 5, 7)
]


def tune_model(name: str, baseline: Dict) -> Dict:
    """Grid-search TP/SL/max_hold for one model. Uses diag solver (~10× faster
    than qp) and a smaller window for speed; final winner re-run with full
    qp solver elsewhere."""
    print(f"\n[tune] {name}: scanning {len(TUNE_GRID)} (tp, sl, max_hold) combos via diag solver ...")
    best = baseline.copy() if baseline else None
    best_sharpe = best['sharpe'] if best else float('-inf')
    best_params = (best.get('tp', 0.03), best.get('sl', 0.02), best.get('max_hold', 5)) if best else (0.03, 0.02, 5)

    runs = []
    for (tp, sl, mh) in TUNE_GRID:
        m = run_backtest(name, tag_suffix=f'_tune_{int(tp*1000)}_{int(sl*1000)}_{mh}',
                          tp=tp, sl=sl, max_hold=mh, solver='diag',
                          # Use shorter tuning window — last 24 months
                          start='2024-04-22', end='2026-04-21')
        if m is None:
            continue
        m['tp'], m['sl'], m['max_hold'] = tp, sl, mh
        runs.append(m)
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_params = (tp, sl, mh)
    print(f"[tune] {name}: best (Sharpe={best_sharpe:.2f}) → "
          f"tp={best_params[0]} sl={best_params[1]} max_hold={best_params[2]}")
    return {'best_params': best_params, 'best_sharpe': best_sharpe, 'all_runs': runs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tune', action='store_true', help='per-model param tuning')
    p.add_argument('--solver', default='qp')
    p.add_argument('--max_st_per_day', type=int, default=4)
    p.add_argument('--models', nargs='+', default=None,
                   help='Subset of models (default = all that have predictions)')
    args = p.parse_args()

    pool = args.models or MODELS

    # ── Phase 1: baseline backtest per model ──
    baselines = []
    for name in pool:
        r = run_backtest(name, solver=args.solver, max_st_per_day=args.max_st_per_day)
        if r:
            baselines.append(r)

    print('\n=== BASELINE backtest table (default params: tp=3%, sl=2%, max_hold=5) ===')
    cols = ['name','cagr','sharpe','mdd','alpha_ann','info_ratio','n_trades','hit_rate','final_nav']
    print(f"{'name':22} {'CAGR%':>8} {'Sharpe':>7} {'MDD%':>7} {'α%':>8} {'IR':>6} {'trades':>7} {'hit%':>6}")
    for r in sorted(baselines, key=lambda x: x.get('sharpe', -999), reverse=True):
        print(f"{r['name']:22} {r.get('cagr', float('nan')):>+8.2f} "
              f"{r.get('sharpe', float('nan')):>7.2f} {r.get('mdd', float('nan')):>+7.2f} "
              f"{r.get('alpha_ann', float('nan')):>+8.2f} {r.get('info_ratio', float('nan')):>6.2f} "
              f"{r.get('n_trades', 0):>7,} {r.get('hit_rate', 0):>6.2f}")

    # ── Phase 2: per-model tuning (optional) ──
    tuned = []
    if args.tune:
        for r in baselines:
            t = tune_model(r['name'], r)
            tp, sl, mh = t['best_params']
            # Run the tuned config with full QP solver on the 5-year window
            final = run_backtest(r['name'], tag_suffix='_tuned',
                                  tp=tp, sl=sl, max_hold=mh,
                                  solver=args.solver,
                                  max_st_per_day=args.max_st_per_day)
            if final:
                final['baseline_sharpe'] = r.get('sharpe')
                final['tune_runs']       = len(t['all_runs'])
                tuned.append(final)

        print('\n=== TUNED backtest table (per-model best params) ===')
        print(f"{'name':22} {'tp':>5} {'sl':>5} {'mh':>3} {'CAGR%':>8} {'Sharpe':>7} {'MDD%':>7} {'Δ Sharpe':>9}")
        for r in sorted(tuned, key=lambda x: x.get('sharpe', -999), reverse=True):
            d_sharpe = (r.get('sharpe', 0) - r.get('baseline_sharpe', 0))
            print(f"{r['name']:22} {r['tp']:>5.3f} {r['sl']:>5.3f} {r['max_hold']:>3} "
                  f"{r.get('cagr', 0):>+8.2f} {r.get('sharpe', 0):>7.2f} "
                  f"{r.get('mdd', 0):>+7.2f} {d_sharpe:>+9.2f}")

    # Persist comparison artefacts — MERGE with existing file so the watcher
    # can call us per-engine without losing prior results.
    out_path = ROOT / 'stock_data' / 'models_backtest_comparison.json'
    existing = {'baselines': [], 'tuned': []}
    if out_path.exists():
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            pass
    # Merge: replace prior entries for the same `name`, append new ones
    def _merge(prior, fresh):
        prior_by_name = {r['name']: r for r in prior if r.get('name')}
        for r in fresh:
            prior_by_name[r['name']] = r
        return list(prior_by_name.values())
    merged = {
        'baselines': _merge(existing.get('baselines', []), baselines),
        'tuned':     _merge(existing.get('tuned',     []), tuned),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n[bt] wrote {out_path}  ({len(merged['baselines'])} baselines, "
          f"{len(merged['tuned'])} tuned)")


if __name__ == '__main__':
    main()
