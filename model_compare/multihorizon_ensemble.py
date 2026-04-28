"""
Combine multi-horizon predictions across engines into a single OOF table.

Reads each engine's per-(ts_code, trade_date) test predictions from
`stock_data/models_<engine>_mh_<basis>/xgb_preds/test.csv` and produces:

  stock_data/models_ensemble_mh_<basis>/xgb_preds/test.csv

with columns:  ts_code, trade_date, pred (= mean of pred_d1 across engines),
              pred_d1..pred_d5 (cross-engine simple mean for each horizon),
              target (= target_d1, for backtest compatibility).

The "pred" column is the d1 mean — what `backtest/xgb_rebalance.py` consumes.
The pred_d1..d5 columns are kept so the dashboard can show the multi-horizon
ensemble trajectory.

Engines auto-detected (any that have models_*_mh_<basis>/xgb_preds/test.csv):
  • xgb_default_d1_oc / lightgbm_d1_oc / catboost_d1_oc  (5 horizon-specific)
  • transformer_mh_oc                  (multi-horizon NN)
  • tft_paper_mh_oc                    (multi-horizon NN)

Run:
    ./venv/Scripts/python -m model_compare.multihorizon_ensemble --basis oc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as sst

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'

HORIZONS = (1, 2, 3, 4, 5)


def _log(m): print(f"[ens-mh] {m}", flush=True)


# Each entry: (engine_name_in_output, source_pattern, kind)
#   kind = 'multi_horizon'  → file already has pred_d1..pred_d5
#   kind = 'single_horizon' → file is one of 5 per-horizon files
ENGINE_SOURCES = [
    ('transformer_mh', 'models_transformer_mh_{basis}/xgb_preds/test.csv', 'multi_horizon'),
    ('tft_paper_mh',   'models_tft_paper_mh_{basis}/xgb_preds/test.csv',   'multi_horizon'),
    ('xgb_default',    'models_xgb_default_d{h}_oc/xgb_preds/test.csv',    'single_horizon'),
    ('lightgbm',       'models_lightgbm_d{h}_oc/xgb_preds/test.csv',       'single_horizon'),
    ('catboost',       'models_catboost_d{h}_oc/xgb_preds/test.csv',       'single_horizon'),
]


def load_engine_preds(name: str, pattern: str, kind: str, basis: str):
    """Return DataFrame keyed by (ts_code, trade_date) with pred_d1..pred_d5."""
    if kind == 'multi_horizon':
        fp = DATA / pattern.format(basis=basis)
        if not fp.exists():
            return None
        df = pd.read_csv(fp, parse_dates=['trade_date'])
        # Ensure pred_d1..pred_d5 columns exist
        cols = ['ts_code', 'trade_date'] + [f'pred_d{h}' for h in HORIZONS]
        for c in cols:
            if c not in df.columns:
                _log(f"  {name}: column {c} missing — skipping")
                return None
        return df[cols]

    if kind == 'single_horizon':
        # Five separate files, one per horizon — merge into wide
        per_h = {}
        for h in HORIZONS:
            fp = DATA / pattern.format(h=h)
            if not fp.exists():
                _log(f"  {name} d{h}: missing {fp}, skipping engine")
                return None
            df = pd.read_csv(fp, parse_dates=['trade_date'])
            df = df.rename(columns={'pred': f'pred_d{h}'})
            per_h[h] = df[['ts_code', 'trade_date', f'pred_d{h}']]
        out = per_h[1]
        for h in HORIZONS[1:]:
            out = out.merge(per_h[h], on=['ts_code', 'trade_date'], how='outer')
        return out

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--basis', default='oc', choices=['cc', 'oc'])
    p.add_argument('--engines', nargs='+', default=None,
                   help='subset of engine names; default = all auto-detected')
    p.add_argument('--rank_avg', action='store_true', default=True,
                   help='cross-sectionally rank-average predictions before mean '
                        '(more robust to engine-specific scale)')
    args = p.parse_args()

    selected = args.engines or [e[0] for e in ENGINE_SOURCES]
    _log(f"basis={args.basis}  engines={selected}")

    frames = {}
    for name, pat, kind in ENGINE_SOURCES:
        if name not in selected: continue
        df = load_engine_preds(name, pat, kind, args.basis)
        if df is None or df.empty:
            _log(f"  ✗ {name}: no preds available")
            continue
        # Drop rows with any NaN pred
        before = len(df)
        df = df.dropna(subset=[f'pred_d{h}' for h in HORIZONS])
        _log(f"  ✓ {name}: {len(df):,} rows ({before-len(df):,} dropped due to NaN)")
        frames[name] = df

    if not frames:
        raise SystemExit("No engines produced predictions; cannot ensemble.")

    # Optional: cross-sectional rank within each (engine, trade_date) per horizon
    if args.rank_avg:
        for name in list(frames):
            df = frames[name].copy()
            for h in HORIZONS:
                col = f'pred_d{h}'
                df[col] = df.groupby('trade_date')[col].rank(pct=True)
            frames[name] = df
        _log(f"  applied per-engine cross-sectional rank-pct transform")

    # Merge all engines on (ts_code, trade_date) — outer merge so we keep
    # every (stock, date) that appears in at least one engine.
    base = None
    for name, df in frames.items():
        df = df.add_suffix(f'__{name}')
        df = df.rename(columns={f'ts_code__{name}': 'ts_code',
                                  f'trade_date__{name}': 'trade_date'})
        base = df if base is None else base.merge(df, on=['ts_code','trade_date'], how='outer')

    # Mean across engines per horizon
    out_cols = ['ts_code', 'trade_date']
    for h in HORIZONS:
        cols_h = [f'pred_d{h}__{n}' for n in frames if f'pred_d{h}__{n}' in base.columns]
        base[f'pred_d{h}'] = base[cols_h].mean(axis=1)
        out_cols.append(f'pred_d{h}')
    out = base[out_cols].sort_values(['trade_date','ts_code']).reset_index(drop=True)

    # The single-horizon `pred` column for backtest compatibility = pred_d1
    out['pred']   = out['pred_d1']
    # Add a synthetic 'target' column = NaN (rebalance backtest doesn't need it)
    out['target'] = np.nan

    out_dir = DATA / f'models_ensemble_mh_{args.basis}/xgb_preds'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / 'test.csv'
    out.to_csv(out_p, index=False)
    _log(f"wrote {out_p}  ({len(out):,} rows × {len(out.columns)} cols)")

    # Also write a per-horizon summary IC over the 'pred' agreement against
    # the GBM target column where it's available
    print()
    print("=" * 60)
    print(f"ENSEMBLE RANK CORR ACROSS ENGINES  (basis={args.basis})")
    print("=" * 60)
    eng_names = list(frames)
    for h in HORIZONS:
        cols_h = [f'pred_d{h}__{n}' for n in eng_names if f'pred_d{h}__{n}' in base.columns]
        if len(cols_h) < 2: continue
        sub = base[cols_h].dropna()
        if len(sub) < 100: continue
        corr_mat = sub.corr(method='spearman').values
        # Average off-diagonal correlation
        n = len(corr_mat)
        avg_off = (corr_mat.sum() - np.trace(corr_mat)) / (n * (n - 1))
        print(f"  d{h}: avg cross-engine Spearman = {avg_off:+.3f}  ({len(cols_h)} engines)")
    print()

    # Save manifest
    manifest = {
        'basis':       args.basis,
        'engines':     list(frames.keys()),
        'rank_avg':    bool(args.rank_avg),
        'horizons':    list(HORIZONS),
        'n_rows':      int(len(out)),
        'n_unique_stocks': int(out['ts_code'].nunique()),
        'date_range':  [str(out['trade_date'].min()), str(out['trade_date'].max())],
    }
    with open(out_dir.parent / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _log(f"wrote manifest → {out_dir.parent / 'meta.json'}")


if __name__ == '__main__':
    main()
