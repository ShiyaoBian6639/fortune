"""
Load 15 trained models (3 engines × 5 horizons) and emit per-engine live
next-trading-week predictions for every stock.

Outputs (CSV per engine):
    stock_predictions_multihorizon_xgb.csv
    stock_predictions_multihorizon_lightgbm.csv
    stock_predictions_multihorizon_catboost.csv

Schema:
    ts_code, trade_date (= feature date),
    pred_d1, pred_d2, pred_d3, pred_d4, pred_d5,
    mean5    -- arithmetic mean of the 5 horizon preds
    max5     -- max of the 5 horizon preds (peak-day signal)
    slope    -- linear slope of (1..5) vs preds (positive = rising trajectory)

Run:
    ./venv/Scripts/python -m xgbmodel.predict_multihorizon
    ./venv/Scripts/python -m xgbmodel.predict_multihorizon --engines xgb_default
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'

from xgbmodel.config      import get_config
from xgbmodel.data_loader import build_panel, list_feature_columns

HORIZONS = (1, 2, 3, 4, 5)


def _log(m): print(f"[mh-predict] {m}", flush=True)


def _load_model(engine: str, horizon: int):
    md = DATA / f'models_{engine}_d{horizon}'
    if not (md / 'meta.json').exists():
        return None
    if engine == 'xgb_default':
        import xgboost as xgb
        m = xgb.XGBRegressor()
        m.load_model(str(md / 'xgb_pct_chg.json'))
        return ('xgb', m)
    if engine == 'lightgbm':
        import lightgbm as lgb
        bst = lgb.Booster(model_file=str(md / 'lgbm.txt'))
        return ('lgbm', bst)
    if engine == 'catboost':
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(md / 'catboost.cbm'))
        return ('cat', m)
    return None


def _predict(loaded, X):
    kind, m = loaded
    if kind == 'xgb':  return m.predict(X)
    if kind == 'lgbm': return m.predict(X)
    if kind == 'cat':  return m.predict(X)
    raise ValueError(kind)


# Per-stock slope of preds vs (1..5)  via least-squares on a 5-point series.
# pre-computed terms (x = 1..5, mean=3, var=2):  slope = sum((x-3)*y) / 10
_X_CENTERED = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
_X_VAR_TIMES_N = 10.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engines', nargs='+',
                   default=['xgb_default', 'lightgbm', 'catboost'])
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    cfg = get_config(max_stocks=0, device=args.device, for_inference=True)
    _log("building inference panel ...")
    panel = build_panel(cfg)
    feat_cols = list_feature_columns(panel)
    last_date = panel['trade_date'].max()
    latest = panel[panel['trade_date'] == last_date].copy().reset_index(drop=True)
    _log(f"panel: {panel.shape}  last_date={last_date.date()}  "
         f"latest_rows={len(latest):,}")

    Xl = latest[feat_cols].astype('float32')

    for engine in args.engines:
        out_name = {
            'xgb_default': 'xgb',
            'lightgbm':    'lightgbm',
            'catboost':    'catboost',
        }.get(engine, engine)
        _log(f"=== {engine} ===")
        preds = {}
        for h in HORIZONS:
            loaded = _load_model(engine, h)
            if loaded is None:
                _log(f"  d{h} model missing — skipping engine")
                preds = None; break
            yhat = _predict(loaded, Xl)
            preds[f'pred_d{h}'] = np.asarray(yhat, dtype='float32')
            _log(f"  d{h}: top5 = {np.sort(yhat)[-5:][::-1].round(3)}")
        if preds is None:
            continue

        # Aggregate stats per stock
        P = np.stack([preds[f'pred_d{h}'] for h in HORIZONS], axis=1)   # (N, 5)
        mean5 = P.mean(axis=1).astype('float32')
        max5  = P.max (axis=1).astype('float32')
        slope = (P @ _X_CENTERED) / _X_VAR_TIMES_N

        out = pd.DataFrame({
            'ts_code':    latest['ts_code'].values,
            'trade_date': latest['trade_date'].values,
            **preds,
            'mean5':  mean5,
            'max5':   max5,
            'slope':  slope.astype('float32'),
        })
        out = out.sort_values('mean5', ascending=False).reset_index(drop=True)

        out_path = ROOT / f'stock_predictions_multihorizon_{out_name}.csv'
        out.to_csv(out_path, index=False)
        _log(f"  wrote {out_path}  ({len(out):,} stocks)")
        # Top 5 picks summary
        for _, r in out.head(5).iterrows():
            traj = ' → '.join(f"{r[f'pred_d{h}']:+.2f}" for h in HORIZONS)
            _log(f"    {r['ts_code']}  mean={r['mean5']:+.3f}  "
                 f"max={r['max5']:+.3f}  slope={r['slope']:+.3f}  [{traj}]")


if __name__ == '__main__':
    main()
