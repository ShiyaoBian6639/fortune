"""
After walk-forward CV, refit a canonical model on the full panel for each
engine and run predict_latest. Outputs:

  stock_predictions_<engine>.csv  (live, next-trading-day predictions, all stocks)

Reuses the engine's canonical_n_estimators from meta.json. Skips engines that
don't have meta.json yet (still training).

Run:
    ./venv/Scripts/python -m model_compare.refit_canonical
    ./venv/Scripts/python -m model_compare.refit_canonical --engines xgb_shallow lightgbm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'

from xgbmodel.config      import get_config
from xgbmodel.data_loader import build_panel, list_feature_columns


def _log(m): print(f"[refit] {m}", flush=True)


def _refit_xgb(name: str, X, y, panel: pd.DataFrame,
                feat_cols: list, cfg: dict, n_estimators: int,
                hyper_overrides: dict):
    import xgboost as xgb
    params = dict(cfg['xgb_params'])
    params.update(hyper_overrides)
    params['n_estimators'] = n_estimators
    params.pop('early_stopping_rounds', None)
    if cfg.get('device') == 'cuda':
        params['device'] = 'cuda'
    m = xgb.XGBRegressor(**params)
    _log(f"{name}: fitting XGBRegressor on {len(X):,} rows × {X.shape[1]} feats, n_est={n_estimators}")
    m.fit(X, y, verbose=False)
    return m


def _refit_lgbm(name: str, X, y, panel, feat_cols, cfg, n_estimators):
    import lightgbm as lgb
    x = cfg.get('xgb_params', {})
    params = {
        'objective': 'huber', 'alpha': 0.9,
        'learning_rate':     x.get('learning_rate', 0.015),
        'num_leaves':        2 ** x.get('max_depth', 5) - 1,
        'max_depth':         -1,
        'min_child_samples': x.get('min_child_weight', 200),
        'feature_fraction':  x.get('colsample_bytree', 0.6),
        'bagging_fraction':  x.get('subsample',        0.7),
        'bagging_freq':      1,
        'lambda_l1':         x.get('reg_alpha', 0.1),
        'lambda_l2':         x.get('reg_lambda', 3.0),
        'verbosity':         -1,
        'n_estimators':      n_estimators,
        'random_state':      cfg.get('random_seed', 42),
    }
    m = lgb.LGBMRegressor(**params)
    _log(f"{name}: fitting LightGBM on {len(X):,} rows × {X.shape[1]} feats, n_est={n_estimators}")
    m.fit(X, y)
    return m


def _refit_catboost(name: str, X, y, panel, feat_cols, cfg, iterations):
    from catboost import CatBoostRegressor
    x = cfg.get('xgb_params', {})
    params = {
        'loss_function':    'Huber:delta=1.0',
        'learning_rate':    x.get('learning_rate', 0.015),
        'depth':            min(x.get('max_depth', 5), 8),
        'l2_leaf_reg':      x.get('reg_lambda', 3.0),
        'subsample':        x.get('subsample', 0.7),
        'min_data_in_leaf': x.get('min_child_weight', 200),
        'iterations':       iterations,
        'random_seed':      cfg.get('random_seed', 42),
        'allow_writing_files': False,
        'verbose':          False,
        'task_type':        'GPU' if cfg.get('device') == 'cuda' else 'CPU',
    }
    if params['task_type'] == 'GPU':
        params['bootstrap_type'] = 'Bernoulli'
    m = CatBoostRegressor(**params)
    _log(f"{name}: fitting CatBoost on {len(X):,} rows × {X.shape[1]} feats, iters={iterations}")
    m.fit(X, y)
    return m


XGB_OVERRIDES = {
    'xgb_default':     {},
    'xgb_shallow':     {'max_depth': 3, 'min_child_weight': 100},
    'xgb_deep':        {'max_depth': 8, 'min_child_weight': 400},
    'xgb_strong_reg':  {'reg_alpha': 0.5, 'reg_lambda': 10.0,
                         'subsample': 0.6, 'colsample_bytree': 0.5},
}


def refit_one(name: str, panel: pd.DataFrame, feat_cols: list, cfg: dict):
    md = DATA / f'models_{name}'
    meta_p = md / 'meta.json'
    if not meta_p.exists():
        _log(f"{name}: no meta.json (not trained yet) — skip")
        return None
    with open(meta_p, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    n_can = meta.get('canonical_n_estimators') or 220

    # for_inference=True keeps NaN-target rows for the most recent date —
    # those must be EXCLUDED from fit but INCLUDED in predict.
    fit_mask = panel['target'].notna()
    Xfit = panel.loc[fit_mask, feat_cols].astype('float32')
    yfit = panel.loc[fit_mask, 'target'].astype('float32')
    _log(f"{name}: fit panel = {len(Xfit):,} rows (dropped {(~fit_mask).sum():,} NaN-target rows)")

    t0 = time.time()
    if name in XGB_OVERRIDES:
        m = _refit_xgb(name, Xfit, yfit, panel, feat_cols, cfg, n_can, XGB_OVERRIDES[name])
        predict = m.predict
    elif name == 'lightgbm':
        m = _refit_lgbm(name, Xfit, yfit, panel, feat_cols, cfg, n_can)
        predict = m.predict
    elif name == 'catboost':
        m = _refit_catboost(name, Xfit, yfit, panel, feat_cols, cfg, n_can)
        predict = m.predict
    else:
        _log(f"{name}: unsupported engine"); return None
    _log(f"{name}: fit done in {time.time() - t0:.1f}s")

    # Predict on last row per stock — same logic as predict_latest
    last_date = panel['trade_date'].max()
    latest = panel[panel['trade_date'] == last_date]
    _log(f"{name}: predicting {len(latest):,} stocks for next trading day after {last_date.date()}")
    Xl = latest[feat_cols].astype('float32')
    pred = predict(Xl)

    out = pd.DataFrame({
        'ts_code':            latest['ts_code'].values,
        'trade_date':         latest['trade_date'].values,   # feature date
        'pred_pct_chg_next':  pred.astype('float32'),
    }).sort_values('pred_pct_chg_next', ascending=False).reset_index(drop=True)
    out_path = ROOT / f'stock_predictions_{name}.csv'
    out.to_csv(out_path, index=False)
    _log(f"{name}: wrote {out_path}  ({len(out):,} stocks, "
         f"top μ={out['pred_pct_chg_next'].max():+.3f}%, "
         f"bottom μ={out['pred_pct_chg_next'].min():+.3f}%)")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engines', nargs='+', default=None,
                   help='Subset (default: all engines whose meta.json exists)')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    pool = args.engines
    if pool is None:
        pool = [d.name.replace('models_', '')
                for d in DATA.glob('models_*')
                if (d / 'meta.json').exists() and not d.name.endswith('_backup')
                and not d.name.endswith('_fw5')]
    pool = [n for n in pool
            if (DATA / f'models_{n}' / 'meta.json').exists()
            and n not in ('ensemble_mean', 'ensemble_rankavg', 'transformer_reg', 'tft')]
    if not pool:
        _log('no engines to refit'); return

    cfg = get_config(max_stocks=0, device=args.device, for_inference=True)
    _log(f"building inference panel (with last row preserved) ...")
    panel = build_panel(cfg)
    feat_cols = list_feature_columns(panel)
    _log(f"panel: {panel.shape}, last_date={panel['trade_date'].max().date()}")

    for name in pool:
        try:
            refit_one(name, panel, feat_cols, cfg)
        except Exception as e:
            import traceback; traceback.print_exc()
            _log(f"{name}: FAILED — {e}")


if __name__ == '__main__':
    main()
