"""
Train 15 single-target regressors: 3 engines × 5 horizons.

Engines     : xgb_default, lightgbm, catboost
Horizons    : t+1, t+2, t+3, t+4, t+5 (excess return vs CSI300, demeaned)
Split       : time-based 80/10/10
              train  < 2024-01-01
              val    2024-01-01 .. 2025-06-30
              test   ≥ 2025-07-01
Outputs     : stock_data/models_<engine>_d<h>/{model, meta.json, xgb_preds/{val,test}.csv}

Run:
    ./venv/Scripts/python -m xgbmodel.train_multihorizon
    ./venv/Scripts/python -m xgbmodel.train_multihorizon --engines xgb_default
    ./venv/Scripts/python -m xgbmodel.train_multihorizon --device cpu
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


VAL_START   = pd.Timestamp('2024-01-01')
TEST_START  = pd.Timestamp('2025-07-01')
HORIZONS    = (1, 2, 3, 4, 5)


def _log(msg): print(f"[mh-train] {msg}", flush=True)


def _split(panel: pd.DataFrame, target_col: str):
    """Drop NaN targets per horizon, then split by date."""
    df = panel[panel[target_col].notna()]
    train = df[df['trade_date'] <  VAL_START]
    val   = df[(df['trade_date'] >= VAL_START) & (df['trade_date'] < TEST_START)]
    test  = df[df['trade_date'] >= TEST_START]
    return train, val, test


def _rank_ic(y_true, y_pred) -> float:
    if len(y_true) < 50: return float('nan')
    s = pd.DataFrame({'y': y_true, 'p': y_pred})
    return float(s['y'].corr(s['p'], method='spearman'))


def _fit_xgb(train, val, test, feat_cols, target_col, cfg):
    import xgboost as xgb
    params = dict(cfg['xgb_params'])
    if cfg.get('device') == 'cuda':
        params['device'] = 'cuda'
    params['n_estimators']           = 500
    params['early_stopping_rounds']  = 30
    m = xgb.XGBRegressor(**params)
    m.fit(
        train[feat_cols], train[target_col],
        eval_set=[(val[feat_cols], val[target_col])],
        verbose=False,
    )
    val_pred  = m.predict(val [feat_cols])
    test_pred = m.predict(test[feat_cols])
    return m, val_pred, test_pred, int(m.best_iteration or m.n_estimators)


def _fit_lgbm(train, val, test, feat_cols, target_col, cfg):
    import lightgbm as lgb
    x = cfg.get('xgb_params', {})
    params = {
        'objective':         'huber', 'alpha': 0.9,
        'learning_rate':     x.get('learning_rate', 0.015),
        'num_leaves':        2 ** x.get('max_depth', 5) - 1,
        'max_depth':         -1,
        'min_child_samples': x.get('min_child_weight', 200),
        'feature_fraction':  x.get('colsample_bytree', 0.6),
        'bagging_fraction':  x.get('subsample',        0.7),
        'bagging_freq':      1,
        'lambda_l1':         x.get('reg_alpha',  0.1),
        'lambda_l2':         x.get('reg_lambda', 3.0),
        'verbosity':         -1,
        'n_estimators':      1000,
        'random_state':      cfg.get('random_seed', 42),
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(
        train[feat_cols], train[target_col],
        eval_set=[(val[feat_cols], val[target_col])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    val_pred  = m.predict(val [feat_cols])
    test_pred = m.predict(test[feat_cols])
    return m, val_pred, test_pred, int(m.best_iteration_ or m.n_estimators)


def _fit_catboost(train, val, test, feat_cols, target_col, cfg):
    from catboost import CatBoostRegressor, Pool
    x = cfg.get('xgb_params', {})
    params = {
        'loss_function':     'Huber:delta=1.0',
        'learning_rate':     x.get('learning_rate', 0.015),
        'depth':             min(x.get('max_depth', 5), 8),
        'l2_leaf_reg':       x.get('reg_lambda', 3.0),
        'subsample':         x.get('subsample', 0.7),
        'min_data_in_leaf':  x.get('min_child_weight', 200),
        'iterations':        1000,
        'random_seed':       cfg.get('random_seed', 42),
        'allow_writing_files': False,
        'verbose':           False,
        'task_type':         'GPU' if cfg.get('device') == 'cuda' else 'CPU',
        'early_stopping_rounds': 30,
    }
    if params['task_type'] == 'GPU':
        params['bootstrap_type'] = 'Bernoulli'
    m = CatBoostRegressor(**params)
    m.fit(
        Pool(train[feat_cols], train[target_col]),
        eval_set=Pool(val[feat_cols], val[target_col]),
        verbose=False,
    )
    val_pred  = m.predict(val [feat_cols])
    test_pred = m.predict(test[feat_cols])
    return m, val_pred, test_pred, int(getattr(m, 'best_iteration_', None) or m.tree_count_)


ENGINE_FITTERS = {
    'xgb_default':  _fit_xgb,
    'lightgbm':     _fit_lgbm,
    'catboost':     _fit_catboost,
}


def _save_model(engine: str, horizon: int, model, suffix: str = ''):
    md = DATA / f'models_{engine}_d{horizon}{suffix}'
    md.mkdir(parents=True, exist_ok=True)
    if engine == 'xgb_default':
        model.save_model(md / 'xgb_pct_chg.json')
    elif engine == 'lightgbm':
        model.booster_.save_model(str(md / 'lgbm.txt'))
    elif engine == 'catboost':
        model.save_model(str(md / 'catboost.cbm'))
    return md


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engines', nargs='+',
                   default=['xgb_default', 'lightgbm', 'catboost'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_stocks', type=int, default=0)
    p.add_argument('--target_basis', choices=['cc', 'oc'], default='cc',
                   help='Target return basis. cc = close-to-close (legacy, '
                        'requires zero-lag execution to capture). oc = open-to-'
                        'close intra-day return — what realistic open-entry '
                        'trades can capture (Path 1 fix).')
    p.add_argument('--model_suffix', default='',
                   help='Optional suffix appended to model_dir, e.g. "_oc". '
                        'Lets us keep cc and oc trained models side-by-side '
                        'for comparison without overwriting.')
    args = p.parse_args()

    cfg = get_config(max_stocks=args.max_stocks, device=args.device,
                     for_inference=False)
    cfg['target_basis'] = args.target_basis
    _log(f"building panel (device={args.device}, max_stocks={args.max_stocks or 'all'}) ...")
    panel = build_panel(cfg)
    feat_cols = list_feature_columns(panel)
    _log(f"panel: {panel.shape}, features={len(feat_cols)}")
    _log(f"date range: {panel['trade_date'].min().date()} → "
         f"{panel['trade_date'].max().date()}")

    grid = {}   # (engine, h) -> {val_ic, test_ic, val_rmse, test_rmse, best_iter, t_fit}

    for engine in args.engines:
        if engine not in ENGINE_FITTERS:
            _log(f"WARN: unknown engine {engine}, skipping"); continue
        fitter = ENGINE_FITTERS[engine]

        for h in HORIZONS:
            target_col = f'target_d{h}'
            train, val, test = _split(panel, target_col)
            _log(f"=== {engine} d{h}: train={len(train):,}  val={len(val):,}  test={len(test):,} ===")
            t0 = time.time()
            try:
                model, val_pred, test_pred, best_iter = fitter(
                    train, val, test, feat_cols, target_col, cfg)
            except Exception as e:
                import traceback; traceback.print_exc()
                _log(f"{engine} d{h} FAILED: {e}"); continue
            t_fit = time.time() - t0

            val_ic   = _rank_ic(val [target_col].values, val_pred)
            test_ic  = _rank_ic(test[target_col].values, test_pred)
            val_rmse = float(np.sqrt(np.mean((val [target_col].values - val_pred ) ** 2)))
            test_rmse= float(np.sqrt(np.mean((test[target_col].values - test_pred) ** 2)))

            md = _save_model(engine, h, model, suffix=args.model_suffix)

            preds_dir = md / 'xgb_preds'; preds_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                'ts_code':    val ['ts_code'].values,
                'trade_date': val ['trade_date'].values,
                'pred':       val_pred,
                'target':     val [target_col].values,
            }).to_csv(preds_dir / 'val.csv', index=False)
            pd.DataFrame({
                'ts_code':    test['ts_code'].values,
                'trade_date': test['trade_date'].values,
                'pred':       test_pred,
                'target':     test[target_col].values,
            }).to_csv(preds_dir / 'test.csv', index=False)

            with open(md / 'meta.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'engine':            engine,
                    'horizon':           h,
                    'mode':              'time_split',
                    'val_start':         VAL_START.strftime('%Y-%m-%d'),
                    'test_start':        TEST_START.strftime('%Y-%m-%d'),
                    'n_features':        len(feat_cols),
                    'n_train':           len(train),
                    'n_val':             len(val),
                    'n_test':            len(test),
                    'val_ic':            val_ic,
                    'test_ic':           test_ic,
                    'val_rmse':          val_rmse,
                    'test_rmse':         test_rmse,
                    'best_iteration':    best_iter,
                    'fit_seconds':       t_fit,
                }, f, ensure_ascii=False, indent=2)

            grid[(engine, h)] = {
                'val_ic': val_ic, 'test_ic': test_ic,
                'val_rmse': val_rmse, 'test_rmse': test_rmse,
                'best_iter': best_iter, 't_fit': t_fit,
            }
            _log(f"    val IC={val_ic:+.4f}  test IC={test_ic:+.4f}  "
                 f"test RMSE={test_rmse:.3f}  best_iter={best_iter}  "
                 f"({t_fit:.1f}s)")

    # ── Print 3 × 5 IC grid ──
    print()
    print("=" * 60)
    print("Per-engine × per-horizon test IC (Spearman rank)")
    print("=" * 60)
    print(f"{'engine':14}  " + "  ".join(f' d{h:>1} '   for h in HORIZONS))
    for engine in args.engines:
        if engine not in ENGINE_FITTERS: continue
        row = f"{engine:14}  "
        for h in HORIZONS:
            r = grid.get((engine, h))
            row += f"{r['test_ic']:+.3f} " if r else "  -  "
            row += " "
        print(row)
    print()
    print(f"{'engine':14}  " + "  ".join(f'val{h:>1}' for h in HORIZONS))
    for engine in args.engines:
        if engine not in ENGINE_FITTERS: continue
        row = f"{engine:14}  "
        for h in HORIZONS:
            r = grid.get((engine, h))
            row += f"{r['val_ic']:+.3f} " if r else "  -  "
            row += " "
        print(row)


if __name__ == '__main__':
    main()
