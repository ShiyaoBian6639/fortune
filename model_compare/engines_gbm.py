"""
Gradient-boosting engines: XGBoost (default + 3 hyperparameter variants),
LightGBM, CatBoost.

All produce predictions in the canonical xgb_preds/test.csv schema.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from xgbmodel.train import compute_metrics

from model_compare.engine import Engine, FitResult, score_frame


# ─── XGBoost engines (baseline + 3 variants) ────────────────────────────────
class XGBEngine(Engine):
    """Generic XGBoost engine — `name` and `xgb_param_overrides` make variants."""
    name = 'xgb_default'
    xgb_param_overrides: dict = {}

    def fit_fold(self, train_df, val_df, test_df, feat_cols):
        import xgboost as xgb
        params = dict(self.cfg['xgb_params'])
        params.update(self.xgb_param_overrides)
        if self.cfg.get('device') == 'cuda':
            params['device'] = 'cuda'
        params['random_state'] = self.cfg.get('random_seed', 42)

        X_tr, y_tr = train_df[feat_cols], train_df['target']
        X_va, y_va = val_df[feat_cols],   val_df['target']

        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=0)

        preds = {
            'val':  score_frame(m.predict, val_df,  feat_cols),
            'test': score_frame(m.predict, test_df, feat_cols),
        }
        metrics = {
            'val':  compute_metrics(preds['val'],  'val' ).as_dict(),
            'test': compute_metrics(preds['test'], 'test').as_dict(),
        }
        return FitResult(model=m, preds=preds, metrics=metrics,
                         best_iteration=getattr(m, 'best_iteration', None))


class XGBShallow(XGBEngine):
    name = 'xgb_shallow'
    xgb_param_overrides = {'max_depth': 3, 'min_child_weight': 100}


class XGBDeep(XGBEngine):
    name = 'xgb_deep'
    xgb_param_overrides = {'max_depth': 8, 'min_child_weight': 400, 'n_estimators': 800}


class XGBStrongReg(XGBEngine):
    name = 'xgb_strong_reg'
    xgb_param_overrides = {'reg_alpha': 0.5, 'reg_lambda': 10.0,
                            'subsample': 0.6, 'colsample_bytree': 0.5}


# ─── LightGBM engine ────────────────────────────────────────────────────────
class LightGBMEngine(Engine):
    name = 'lightgbm'

    def _params(self) -> dict:
        # Match XGBoost's defaults where sensible; LightGBM-specific names below.
        x = self.cfg.get('xgb_params', {})
        return {
            'objective':         'huber',           # close analogue to pseudo-huber
            'alpha':             0.9,               # huber quantile-ish param
            'learning_rate':     x.get('learning_rate', 0.015),
            'num_leaves':        2 ** x.get('max_depth', 5) - 1,
            'max_depth':         -1,                # no hard cap, num_leaves controls
            'min_child_samples': x.get('min_child_weight', 200),
            'feature_fraction':  x.get('colsample_bytree', 0.6),
            'bagging_fraction':  x.get('subsample',        0.7),
            'bagging_freq':      1,
            'lambda_l1':         x.get('reg_alpha',  0.1),
            'lambda_l2':         x.get('reg_lambda', 3.0),
            'verbosity':         -1,
            'n_estimators':      x.get('n_estimators', 1000),
            'early_stopping_round': x.get('early_stopping_rounds', 50),
            'random_state':      self.cfg.get('random_seed', 42),
            'device':            'gpu' if self.cfg.get('device') == 'cuda' else 'cpu',
            'num_threads':       0,
        }

    def fit_fold(self, train_df, val_df, test_df, feat_cols):
        import lightgbm as lgb
        params = self._params()
        # GPU support in lightgbm wheel is flaky on Windows — fall back to CPU
        if params.get('device') == 'gpu':
            try:
                lgb.Dataset(np.zeros((4, 4)), label=np.zeros(4)).construct()
            except Exception:
                pass
            params['device'] = 'cpu'   # safer default for pip wheel

        X_tr, y_tr = train_df[feat_cols], train_df['target']
        X_va, y_va = val_df[feat_cols],   val_df['target']

        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=params.pop('early_stopping_round', 50),
                                              verbose=False)])

        preds = {
            'val':  score_frame(m.predict, val_df,  feat_cols),
            'test': score_frame(m.predict, test_df, feat_cols),
        }
        metrics = {
            'val':  compute_metrics(preds['val'],  'val' ).as_dict(),
            'test': compute_metrics(preds['test'], 'test').as_dict(),
        }
        return FitResult(model=m, preds=preds, metrics=metrics,
                         best_iteration=getattr(m, 'best_iteration_', None))


# ─── CatBoost engine ────────────────────────────────────────────────────────
class CatBoostEngine(Engine):
    name = 'catboost'

    def _params(self) -> dict:
        x = self.cfg.get('xgb_params', {})
        params = {
            'loss_function':    'Huber:delta=1.0',
            'learning_rate':    x.get('learning_rate', 0.015),
            'depth':            min(x.get('max_depth', 5), 8),   # CatBoost cap
            'l2_leaf_reg':      x.get('reg_lambda', 3.0),
            'subsample':        x.get('subsample', 0.7),
            'min_data_in_leaf': x.get('min_child_weight', 200),
            'iterations':       x.get('n_estimators', 1000),
            'od_type':          'Iter',
            'od_wait':          x.get('early_stopping_rounds', 50),
            'random_seed':      self.cfg.get('random_seed', 42),
            'allow_writing_files': False,
            'verbose':          False,
            'task_type':        'GPU' if self.cfg.get('device') == 'cuda' else 'CPU',
        }
        # `rsm` (column subsample) is incompatible with Huber loss on GPU
        # (CatBoost only supports rsm on GPU for pairwise modes). Apply on CPU only.
        if params['task_type'] == 'CPU':
            params['rsm'] = x.get('colsample_bytree', 0.6)
        return params

    def fit_fold(self, train_df, val_df, test_df, feat_cols):
        from catboost import CatBoostRegressor
        params = self._params()
        # CatBoost GPU + subsample requires Bernoulli bootstrap
        if params['task_type'] == 'GPU':
            params['bootstrap_type'] = 'Bernoulli'

        X_tr, y_tr = train_df[feat_cols], train_df['target']
        X_va, y_va = val_df[feat_cols],   val_df['target']

        m = CatBoostRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))

        def _predict(X):
            return m.predict(X)

        preds = {
            'val':  score_frame(_predict, val_df,  feat_cols),
            'test': score_frame(_predict, test_df, feat_cols),
        }
        metrics = {
            'val':  compute_metrics(preds['val'],  'val' ).as_dict(),
            'test': compute_metrics(preds['test'], 'test').as_dict(),
        }
        return FitResult(model=m, preds=preds, metrics=metrics,
                         best_iteration=getattr(m, 'tree_count_', None))


# ─── Registry ────────────────────────────────────────────────────────────────
ENGINES = {
    'xgb_default':     XGBEngine,
    'xgb_shallow':     XGBShallow,
    'xgb_deep':        XGBDeep,
    'xgb_strong_reg':  XGBStrongReg,
    'lightgbm':        LightGBMEngine,
    'catboost':        CatBoostEngine,
}
