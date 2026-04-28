"""
Engine-agnostic walk-forward orchestrator.

Reuses xgbmodel.data_loader.build_panel and xgbmodel.split.walk_forward_folds
to construct identical folds across all engines. Each fold is sliced from
the full panel and handed to engine.fit_fold(). OOF val + test predictions
are concatenated and saved in the canonical xgb_preds CSV schema so the
existing backtest and dashboard infra work unchanged.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from xgbmodel.data_loader import build_panel, list_feature_columns
from xgbmodel.split       import walk_forward_folds, summarize_folds
from xgbmodel.train       import compute_metrics

from model_compare.engine import Engine


def run_walk_forward(engine: Engine, cfg: dict,
                     panel: pd.DataFrame = None) -> dict:
    """Walk-forward CV with `engine` on the same fold structure as xgbmodel.

    Returns a meta dict (saved to model_dir/meta.json). OOF val+test predictions
    are saved to model_dir/xgb_preds/{val,test}.csv.
    """
    t0 = time.time()
    if panel is None:
        panel = build_panel(cfg)
    feat_cols = list_feature_columns(panel)
    print(f"[{engine.name}] using {len(feat_cols)} features, "
          f"panel shape: {panel.shape}")

    folds = walk_forward_folds(
        panel,
        fold_train_weeks = cfg.get('fold_train_weeks', 12),
        fold_val_weeks   = cfg.get('fold_val_weeks',   2),
        fold_test_weeks  = cfg.get('fold_test_weeks',  2),
        fold_step_weeks  = cfg.get('fold_step_weeks',  2),
        purge_days       = cfg.get('purge_days',       5),
        embargo_days     = cfg.get('embargo_days',     2),
        expanding        = cfg.get('expanding_train',  False),
    )
    print(f"[{engine.name}] {summarize_folds(folds)}")

    # Sequence engines (Transformer, TFT) need the full panel to build the
    # windowed dataset once before iterating folds — a per-fold slice loses
    # the historical sequence context. Hook is optional.
    if hasattr(engine, 'set_full_panel'):
        engine.set_full_panel(panel, feat_cols)

    oof_val, oof_test, per_fold, best_iters = [], [], [], []
    fold_limit = cfg.get('max_folds', 0) or len(folds)
    run_folds  = folds[:fold_limit]

    for fold in run_folds:
        train_df, val_df, test_df = fold.slice(panel)
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print(f"[{engine.name}] skip fold {fold.index}: empty after purge")
            continue
        print(f"\n[{engine.name}] {fold.summary(train_df, val_df, test_df)}")

        result = engine.fit_fold(train_df, val_df, test_df, feat_cols)
        if result.best_iteration is not None:
            best_iters.append(result.best_iteration)
        result.preds['val' ]['fold'] = fold.index
        result.preds['test']['fold'] = fold.index
        oof_val .append(result.preds['val'])
        oof_test.append(result.preds['test'])
        per_fold.append({
            'fold': fold.index,
            'metrics': result.metrics,
            'best_iteration': result.best_iteration,
            **result.extra,
        })

    if not per_fold:
        raise RuntimeError(f"[{engine.name}] no folds produced predictions")

    # Aggregate
    def _agg(metric, split):
        vals = np.array([f['metrics'][split][metric] for f in per_fold
                         if split in f['metrics']], dtype='float64')
        n = int(np.isfinite(vals).sum())
        if n == 0:
            return float('nan'), float('nan'), 0
        return float(np.nanmean(vals)), float(np.nanstd(vals)), n
    ic_v_m, ic_v_s, n_v_v   = _agg('rank_ic', 'val')
    ic_t_m, ic_t_s, n_t_v   = _agg('rank_ic', 'test')
    rmse_t_m, rmse_t_s, _   = _agg('rmse',    'test')
    pos_test = sum(1 for f in per_fold
                   if np.isfinite(f['metrics']['test']['rank_ic'])
                   and f['metrics']['test']['rank_ic'] > 0)

    print(f"\n[{engine.name}] summary over {len(per_fold)} folds:")
    print(f"  val  IC mean: {ic_v_m:+.4f}  std={ic_v_s:.4f}")
    print(f"  test IC mean: {ic_t_m:+.4f}  std={ic_t_s:.4f}  positive={pos_test}/{n_t_v}")
    print(f"  test RMSE: {rmse_t_m:.4f}  std={rmse_t_s:.4f}")

    # Save OOF predictions
    md = engine.model_dir()
    pd.concat(oof_val,  ignore_index=True).to_csv(md / 'xgb_preds' / 'val.csv', index=False)
    pd.concat(oof_test, ignore_index=True).to_csv(md / 'xgb_preds' / 'test.csv', index=False)

    meta = {
        'engine':            engine.name,
        'mode':              'walk_forward',
        'target_mode':       cfg.get('target_mode', 'excess'),
        'forward_window':    cfg.get('forward_window', 1),
        'n_features':        len(feat_cols),
        'fold_config': {
            'train_weeks':   cfg.get('fold_train_weeks', 12),
            'val_weeks':     cfg.get('fold_val_weeks',   2),
            'test_weeks':    cfg.get('fold_test_weeks',  2),
            'step_weeks':    cfg.get('fold_step_weeks',  2),
            'purge_days':    cfg.get('purge_days',       5),
            'embargo_days':  cfg.get('embargo_days',     2),
            'expanding':     cfg.get('expanding_train',  False),
        },
        'metric_summary': {
            'val_rank_ic_mean':     ic_v_m, 'val_rank_ic_std':     ic_v_s,
            'test_rank_ic_mean':    ic_t_m, 'test_rank_ic_std':    ic_t_s,
            'test_rmse_mean':       rmse_t_m, 'test_rmse_std':     rmse_t_s,
            'test_ic_positive_ratio': pos_test / max(n_t_v, 1),
        },
        'per_fold':           per_fold,
        'canonical_n_estimators': int(np.median(best_iters)) if best_iters else None,
        'total_seconds':      time.time() - t0,
    }
    with open(md / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[{engine.name}] meta + predictions saved → {md}")
    return meta
