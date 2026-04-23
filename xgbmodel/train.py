"""
Training entry for xgbmodel — next-day pct_chg regression.

Workflow:
  1. build_panel() → a single DataFrame
  2. Split by trade_date: train < VAL_START ≤ val < TEST_START ≤ test
  3. Fit XGBRegressor with early stopping on the val set
  4. Report IC (daily cross-sectional rank corr) / RMSE / MAE on val+test
  5. Persist model + feature list + metadata JSON
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import MODEL_DIR
from .data_loader import build_panel, list_feature_columns
from .split import walk_forward_folds, summarize_folds, Fold


# ─── Split ──────────────────────────────────────────────────────────────────

def time_split(panel: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) frames split by trade_date cutoffs in cfg."""
    train_start = pd.to_datetime(str(cfg['train_start']))
    val_start   = pd.to_datetime(str(cfg['val_start']))
    test_start  = pd.to_datetime(str(cfg['test_start']))

    train = panel[(panel['trade_date'] >= train_start) & (panel['trade_date'] < val_start)]
    val   = panel[(panel['trade_date'] >= val_start)   & (panel['trade_date'] < test_start)]
    test  = panel[panel['trade_date'] >= test_start]

    print(f"[xgbmodel] split sizes: "
          f"train={len(train):,} ({train['trade_date'].min().date()}–{train['trade_date'].max().date()})  "
          f"val={len(val):,}  test={len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ─── Metrics ────────────────────────────────────────────────────────────────

def daily_rank_ic(df: pd.DataFrame, pred_col: str = 'pred', target_col: str = 'target') -> float:
    """Mean cross-sectional Spearman IC across trade dates.

    Silently skips days with zero-variance pred or target (those produce NaN
    correlations from `corr()` — e.g. when the model barely trained and
    outputs a near-constant value).
    """
    ics = []
    with np.errstate(invalid='ignore', divide='ignore'):
        for _, grp in df.groupby('trade_date'):
            if len(grp) < 10:
                continue
            p = grp[pred_col].values
            y = grp[target_col].values
            # Zero-variance days cannot produce a rank correlation — skip.
            if not (np.isfinite(p).all() and np.isfinite(y).all()):
                continue
            if p.std() == 0 or y.std() == 0:
                continue
            r = grp[[pred_col, target_col]].rank(method='average')
            c = r.corr().iloc[0, 1]
            if np.isfinite(c):
                ics.append(c)
    return float(np.mean(ics)) if ics else float('nan')


@dataclass
class Metrics:
    split:   str
    n:       int
    rmse:    float
    mae:     float
    pearson: float
    rank_ic: float

    def as_dict(self):
        return asdict(self)


def compute_metrics(df: pd.DataFrame, split_name: str,
                    pred_col: str = 'pred', target_col: str = 'target') -> Metrics:
    y = df[target_col].values.astype('float64', copy=False)
    p = df[pred_col ].values.astype('float64', copy=False)
    # Drop non-finite rows (shouldn't happen, but be defensive)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(y) == 0:
        return Metrics(split=split_name, n=0,
                       rmse=float('nan'), mae=float('nan'),
                       pearson=float('nan'), rank_ic=float('nan'))

    resid = p - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae  = float(np.mean(np.abs(resid)))
    # Zero-variance inputs → corrcoef divides by zero. Guard explicitly.
    if len(y) < 2 or p.std() == 0 or y.std() == 0:
        pe = float('nan')
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            pe = float(np.corrcoef(p, y)[0, 1])
    ic = daily_rank_ic(df, pred_col, target_col)
    m = Metrics(split=split_name, n=len(y), rmse=rmse, mae=mae, pearson=pe, rank_ic=ic)
    print(f"  {split_name}: n={m.n:,}  RMSE={m.rmse:.4f}  MAE={m.mae:.4f}  "
          f"Pearson={m.pearson:+.4f}  Daily-IC={m.rank_ic:+.4f}")
    return m


# ─── Single-model fit helper (shared by fixed split and walk-forward) ──────

def _fit_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feat_cols: List[str],
    cfg: dict,
    verbose_every: int = 100,
) -> Tuple[xgb.XGBRegressor, Dict[str, pd.DataFrame], Dict[str, dict]]:
    """Fit one XGBRegressor on (train, val, test) frames. Returns (model, preds, metrics)."""
    xgb_params = dict(cfg['xgb_params'])
    device = cfg.get('device', 'cpu')
    if device == 'cuda':
        xgb_params['device'] = 'cuda'
    xgb_params['random_state'] = cfg.get('random_seed', 42)

    X_train, y_train = train_df[feat_cols], train_df['target']
    X_val,   y_val   = val_df[feat_cols],   val_df['target']

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=verbose_every,
    )

    def _score_frame(df_wide: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'ts_code':    df_wide['ts_code'].values,
            'trade_date': df_wide['trade_date'].values,
            'pred':       model.predict(df_wide[feat_cols]).astype('float32'),
            'target':     df_wide['target'].values,
        })

    preds = {'train': _score_frame(train_df), 'val': _score_frame(val_df)}
    metrics = {
        'train': compute_metrics(preds['train'], 'train').as_dict(),
        'val':   compute_metrics(preds['val'],   'val'  ).as_dict(),
    }
    if test_df is not None and len(test_df):
        preds['test']    = _score_frame(test_df)
        metrics['test']  = compute_metrics(preds['test'], 'test').as_dict()
    return model, preds, metrics


# ─── Training driver ────────────────────────────────────────────────────────

def train(cfg: dict) -> dict:
    """Build panel, split, train XGBRegressor, save model + metrics."""
    t0 = time.time()
    panel = build_panel(cfg)

    feat_cols = list_feature_columns(panel)
    print(f"[xgbmodel] using {len(feat_cols)} features")

    train_df, val_df, test_df = time_split(panel, cfg)
    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(
            f"Empty train/val split. train={len(train_df)} val={len(val_df)} — "
            f"check cfg['train_start']={cfg['train_start']} and cfg['val_start']={cfg['val_start']}"
        )

    device = cfg.get('device', 'cpu')
    xgb_params = cfg['xgb_params']
    print(f"[xgbmodel] fitting XGBRegressor on {len(train_df):,} × {len(feat_cols)} "
          f"(device={device}, max_depth={xgb_params.get('max_depth')}, "
          f"lr={xgb_params.get('learning_rate')}, n_est={xgb_params.get('n_estimators')})")

    fit_t0 = time.time()
    model, preds, metrics = _fit_one(train_df, val_df, test_df, feat_cols, cfg, verbose_every=100)
    fit_secs = time.time() - fit_t0
    best_iter = getattr(model, 'best_iteration', None)
    print(f"  fit time: {fit_secs:.1f}s   best_iteration={best_iter}")

    # Rebind for persistence below
    train_df = preds['train']
    val_df   = preds['val']
    test_df  = preds.get('test')

    # ─── Persist artifacts ───────────────────────────────────────────────────
    model_dir = cfg.get('model_dir', MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgb_pct_chg.json')
    feats_path = os.path.join(model_dir, 'xgb_pct_chg.features.json')
    meta_path  = os.path.join(model_dir, 'xgb_pct_chg.meta.json')

    model.save_model(model_path)
    with open(feats_path, 'w', encoding='utf-8') as f:
        json.dump(feat_cols, f, indent=2)

    # Feature importance (gain) for quick inspection
    booster = model.get_booster()
    gain = booster.get_score(importance_type='gain')
    total_gain = booster.get_score(importance_type='total_gain')
    fi = sorted(
        [{'feature': k, 'gain': float(gain[k]), 'total_gain': float(total_gain.get(k, 0.0))}
         for k in gain],
        key=lambda r: -r['total_gain'],
    )

    meta = {
        'target_mode':     cfg.get('target_mode'),
        'forward_window':  cfg.get('forward_window'),
        'train_start':     cfg.get('train_start'),
        'val_start':       cfg.get('val_start'),
        'test_start':      cfg.get('test_start'),
        'n_features':      len(feat_cols),
        'best_iteration':  best_iter,
        'fit_seconds':     fit_secs,
        'total_seconds':   time.time() - t0,
        'metrics':         metrics,
        'feature_importance_top50': fi[:50],
        'xgb_params':      xgb_params,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)

    # Save predictions for plotting
    preds_dir = os.path.join(model_dir, 'xgb_preds')
    os.makedirs(preds_dir, exist_ok=True)
    val_df.to_csv(os.path.join(preds_dir, 'val.csv'),  index=False)
    if isinstance(test_df, pd.DataFrame) and len(test_df):
        test_df.to_csv(os.path.join(preds_dir, 'test.csv'), index=False)

    print(f"[xgbmodel] saved model   → {model_path}")
    print(f"[xgbmodel] saved feats   → {feats_path}")
    print(f"[xgbmodel] saved meta    → {meta_path}")
    print(f"[xgbmodel] total wall   = {time.time() - t0:.1f}s")
    return meta


# ─── Walk-forward CV driver ────────────────────────────────────────────────

def train_walk_forward(cfg: dict) -> dict:
    """Run purged walk-forward CV and refit a final model on the full pool.

    For each generated fold:
      1. Slice the panel into (train, val, test) using purge + embargo gaps.
      2. Fit an XGBRegressor with early stopping on val.
      3. Record per-fold val and test metrics, plus the best_iteration.

    Mean best_iteration is used to refit a final model on
    train+val+test, which is saved as the canonical model for prediction.
    """
    t0 = time.time()
    panel = build_panel(cfg)
    feat_cols = list_feature_columns(panel)
    print(f"[xgbmodel.wf] using {len(feat_cols)} features")

    folds = walk_forward_folds(
        panel,
        fold_train_weeks = cfg['fold_train_weeks'],
        fold_val_weeks   = cfg['fold_val_weeks'],
        fold_test_weeks  = cfg['fold_test_weeks'],
        fold_step_weeks  = cfg['fold_step_weeks'],
        purge_days       = cfg['purge_days'],
        embargo_days     = cfg['embargo_days'],
        expanding        = cfg['expanding_train'],
        start_date       = pd.to_datetime(str(cfg['train_start'])),
    )
    print(f"[xgbmodel.wf] {summarize_folds(folds)}")

    per_fold = []
    oof_val_preds:  List[pd.DataFrame] = []
    oof_test_preds: List[pd.DataFrame] = []
    best_iters: List[int] = []

    fold_limit = cfg.get('max_folds', 0) or len(folds)
    run_folds  = folds[:fold_limit]

    for fold in run_folds:
        train_df, val_df, test_df = fold.slice(panel)
        print(f"\n[xgbmodel.wf] {fold.summary(train_df, val_df, test_df)}")
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print("  skipping fold — empty window after purge")
            continue

        model, preds, metrics = _fit_one(
            train_df, val_df, test_df, feat_cols, cfg, verbose_every=0
        )
        bi = getattr(model, 'best_iteration', None)
        if bi is not None:
            best_iters.append(bi)
        per_fold.append({'fold': fold.index, 'metrics': metrics, 'best_iteration': bi})

        preds['val' ]['fold'] = fold.index
        preds['test']['fold'] = fold.index
        oof_val_preds.append(preds['val'])
        oof_test_preds.append(preds['test'])

    if not per_fold:
        raise RuntimeError("walk_forward produced no folds with data")

    # Aggregate: NaN-safe mean / std of daily-IC across folds. Folds with
    # degenerate inputs (e.g. barely-trained model → constant preds → no
    # computable IC) yield NaN metrics; we skip those via np.nanmean.
    def _agg(metric_name: str, split: str) -> Tuple[float, float, int]:
        vals = np.array([
            f['metrics'][split][metric_name]
            for f in per_fold if split in f['metrics']
        ], dtype='float64')
        n_valid = int(np.isfinite(vals).sum())
        if n_valid == 0:
            return float('nan'), float('nan'), 0
        return float(np.nanmean(vals)), float(np.nanstd(vals)), n_valid

    ic_val_m,   ic_val_s,   n_val_valid  = _agg('rank_ic', 'val')
    ic_test_m,  ic_test_s,  n_test_valid = _agg('rank_ic', 'test')
    rmse_test_m, rmse_test_s, _          = _agg('rmse',   'test')
    # Count positive-IC folds only among those that produced a valid IC
    pos_test = sum(1 for f in per_fold
                   if np.isfinite(f['metrics']['test']['rank_ic'])
                   and f['metrics']['test']['rank_ic'] > 0)

    print(f"\n[xgbmodel.wf] summary over {len(per_fold)} folds "
          f"({n_val_valid} val / {n_test_valid} test with valid IC):")
    print(f"  val  IC: mean={ic_val_m:+.4f}  std={ic_val_s:.4f}")
    print(f"  test IC: mean={ic_test_m:+.4f}  std={ic_test_s:.4f}  "
          f"(positive ratio = {pos_test}/{n_test_valid})")
    print(f"  test RMSE: mean={rmse_test_m:.4f}  std={rmse_test_s:.4f}")

    # ─── Persist OOF predictions & per-fold metrics ─────────────────────────
    model_dir = cfg.get('model_dir', MODEL_DIR)
    preds_dir = os.path.join(model_dir, 'xgb_preds')
    os.makedirs(preds_dir, exist_ok=True)
    pd.concat(oof_val_preds,  ignore_index=True).to_csv(
        os.path.join(preds_dir, 'val.csv'), index=False)
    pd.concat(oof_test_preds, ignore_index=True).to_csv(
        os.path.join(preds_dir, 'test.csv'), index=False)

    # ─── Refit canonical model on the full pool using average best_iteration ─
    # Robust stopping-point estimate: drop folds that triggered early-stopping
    # almost immediately (best_iter<min_iter_floor) — those folds failed to
    # learn anything useful and would pull the median toward 0. Then take the
    # median of the survivors. If ALL folds are pathological, fall back to
    # 1/10 of the requested n_estimators so we at least ship a real model.
    MIN_ITER_FLOOR  = cfg.get('canonical_min_iter_floor', 20)
    ABS_FLOOR       = cfg.get('canonical_abs_floor',      100)
    configured_n    = cfg['xgb_params'].get('n_estimators', 1000)

    good_iters = [b for b in best_iters if b is not None and b >= MIN_ITER_FLOOR]
    if good_iters:
        canonical_n = int(np.median(good_iters)) + 1
    else:
        canonical_n = max(ABS_FLOOR, configured_n // 10)
        print(f"[xgbmodel.wf] WARN: no folds reached best_iter>={MIN_ITER_FLOOR}; "
              f"falling back to canonical_n={canonical_n}")
    # Always enforce the absolute floor so we never save a 3-tree model
    canonical_n = max(canonical_n, ABS_FLOOR)
    print(f"[xgbmodel.wf] best_iterations: n={len(best_iters)}  "
          f"survivors≥{MIN_ITER_FLOOR}: {len(good_iters)}  "
          f"median_survivor={int(np.median(good_iters)) if good_iters else 'n/a'}  "
          f"→ canonical_n={canonical_n}")

    print(f"\n[xgbmodel.wf] refitting canonical model on full pool with "
          f"n_estimators={canonical_n}")
    final_params = dict(cfg['xgb_params'])
    final_params['n_estimators'] = canonical_n
    final_params.pop('early_stopping_rounds', None)
    if cfg.get('device') == 'cuda':
        final_params['device'] = 'cuda'
    final_params['random_state'] = cfg.get('random_seed', 42)

    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(panel[feat_cols], panel['target'], verbose=False)

    model_path = os.path.join(model_dir, 'xgb_pct_chg.json')
    feats_path = os.path.join(model_dir, 'xgb_pct_chg.features.json')
    meta_path  = os.path.join(model_dir, 'xgb_pct_chg.meta.json')
    final_model.save_model(model_path)
    with open(feats_path, 'w', encoding='utf-8') as f:
        json.dump(feat_cols, f, indent=2)

    booster = final_model.get_booster()
    total_gain = booster.get_score(importance_type='total_gain')
    gain       = booster.get_score(importance_type='gain')
    fi = sorted(
        [{'feature': k, 'gain': float(gain[k]), 'total_gain': float(total_gain.get(k, 0.0))}
         for k in total_gain],
        key=lambda r: -r['total_gain'],
    )

    meta = {
        'mode':            'walk_forward',
        'target_mode':     cfg.get('target_mode'),
        'forward_window':  cfg.get('forward_window'),
        'n_features':      len(feat_cols),
        'canonical_n_estimators': canonical_n,
        'fold_config': {
            'train_weeks':   cfg['fold_train_weeks'],
            'val_weeks':     cfg['fold_val_weeks'],
            'test_weeks':    cfg['fold_test_weeks'],
            'step_weeks':    cfg['fold_step_weeks'],
            'purge_days':    cfg['purge_days'],
            'embargo_days':  cfg['embargo_days'],
            'expanding':     cfg['expanding_train'],
            'n_folds_run':   len(per_fold),
        },
        'metric_summary': {
            'val_rank_ic_mean':   ic_val_m,
            'val_rank_ic_std':    ic_val_s,
            'test_rank_ic_mean':  ic_test_m,
            'test_rank_ic_std':   ic_test_s,
            'test_rmse_mean':     rmse_test_m,
            'test_rmse_std':      rmse_test_s,
            'test_ic_positive_ratio': sum(1 for f in per_fold
                                           if f['metrics']['test']['rank_ic'] > 0) / len(per_fold),
        },
        'per_fold':              per_fold,
        'feature_importance_top50': fi[:50],
        'xgb_params':            final_params,
        'total_seconds':         time.time() - t0,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[xgbmodel.wf] saved model → {model_path}")
    print(f"[xgbmodel.wf] total wall  = {time.time() - t0:.1f}s")
    return meta
