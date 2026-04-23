"""
Load a trained XGBRegressor and predict next-day pct_chg.

Two modes:
  - predict_latest(cfg): predict for the most recent trade_date in the panel
    (live signal — what the model thinks every stock will do tomorrow)
  - predict_test(cfg): reload model + rebuild panel, emit per-stock predictions
    for the held-out test window
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import MODEL_DIR, PREDICT_OUT
from .data_loader import build_panel, list_feature_columns


def _load_model(cfg: dict) -> Tuple[xgb.XGBRegressor, List[str]]:
    model_dir  = cfg.get('model_dir', MODEL_DIR)
    model_path = os.path.join(model_dir, 'xgb_pct_chg.json')
    feats_path = os.path.join(model_dir, 'xgb_pct_chg.features.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path} — run train first.")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(feats_path, 'r', encoding='utf-8') as f:
        feats = json.load(f)
    return model, feats


def _align_features(panel: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    """Ensure the panel exposes every training feature; fill missing cols with 0."""
    missing = [c for c in feats if c not in panel.columns]
    if missing:
        print(f"[xgbmodel.predict] WARN: {len(missing)} missing features filled with 0 "
              f"(first few: {missing[:6]})")
        for c in missing:
            panel[c] = np.float32(0.0)
    return panel


def predict_latest(cfg: dict, out_path: str = PREDICT_OUT) -> pd.DataFrame:
    """Predict next-day pct_chg for the most recent trade_date in the panel."""
    model, feats = _load_model(cfg)
    panel = build_panel(cfg)
    panel = _align_features(panel, feats)

    last_date = panel['trade_date'].max()
    latest = panel[panel['trade_date'] == last_date].copy()
    if latest.empty:
        raise RuntimeError("No rows for the latest trade_date — check data freshness.")

    X = latest[feats]
    latest['pred_pct_chg_next'] = model.predict(X).astype('float32')

    keep = ['ts_code', 'trade_date', 'pred_pct_chg_next']
    out = latest[keep].sort_values('pred_pct_chg_next', ascending=False).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[xgbmodel.predict] {len(out):,} stocks predicted for {last_date.date()} "
          f"(next-day), saved to {out_path}")
    print(f"  top 10:\n{out.head(10).to_string(index=False)}")
    print(f"  bottom 10:\n{out.tail(10).to_string(index=False)}")
    return out


def predict_test(cfg: dict, out_path: str = None) -> pd.DataFrame:
    """Score all rows in the test window and dump predictions + true target."""
    model, feats = _load_model(cfg)
    panel = build_panel(cfg)
    panel = _align_features(panel, feats)

    test_start = pd.to_datetime(str(cfg['test_start']))
    test = panel[panel['trade_date'] >= test_start].copy()
    if test.empty:
        raise RuntimeError(f"No test rows after {test_start.date()}")

    test['pred'] = model.predict(test[feats]).astype('float32')
    keep = ['ts_code', 'trade_date', 'pred', 'target']
    out = test[keep].sort_values(['trade_date', 'ts_code']).reset_index(drop=True)

    if out_path is None:
        out_path = os.path.join(cfg.get('model_dir', MODEL_DIR), 'xgb_preds', 'test_full.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[xgbmodel.predict] wrote {len(out):,} test predictions → {out_path}")
    return out
