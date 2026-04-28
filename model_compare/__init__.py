"""
model_compare/ — apples-to-apples comparison of multiple regression models
on the same XGB feature panel, walk-forward CV, and prediction CSV schema.

Engines (each subclasses `Engine`):
  - xgb_default            (baseline, mirrors xgbmodel/)
  - xgb_shallow / deep / strong_reg  (hyperparameter variants)
  - lightgbm
  - catboost
  - transformer_reg        (pure encoder-only, regression)
  - tft                    (pure Lim 2019 TFT, regression)

Each engine writes predictions to:
    stock_data/models_<name>/xgb_preds/test.csv
    stock_data/models_<name>/meta.json

so the existing backtest (`backtest.xgb_markowitz`) and dashboard infra
work unchanged — they just point at a different test.csv.
"""
