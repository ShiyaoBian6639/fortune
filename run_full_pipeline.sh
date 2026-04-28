#!/bin/bash
# Full pipeline after delisted stocks were added:
#   1. Retrain 6 GBM models on expanded universe
#   2. Build mean ensemble
#   3. Backtest all 7 with realistic exits + per-model tuning
#   4. Refresh predictions dashboard
#
# Logs to /tmp/full_pipeline.log

set -e
LOG=/tmp/full_pipeline.log
exec > "$LOG" 2>&1

cd /d/didi/stock/tushare

echo "=== [pipeline] $(date) — Phase 1/4: retrain 6 GBM on expanded universe ==="
./venv/Scripts/python -u -m model_compare.main \
    --engines xgb_default xgb_shallow xgb_deep xgb_strong_reg lightgbm catboost \
    --device cuda

echo ""
echo "=== [pipeline] $(date) — Phase 2/4: ensemble_mean from 6 GBM ==="
./venv/Scripts/python -u -m model_compare.ensemble \
    --models xgb_default xgb_shallow xgb_deep xgb_strong_reg lightgbm catboost

echo ""
echo "=== [pipeline] $(date) — Phase 3/4: multi-model backtest + per-model tuning ==="
./venv/Scripts/python -u -m model_compare.run_backtests --tune \
    --models xgb_default xgb_shallow xgb_deep xgb_strong_reg lightgbm catboost ensemble_mean

echo ""
echo "=== [pipeline] $(date) — Phase 4/4: refresh dashboard ==="
./venv/Scripts/python -m dashboard.build_predictions
DASH_PW='Mpt20250422!' ./venv/Scripts/python -m dashboard.package_secure_predictions

echo ""
echo "=== [pipeline] $(date) — DONE ==="
