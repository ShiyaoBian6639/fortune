#!/bin/bash
# Auto-chain runner v2: waits for transformer_reg's meta.json to appear, then
# launches tft → ensemble → dashboard refresh. Logs to /tmp/chain_run.log.

set -e
LOG=/tmp/chain_run.log
exec > "$LOG" 2>&1

echo "[chain] $(date) — waiting for transformer_reg meta.json"
until [ -f stock_data/models_transformer_reg/meta.json ]; do
    sleep 60
done
echo "[chain] $(date) — transformer_reg done"

echo "[chain] $(date) — launching tft (GPU exclusive)"
./venv/Scripts/python -u -m model_compare.main --engine tft --device cuda
echo "[chain] $(date) — tft done"

echo "[chain] $(date) — running ensemble + comparison"
./venv/Scripts/python -u -m model_compare.ensemble \
    --models xgb_default xgb_shallow xgb_deep xgb_strong_reg \
             lightgbm catboost transformer_reg tft

echo "[chain] $(date) — rebuilding predictions dashboard"
./venv/Scripts/python -m dashboard.build_predictions

echo "[chain] $(date) — encrypting secure variant"
DASH_PW='Mpt20250422!' ./venv/Scripts/python -m dashboard.package_secure_predictions

echo "[chain] $(date) — DONE"
