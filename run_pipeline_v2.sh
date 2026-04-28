#!/bin/bash
# Pipeline v2: fastest-first GBM training, with LightGBM (CPU) running in
# parallel to the GPU stream. Each model's summary metrics are printed as it
# completes. After all 6 GBM finish, build ensemble_mean and run multi-model
# backtest + per-model TP/SL tuning.
#
# Logs to /tmp/pipeline_v2.log (top-level), each engine to its own log too.

set -e
LOG=/tmp/pipeline_v2.log
exec > "$LOG" 2>&1

cd /d/didi/stock/tushare

# ── CPU stream (LightGBM) ── runs in background concurrent with GPU
echo "=== [pipeline] $(date) — GPU+CPU streams in parallel ==="
./venv/Scripts/python -u -m model_compare.main --engine lightgbm \
    > /tmp/eng_lightgbm.log 2>&1 &
LGB_PID=$!
echo "[pipeline] LightGBM (CPU) PID=$LGB_PID"

# ── GPU stream: fastest first ── shallow → default → strong_reg → catboost → deep
GPU_ENGINES=(xgb_shallow xgb_default xgb_strong_reg catboost xgb_deep)
for engine in "${GPU_ENGINES[@]}"; do
    echo ""
    echo "=== [pipeline] $(date) — GPU: $engine ==="
    ./venv/Scripts/python -u -m model_compare.main --engine "$engine" --device cuda \
        > /tmp/eng_${engine}.log 2>&1
    echo "[pipeline] $engine done; tail of summary:"
    grep -A4 "summary over" /tmp/eng_${engine}.log | tail -5
done

# ── Wait for LightGBM ──
echo ""
echo "=== [pipeline] $(date) — waiting for LightGBM ==="
wait $LGB_PID || true
echo "[pipeline] LightGBM done; tail of summary:"
grep -A4 "summary over" /tmp/eng_lightgbm.log | tail -5

# ── Build ensemble ──
echo ""
echo "=== [pipeline] $(date) — Phase 2: ensemble_mean ==="
./venv/Scripts/python -u -m model_compare.ensemble \
    --models xgb_default xgb_shallow xgb_deep xgb_strong_reg lightgbm catboost

# ── Multi-model backtest + per-model tuning ──
echo ""
echo "=== [pipeline] $(date) — Phase 3: backtest + tune ==="
./venv/Scripts/python -u -m model_compare.run_backtests --tune \
    --models xgb_default xgb_shallow xgb_deep xgb_strong_reg lightgbm catboost ensemble_mean

# ── Dashboard refresh ──
echo ""
echo "=== [pipeline] $(date) — Phase 4: dashboard ==="
./venv/Scripts/python -m dashboard.build_predictions
DASH_PW='Mpt20250422!' ./venv/Scripts/python -m dashboard.package_secure_predictions

echo ""
echo "=== [pipeline] $(date) — DONE ==="
