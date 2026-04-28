#!/bin/bash
# Pipeline v3: PIT index_member retrain. Fastest-first GBM training (matches v2
# layout) + LightGBM CPU in parallel + transformer + TFT at reduced max_folds.
# No TP/SL/max_hold tuning (per user). All 8 engines feed the multi-model
# backtest at the end.
#
# Logs to /tmp/pipeline_v3.log (top-level), each engine to its own log too.

set -e
LOG=/tmp/pipeline_v3.log
exec > "$LOG" 2>&1

cd /d/didi/stock/tushare

echo "=== [pipeline v3] $(date) — Phase A: GBM (GPU+CPU streams) ==="

# ── CPU stream (LightGBM) ──
./venv/Scripts/python -u -m model_compare.main --engine lightgbm \
    > /tmp/eng_lightgbm_v3.log 2>&1 &
LGB_PID=$!
echo "[pipeline v3] LightGBM (CPU) PID=$LGB_PID"

# ── GPU stream: fastest first ──
GPU_ENGINES=(xgb_shallow xgb_default xgb_strong_reg catboost xgb_deep)
for engine in "${GPU_ENGINES[@]}"; do
    echo ""
    echo "=== [pipeline v3] $(date) — GPU: $engine ==="
    ./venv/Scripts/python -u -m model_compare.main --engine "$engine" --device cuda \
        > /tmp/eng_${engine}_v3.log 2>&1
    echo "[pipeline v3] $engine done; tail of summary:"
    grep -A4 "summary over" /tmp/eng_${engine}_v3.log | tail -5
done

# ── Wait for LightGBM ──
echo ""
echo "=== [pipeline v3] $(date) — waiting for LightGBM ==="
wait $LGB_PID || true
echo "[pipeline v3] LightGBM done; tail of summary:"
grep -A4 "summary over" /tmp/eng_lightgbm_v3.log | tail -5

# ── Phase B: transformer + TFT (sequence engines, --max_folds 30) ──
# Walk-forward at reduced fold count: 30 folds × ~30 epochs at ~5-15 min/fold
# = ~3 hr per engine on GPU. Full 236 folds would be 20-60 GPU-hours.
echo ""
echo "=== [pipeline v3] $(date) — Phase B: transformer_reg ==="
./venv/Scripts/python -u -m model_compare.main \
    --engine transformer_reg --device cuda --max_folds 30 \
    > /tmp/eng_transformer_reg_v3.log 2>&1
echo "[pipeline v3] transformer_reg done; tail of summary:"
grep -A4 "summary over" /tmp/eng_transformer_reg_v3.log | tail -5

echo ""
echo "=== [pipeline v3] $(date) — Phase B: tft ==="
./venv/Scripts/python -u -m model_compare.main \
    --engine tft --device cuda --max_folds 30 \
    > /tmp/eng_tft_v3.log 2>&1
echo "[pipeline v3] tft done; tail of summary:"
grep -A4 "summary over" /tmp/eng_tft_v3.log | tail -5

# ── Build ensemble (mean over all 6 GBMs; sequence engines kept separate) ──
echo ""
echo "=== [pipeline v3] $(date) — Phase C: ensemble_mean ==="
./venv/Scripts/python -u -m model_compare.ensemble \
    --models xgb_default xgb_shallow xgb_deep xgb_strong_reg lightgbm catboost

# ── Multi-model backtest (no tuning per user) ──
echo ""
echo "=== [pipeline v3] $(date) — Phase D: backtest (no tuning) ==="
./venv/Scripts/python -u -m model_compare.run_backtests \
    --models xgb_default xgb_shallow xgb_deep xgb_strong_reg lightgbm catboost \
             transformer_reg tft ensemble_mean

# ── Dashboard refresh ──
echo ""
echo "=== [pipeline v3] $(date) — Phase E: dashboard ==="
./venv/Scripts/python -m dashboard.build_predictions
DASH_PW='Mpt20250422!' ./venv/Scripts/python -m dashboard.package_secure_predictions

echo ""
echo "=== [pipeline v3] $(date) — DONE ==="
