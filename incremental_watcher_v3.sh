#!/bin/bash
# Per-engine incremental backtest + dashboard refresh — V3 (PIT pipeline).
# Same logic as v2 but watches pipeline_v3.log and includes the 8 engines
# that v3 produces (6 GBM + transformer_reg + tft).

set +e
LOG=/tmp/incremental_watcher_v3.log
exec >> "$LOG" 2>&1

cd /d/didi/stock/tushare

ANCHOR=$(($(date +%s) - 60))
echo "[$(date +%H:%M:%S)] [watcher v3] armed; anchor mtime > $ANCHOR (== $(date -d @$ANCHOR +%H:%M:%S))"

ENGINES=(xgb_shallow xgb_default xgb_strong_reg catboost xgb_deep lightgbm transformer_reg tft)

while true; do
    for engine in "${ENGINES[@]}" ensemble_mean; do
        flag="/tmp/.bt_done_v3_${engine}"
        [ -f "$flag" ] && continue

        meta="stock_data/models_${engine}/meta.json"
        preds="stock_data/models_${engine}/xgb_preds/test.csv"
        [ -f "$meta" ] && [ -f "$preds" ] || continue

        meta_mtime=$(stat -c %Y "$meta" 2>/dev/null)
        if [ -z "$meta_mtime" ] || [ "$meta_mtime" -lt "$ANCHOR" ]; then
            continue
        fi

        echo ""
        echo "[$(date +%H:%M:%S)] [watcher v3] === $engine ready (meta.mtime=$(date -d @$meta_mtime +%H:%M:%S)) ==="
        ./venv/Scripts/python -m model_compare.run_backtests --models "$engine"

        ./venv/Scripts/python -m dashboard.build_predictions
        DASH_PW='Mpt20250422!' ./venv/Scripts/python -m dashboard.package_secure_predictions

        touch "$flag"
        echo "[$(date +%H:%M:%S)] [watcher v3] $engine fully processed"
    done

    if grep -q "=== \[pipeline v3\] .* — DONE" /tmp/pipeline_v3.log 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] [watcher v3] pipeline complete, exiting"
        break
    fi
    sleep 30
done
