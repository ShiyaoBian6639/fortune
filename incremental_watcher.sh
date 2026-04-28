#!/bin/bash
# Per-engine incremental backtest + dashboard refresh — V2.
# Uses meta.json modification time relative to a lock-time anchor written
# to /tmp/.watcher_anchor at watcher start.  Avoids stale-meta false triggers.

set +e
LOG=/tmp/incremental_watcher.log
exec >> "$LOG" 2>&1

cd /d/didi/stock/tushare

# Anchor: any meta.json older than this is "stale from previous run"
# Set to current time minus 60s (in case xgb_shallow just finished a moment ago)
ANCHOR=$(($(date +%s) - 60))
echo "[$(date +%H:%M:%S)] [watcher v2] armed; anchor mtime > $ANCHOR (== $(date -d @$ANCHOR +%H:%M:%S))"

ENGINES=(xgb_shallow xgb_default xgb_strong_reg catboost xgb_deep lightgbm)

while true; do
    for engine in "${ENGINES[@]}" ensemble_mean; do
        flag="/tmp/.bt_done_${engine}"
        [ -f "$flag" ] && continue

        meta="stock_data/models_${engine}/meta.json"
        preds="stock_data/models_${engine}/xgb_preds/test.csv"
        [ -f "$meta" ] && [ -f "$preds" ] || continue

        # Freshness gate: meta.json must be newer than anchor
        meta_mtime=$(stat -c %Y "$meta" 2>/dev/null)
        if [ -z "$meta_mtime" ] || [ "$meta_mtime" -lt "$ANCHOR" ]; then
            continue
        fi

        echo ""
        echo "[$(date +%H:%M:%S)] [watcher v2] === $engine ready (meta.mtime=$(date -d @$meta_mtime +%H:%M:%S)) ==="
        ./venv/Scripts/python -m model_compare.run_backtests --models "$engine"

        ./venv/Scripts/python -m dashboard.build_predictions
        DASH_PW='Mpt20250422!' ./venv/Scripts/python -m dashboard.package_secure_predictions

        touch "$flag"
        echo "[$(date +%H:%M:%S)] [watcher v2] $engine fully processed"
    done

    if grep -q "=== \[pipeline\] .* — DONE" /tmp/pipeline_v2.log 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] [watcher v2] pipeline complete, exiting"
        break
    fi
    sleep 30
done
