#!/bin/bash
# Wait for all 6 static-feature streams to finish, then auto-launch pipeline_v2.

set -e
LOG=/tmp/auto_launch.log
exec > "$LOG" 2>&1

cd /d/didi/stock/tushare

echo "[auto] $(date) — waiting for static-feature downloads to complete"

# Wait for all 6 expected outputs
expected=(
    stock_data/static_features/stock_company.csv
    stock_data/static_features/stk_managers_summary.csv
    stock_data/static_features/stk_holdernumber.csv
    stock_data/static_features/top10_holders_summary.csv
    stock_data/static_features/index_member_flags.csv
    stock_data/identity_breakpoints.csv
)
while true; do
    n=0
    for p in "${expected[@]}"; do
        [ -f "$p" ] && n=$((n+1))
    done
    if [ "$n" -ge 6 ]; then
        echo "[auto] $(date) — all 6 outputs present"
        break
    fi
    sleep 30
done

# Force panel cache rebuild (xgbmodel doesn't have a cache by default but be safe)
rm -rf stock_data/cache 2>/dev/null || true

echo "[auto] $(date) — launching pipeline_v2"
bash run_pipeline_v2.sh

echo "[auto] $(date) — DONE"
