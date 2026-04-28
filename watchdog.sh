#!/bin/bash
# Watches pipeline + engine logs every 10s. Emits a single line on:
#   - any new error / Traceback in any log
#   - any engine starting (so we know the GPU stream advances)
#   - any engine summary completing
#   - the final pipeline DONE
# Continues until pipeline DONE or user kills.

set +e   # don't die on grep no-match

LOGS="/tmp/pipeline_v2.log /tmp/eng_xgb_shallow.log /tmp/eng_xgb_default.log /tmp/eng_xgb_strong_reg.log /tmp/eng_xgb_deep.log /tmp/eng_catboost.log /tmp/eng_lightgbm.log"

last_state=""
prev_lines=""

while true; do
    state=""

    # Check process count
    n_py=$(tasklist 2>/dev/null | grep -c -i python.exe)
    state="$state procs=$n_py"

    # Last meaningful line from each log
    for log in $LOGS; do
        [ -f "$log" ] || continue
        last=$(tail -1 "$log" 2>/dev/null | head -c 200 | tr -d '\r\n')
        name=$(basename "$log" .log)
        state="$state | $name=${last:0:80}"
    done

    # Search all logs for fatal patterns since last check
    cur_errors=$(grep -h -E "Traceback|Error:|Exception|FAILED" $LOGS 2>/dev/null | wc -l)

    # Emit only if state changed or first iteration
    if [ "$state" != "$last_state" ]; then
        echo "[$(date '+%H:%M:%S')] err_lines=$cur_errors $state"
        last_state="$state"
    fi

    # Pipeline complete?
    if grep -q "=== \[pipeline\] .* — DONE" /tmp/pipeline_v2.log 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] PIPELINE DONE"
        break
    fi

    sleep 10
done
