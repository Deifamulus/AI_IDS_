#!/bin/bash


PYTHON_PATH="/mnt/d/Projects/ai_ids_project/venv/bin/python"
SCRIPT_PATH="/mnt/d/Projects/ai_ids_project/src/attack_simulation_script.py"


TOTAL_RUNS=15

SLEEP_INTERVAL=120

echo "[*] Starting spaced attack simulation for $TOTAL_RUNS runs..."

for ((i=1; i<=TOTAL_RUNS; i++))
do
    echo ""
    echo "[+] Run $i of $TOTAL_RUNS: $(date)"
    $PYTHON_PATH $SCRIPT_PATH

    if [ "$i" -lt "$TOTAL_RUNS" ]; then
        echo "[*] Sleeping for $SLEEP_INTERVAL seconds before next run..."
        sleep $SLEEP_INTERVAL
    fi
done

echo ""
echo "[✓] All $TOTAL_RUNS attack simulation batches complete."
