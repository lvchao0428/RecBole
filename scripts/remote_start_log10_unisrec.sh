#!/usr/bin/env bash
# 5090 → start UniSRec on log10 (low-VRAM task), then optionally wait+sync
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"
LOG10_ABS="/home/charlie/project/RecBole"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync UniSRec runner → log10"
rsync -avz "$ROOT/run_log10_ts_unisrec_seed2025.sh" "${LOG10_SSH}:${LOG10_ABS}/"
ssh "$LOG10_SSH" "chmod +x ${LOG10_ABS}/run_log10_ts_unisrec_seed2025.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start UniSRec on log10"
ssh "$LOG10_SSH" "cd ${LOG10_ABS} && nohup bash run_log10_ts_unisrec_seed2025.sh > logs/ts_unisrec_seed2025_nohup.log 2>&1 & echo PID=\$!"
sleep 2
ssh "$LOG10_SSH" "ps aux | grep -E 'run_log10_ts_unisrec|two_phase_train' | grep -v grep | head -5"
echo "✅ log10 UniSRec launched"
