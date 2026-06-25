#!/usr/bin/env bash
# Start GRU4Rec ID+TF-IDF baseline jobs on log10 (background).
# Run on 5090 before Phase 4 text runs.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_NAME="log10_gru4rec_baselines_${RUN_ID}.log"
PID_FILE="$ROOT/logs/.log10_gru4rec_pid_${RUN_ID}.txt"
LOG10_LOG="${LOG10_PROJECT}/logs/${LOG_NAME}"

echo ">>> Preflight log10 datasets"
bash "$ROOT/scripts/verify_log10_datasets.sh" || {
  echo ">>> Sync missing data..."
  bash "$ROOT/scripts/sync_log10_gru4rec_data.sh"
  bash "$ROOT/scripts/verify_log10_datasets.sh"
}

echo ">>> Starting log10 GRU4Rec baselines (all seeds, background)"
ssh "$LOG10_SSH" "mkdir -p ${LOG10_PROJECT}/logs"
ssh "$LOG10_SSH" "cd ${LOG10_PROJECT} && nohup bash run_log10_gru4rec_baselines.sh --all-seeds \
  > logs/${LOG_NAME} 2>&1 & echo \$!" | tee "$PID_FILE"

echo "log10 PID: $(cat "$PID_FILE")"
echo "log10 log: ${LOG10_LOG}"
echo "$RUN_ID" > "$ROOT/logs/.log10_gru4rec_run_id.txt"
echo "$LOG_NAME" > "$ROOT/logs/.log10_gru4rec_logname.txt"
