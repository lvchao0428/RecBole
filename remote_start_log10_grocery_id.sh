#!/usr/bin/env bash
# 5090 上启动 log10 Grocery ID-only（后台）

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/log10_env.sh"

SEED="${SEED:-2024}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_grocery}"
LOG_NAME="log10_grocery_id_${RUN_ID}.log"

bash "$ROOT/scripts/sync_grocery_to_log10.sh"

echo ">>> Starting ID-only on log10 (SEED=$SEED)"
ssh "$LOG10_SSH" "cd ${LOG10_PROJECT} && \
  export PYTHONPATH=\$(pwd):\${PYTHONPATH:-} && \
  nohup env SEED=${SEED} GPU_ID=0 bash run_log10_grocery_id_only.sh \
    > logs/${LOG_NAME} 2>&1 & echo LOG10_PID=\$!"

echo "log10 log: ${LOG10_PROJECT}/logs/${LOG_NAME}"
echo "$LOG_NAME" > "$ROOT/logs/.log10_grocery_id_logname_${RUN_ID}.txt"
