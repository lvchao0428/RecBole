#!/usr/bin/env bash
# BC phaseb50 跑完后自动接上 Grocery 流水线
#
# Usage (5090):
#   nohup bash run_5090_queue_after_bc_grocery.sh \
#     > logs/queue_after_bc_grocery_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-120}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_grocery}"
SEED="${SEED:-2024}"
LOG="logs/queue_after_bc_grocery.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== Wait for BC queue ========"

while pgrep -f "run_5090_queue_bc_four_configs_phaseb50" >/dev/null 2>&1; do
  log "BC queue running — sleep ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

while pgrep -f "two_phase_train.py.*book-crossing|run_recbole.py.*book-crossing" >/dev/null 2>&1; do
  log "BC training still active — sleep ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

log "BC complete — starting Grocery pipeline RUN_ID=$RUN_ID SEED=$SEED"

env RUN_ID="$RUN_ID" SEED="$SEED" \
  bash "$ROOT/run_5090_grocery_pipeline.sh" \
  >> "logs/grocery_pipeline_${RUN_ID}.log" 2>&1

log "======== All done (BC + Grocery) ========"
