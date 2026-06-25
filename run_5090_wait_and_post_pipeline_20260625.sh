#!/usr/bin/env bash
# Wait for the main pipeline to finish, then auto-start Phase 3 (mechanism + analysis).
#
# Usage (5090, while main pipeline is running):
#   nohup bash run_5090_wait_and_post_pipeline_20260625.sh \
#     >> logs/wait_post_pipeline_20260625_nohup.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG="logs/wait_post_pipeline_20260625.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Waiting for main pipeline (run_5090_main_pipeline_20260625*) to exit..."
while pgrep -f "run_5090_main_pipeline_20260625" >/dev/null 2>&1; do
  sleep 120
done

# Extra guard: wait until Grocery MV seed42 python job finishes if still running
while pgrep -f "multiview_v3_grocery_stratified.*seed.?42" >/dev/null 2>&1; do
  log "  ... still waiting for Grocery MV seed=42 training"
  sleep 120
done

log "Main pipeline finished. Starting post pipeline..."
exec bash "$ROOT/run_5090_post_main_pipeline_20260625.sh"
