#!/usr/bin/env bash
# Wait for current log10 two_phase_train, then stop run_log10_gru4rec_baselines.sh
#
# Usage (log10):
#   nohup bash scripts/log10_pipeline_pause_after_current.sh \
#     >> logs/log10_pipeline_pause_nohup.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/logs/log10_pipeline_pause.log"
POLL_SEC="${POLL_SEC:-5}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

TARGET_PID="${1:-}"
if [[ -z "$TARGET_PID" ]]; then
  TARGET_PID="$(pgrep -fo "python .*two_phase_train.py" || true)"
fi

if [[ -z "$TARGET_PID" ]]; then
  log "No active two_phase_train found. Exiting."
  exit 0
fi

log "======== log10 pause watcher started (target_pid=$TARGET_PID) ========"

while kill -0 "$TARGET_PID" >/dev/null 2>&1; do
  sleep "$POLL_SEC"
done

log "Current log10 training job finished (target_pid=$TARGET_PID)."

PIPE_PID=""
while read -r pid cmd; do
  [[ "$cmd" == *"log10_pipeline_pause"* ]] && continue
  if [[ "$cmd" == *"run_log10_gru4rec_baselines"* ]]; then
    PIPE_PID="$pid"
    break
  fi
done < <(pgrep -af "run_log10_gru4rec_baselines" || true)

if [[ -n "$PIPE_PID" ]]; then
  log "Stopping log10 baseline queue PID=$PIPE_PID..."
  kill "$PIPE_PID" 2>/dev/null || true
  sleep 3
  kill -9 "$PIPE_PID" 2>/dev/null || true
fi

log "======== Safe to power off log10 now ========"
log "Resume: nohup bash run_log10_gru4rec_baselines_from_2024.sh >> logs/log10_gru4rec_resume_from_2024_nohup.log 2>&1 &"
