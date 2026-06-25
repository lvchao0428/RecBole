#!/usr/bin/env bash
# Wait for log10 GRU4Rec job and pull saved/ + run_metrics/ to 5090.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/log10_env.sh"

PID_FILE="${1:-}"
if [[ -z "$PID_FILE" ]]; then
  PID_FILE=$(ls -t "$ROOT/logs"/.log10_gru4rec_pid_*.txt 2>/dev/null | head -1)
fi
REMOTE="${LOG10_SSH}:${LOG10_PROJECT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ -n "$PID_FILE" && -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  log "Waiting for log10 GRU4Rec PID=$PID ..."
  while ssh "$LOG10_SSH" "kill -0 $PID 2>/dev/null"; do
    sleep 120
  done
  log "log10 GRU4Rec process finished"
else
  log "No PID file — waiting for log10 python to idle (120s poll)"
  while ssh "$LOG10_SSH" "pgrep -f 'run_log10_gru4rec_baselines|two_phase_train.*GRU4Rec' >/dev/null"; do
    sleep 120
  done
fi

log ">>> rsync log10 → 5090 (gru4rec saved + run_metrics + logs)"
rsync -avz "${REMOTE}/saved/gru4rec_"* "${ROOT}/saved/" 2>/dev/null || true
rsync -avz "${REMOTE}/run_metrics/" "${ROOT}/run_metrics/" 2>/dev/null || true
rsync -avz "${REMOTE}/logs/log10_gru4rec_"*.log "${ROOT}/logs/" 2>/dev/null || true

log "✅ log10 GRU4Rec results pulled to 5090"
