#!/usr/bin/env bash
# 5090 handoff: UniSRec 已改派到 log10（低显存）。
# 本脚本等待 log10 完成，再把结果同步回 5090，避免 5090 重复跑。
#
# 若想强制在 5090 本地重跑：FORCE_LOCAL=1 bash run_5090_ts_unisrec_seed2025.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"
LOG10_ABS="/home/charlie/project/RecBole"
LOG="logs/ts_unisrec_seed2025.log"
mkdir -p logs
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [[ "${FORCE_LOCAL:-0}" == "1" ]]; then
  log "FORCE_LOCAL=1 — falling back to local 5090 UniSRec (not implemented in this shim)"
  exit 1
fi

log "======== UniSRec delegated to log10 — wait + sync ========"

# Ensure log10 job started
if ! ssh "$LOG10_SSH" "pgrep -f run_log10_ts_unisrec_seed2025.sh >/dev/null 2>&1 || test -f ${LOG10_ABS}/logs/ts_unisrec_seed2025.DONE"; then
  log "log10 UniSRec not running / not done — starting it"
  bash "$ROOT/scripts/remote_start_log10_unisrec.sh" | tee -a "$LOG"
fi

log "Waiting for log10 DONE marker ..."
while true; do
  if ssh "$LOG10_SSH" "test -f ${LOG10_ABS}/logs/ts_unisrec_seed2025.DONE" 2>/dev/null; then
    log "log10 UniSRec DONE"
    break
  fi
  # still running?
  if ! ssh "$LOG10_SSH" "pgrep -f 'run_log10_ts_unisrec|two_phase_train.py.*UniSRec' >/dev/null 2>&1"; then
    # process gone but no DONE — check logs
    if ssh "$LOG10_SSH" "grep -q 'UniSRec Beauty seed=2025 COMPLETE' ${LOG10_ABS}/logs/ts_unisrec_seed2025.log 2>/dev/null"; then
      ssh "$LOG10_SSH" "touch ${LOG10_ABS}/logs/ts_unisrec_seed2025.DONE"
      log "COMPLETE found in log — marked DONE"
      break
    fi
    log "ERROR: log10 UniSRec process exited without DONE"
    ssh "$LOG10_SSH" "tail -30 ${LOG10_ABS}/logs/ts_unisrec_seed2025.log 2>/dev/null" | tee -a "$LOG" || true
    exit 1
  fi
  sleep 60
done

log "Syncing log10 → 5090 ..."
bash "$ROOT/scripts/sync_log10_results_to_5090.sh" | tee -a "$LOG"
log "======== UniSRec handoff COMPLETE (results on 5090) ========"
