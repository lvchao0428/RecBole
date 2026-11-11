#!/usr/bin/env bash
# Guarded queue: Beauty is already in-flight under watcher PID 642904.
# If the after-beauty follow-on is alive, this script must no-op so a stale
# watcher cannot start a second copy of Early60 / UniSRec / Hybrid.

set -euo pipefail
ROOT="/home/charlie/project/RecBole"
cd "$ROOT"
LOG="$ROOT/logs/paper_rerun_queue_0818.log"
log() { echo "[$(date "+%F %H:%M:%S")] $*" | tee -a "$LOG"; }

if pgrep -f "run_after_beauty_0819.sh" >/dev/null 2>&1 \
   || pgrep -f "run_early60eq_fair_0818.sh" >/dev/null 2>&1; then
  log "SKIP: follow-on or Early60 already running. Refusing to duplicate."
  exit 0
fi

log "======== PAPER RERUN QUEUE (Beauty SKIPPED, already trained) ========"
log "[2] Early60-EQ fair retrain + Last"
bash "$ROOT/scripts/run_early60eq_fair_0818.sh"
log "[3] UniSRec Toys cutoff-fit"
bash "$ROOT/scripts/run_unisrec_toys_ts_0818.sh"
log "[5] Fair Hybrid"
bash "$ROOT/scripts/run_hybrid_fair_ts_0818.sh"
log "======== PAPER RERUN QUEUE ALL DONE ========"
