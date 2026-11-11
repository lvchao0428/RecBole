#!/usr/bin/env bash
# Continue main-text queue AFTER the in-flight Beauty retrain (PID 705779).
# Do NOT rerun Beauty. Launcher 702435 was killed to prevent a 12h duplicate.

set -euo pipefail
ROOT="/home/charlie/project/RecBole"
cd "$ROOT"
LOG="$ROOT/logs/after_beauty_0819.log"
log() { echo "[$(date "+%F %H:%M:%S")] $*" | tee -a "$LOG"; }

BEAUTY_PID="${1:-705779}"
log "======== AFTER-BEAUTY FOLLOW-ON START ========"
log "Waiting for Beauty script PID $BEAUTY_PID (and any leftover eval) ..."

if kill -0 "$BEAUTY_PID" 2>/dev/null; then
  while kill -0 "$BEAUTY_PID" 2>/dev/null; do
    sleep 30
  done
fi
# Beauty script may spawn eval after the last train; wait those out too.
while pgrep -f "eval_gts_last.py" >/dev/null 2>&1 \
   || pgrep -f "eval_gts_random.py" >/dev/null 2>&1 \
   || pgrep -f "analyze_leave_one_view_out.py" >/dev/null 2>&1 \
   || pgrep -f "two_phase_train.py" >/dev/null 2>&1; do
  sleep 30
done

log "Beauty + eval idle. Starting [2] Early60 (skip Beauty)."
bash "$ROOT/scripts/run_early60eq_fair_0818.sh" || log "WARN Early60 non-zero"
log "[3] UniSRec Toys"
bash "$ROOT/scripts/run_unisrec_toys_ts_0818.sh" || log "WARN UniSRec Toys non-zero"
log "[5] Fair Hybrid (Discussion; last)"
bash "$ROOT/scripts/run_hybrid_fair_ts_0818.sh" || log "WARN Hybrid non-zero"
log "======== AFTER-BEAUTY FOLLOW-ON ALL DONE ========"
