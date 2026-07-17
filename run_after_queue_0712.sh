#!/usr/bin/env bash
# Wait for queue_20260712 to finish, then run in order:
#   1. Advisor P(2) Ablation  (Cross on/off × TF-IDF/LLM,  ~3h)
#   2. Beauty Full-Boost V2   (cold+infer boost all on,      ~7h)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

LOG="logs/run_after_queue_0712.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Watcher started: waiting for queue_20260712 to complete..."
while true; do
  if grep -q "Queue 0712 Complete" logs/queue_20260712.log 2>/dev/null; then
    log "queue_20260712 finished."
    break
  fi
  sleep 60
done

log "=== Step 1: Advisor P(2) Ablation (Cross on/off) ==="
bash run_advisor_p2_ablation.sh >> logs/advisor_p2_ablation_nohup.log 2>&1
log "=== Step 1 DONE ==="

log "=== Collecting P(2) ablation results ==="
python3 scripts/collect_p2_ablation.py | tee -a "$LOG"

log "=== Step 2: Beauty TS V2 Full Boost (cold+infer) ==="
bash run_beauty_full_boost_v2.sh >> logs/beauty_full_boost_v2_nohup.log 2>&1
log "=== Step 2 DONE ==="

log "=== ALL COMPLETE ==="
