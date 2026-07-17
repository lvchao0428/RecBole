#!/usr/bin/env bash
# 新 watcher: 等 Grocery MV (PID=1500811) 完成 → p2_ablation → full_boost
set -euo pipefail
ROOT="/home/charlie/project/RecBole"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="logs/new_watcher_0713.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

GROCERY_PID=1500811

log "=== New Watcher Started: waiting for Grocery MV (PID=$GROCERY_PID) ==="
while kill -0 $GROCERY_PID 2>/dev/null; do
  sleep 60
done
log "=== Grocery MV complete. Writing queue complete marker. ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue 0712 Complete" >> logs/queue_20260712.log

log "=== Step 1: Advisor P(2) Ablation (Cross on/off, no-cross rerun) ==="
bash run_advisor_p2_ablation.sh >> logs/advisor_p2_ablation_nohup.log 2>&1
log "=== Step 1 DONE ==="

log "=== Collecting P(2) results ==="
python3 scripts/collect_p2_ablation.py 2>/dev/null | tee -a "$LOG" || log "collect_p2_ablation not found, skip"

log "=== Step 2: Beauty TS V2 Full Boost (cold+infer, WITH Cross) ==="
bash run_beauty_full_boost_v2.sh >> logs/beauty_full_boost_v2_nohup.log 2>&1
log "=== Step 2 DONE ==="

log "=== Step 3: Beauty TS V2 Full Boost + NO-CROSS (cb3+infer) ==="
bash run_beauty_full_boost_nocross_v2.sh >> logs/beauty_full_boost_nocross_v2_nohup.log 2>&1
log "=== Step 3 DONE ==="

log "=== Step 4: UniSRec Beauty seed=2025 (min5) ==="
bash run_5090_ts_unisrec_seed2025.sh >> logs/ts_unisrec_seed2025_nohup.log 2>&1
log "=== Step 4 DONE ==="

log "=== ALL COMPLETE ==="
