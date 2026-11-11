#!/usr/bin/env bash
# Main-text-first launcher (fix for orphaned reorder on 2026-08-18).
#
# Context: the running watcher (PID from 12:56) was started from an OLD
# watch_and_run_p3_fixed.sh that calls run_p3_fixed_0818.sh (Pure LLM eval)
# DIRECTLY and then EXITS -- it never invokes the reordered
# run_paper_rerun_queue_0818.sh. So Beauty main table would never start.
#
# This launcher waits for that Pure LLM eval to finish (avoid GPU contention),
# then runs the main-text-critical jobs in priority order.
# Pure LLM ([4]) is SKIPPED here because the old watcher already completed it
# (paper_recsys/purellm_hybrid_gts_last.csv is populated, 3-seed GTS-Last done).

set -euo pipefail
ROOT="/home/charlie/project/RecBole"
cd "$ROOT"
LOG="$ROOT/logs/maintext_first_0818.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== MAIN-TEXT-FIRST LAUNCHER START ========"

# 1) Wait for the in-flight Pure LLM eval (old watcher's run_p3_fixed) to finish.
log "Waiting for run_p3_fixed_0818.sh / eval_gts_* to finish before training..."
while pgrep -f "run_p3_fixed_0818.sh" >/dev/null 2>&1 \
   || pgrep -f "eval_gts_last.py" >/dev/null 2>&1 \
   || pgrep -f "eval_gts_random.py" >/dev/null 2>&1; do
  sleep 30
done
# Also make sure no stray training is holding the GPU.
while pgrep -f "two_phase_train.py" >/dev/null 2>&1; do
  sleep 30
done
log "GPU idle confirmed. Starting main-text jobs."

# 2) [1] Beauty main table (MUST) -> tab:protocol/ndcg_hit/perseed/loov Beauty
log "[1] Beauty TF/LLM/MV cutoff-fit retrain + Last/Random + Beauty LOOV"
bash "$ROOT/scripts/run_beauty_ts_maintable_0818.sh" || log "WARN [1] Beauty returned non-zero"

# 3) [2] Early60-EQ (MUST) -> tab:early
log "[2] Early60-EQ fair retrain + Last"
bash "$ROOT/scripts/run_early60eq_fair_0818.sh" || log "WARN [2] Early60 returned non-zero"

# 4) [3] UniSRec Toys (recommended secondary column)
log "[3] UniSRec Toys cutoff-fit retrain + Last/Random"
bash "$ROOT/scripts/run_unisrec_toys_ts_0818.sh" || log "WARN [3] UniSRec Toys returned non-zero"

# 5) [5] Fair hybrid (Discussion; droppable). Kept last.
log "[5] Fair Hybrid TF+1view + Last/Random (Discussion only)"
bash "$ROOT/scripts/run_hybrid_fair_ts_0818.sh" || log "WARN [5] hybrid returned non-zero"

log "======== MAIN-TEXT-FIRST LAUNCHER ALL DONE ========"
log "[1] Beauty Last: paper_recsys/gts_last_eval_beauty_ts.csv"
log "[2] Early60 Last: paper_recsys/early60eq_gts_last_eval_ts.csv"
log "[3] UniSRec Toys Last: paper_recsys/gts_last_eval_unisrec_toys_ts.csv"
log "[5] Hybrid Last: paper_recsys/hybrid_ts_gts_last.csv"
log "NOTE: [4] Pure LLM already done by old watcher -> paper_recsys/purellm_hybrid_gts_last.csv"
