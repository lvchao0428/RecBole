#!/usr/bin/env bash
# 5090 串行：book-crossing 四配置 phaseb50（与 Food 协议对齐）
# ID-only → TF-IDF → single-view → MV
#
# Usage:
#   nohup bash run_5090_queue_bc_four_configs_phaseb50.sh \
#     > logs/queue_bc_four_phaseb50_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_bc_four_phaseb50}"
LOG="${LOG:-logs/queue_bc_four_phaseb50.log}"
POLL_SEC="${POLL_SEC:-120}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
METRIC_BASELINE="${METRIC_BASELINE:-0.028}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
export PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
export PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
export CHECKPOINT_TAG="${CHECKPOINT_TAG:-phaseb50}"

mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$RUN_ID] $*" | tee -a "$LOG"
}

gpu_busy() {
  pgrep -f "two_phase_train.py|run_recbole.py" >/dev/null 2>&1
}

wait_gpu() {
  while gpu_busy; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null || echo "?")
    log "GPU busy (util=${util}%) — sleep ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

log "======== BC four-config phaseb50 queue ========"
log "SEED=$SEED GPU=$GPU_ID PHASE_A/B=${PHASE_A_EPOCHS}/${PHASE_B_EPOCHS} METRIC_BASELINE=$METRIC_BASELINE"

wait_gpu

if [[ "$SKIP_BASELINE" != "1" ]]; then
  log ">>> 1/4 ID-only baseline (50ep)"
  env SEED="$SEED" GPU_ID="$GPU_ID" \
    bash "$ROOT/run50epBase_v3_book_crossing_stratified.sh" \
    >> "logs/exp_bc_baseline_phaseb50_${RUN_ID}.log" 2>&1
else
  log ">>> 1/4 baseline SKIP"
fi

wait_gpu
log ">>> 2/4 TF-IDF"
env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
  bash "$ROOT/two_phase_run_tfidf_v3_book_crossing_stratified.sh" \
  >> "logs/exp_bc_tfidf_phaseb50_${RUN_ID}.log" 2>&1

wait_gpu
log ">>> 3/4 single-view (TF-IDF+LLM)"
env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
  bash "$ROOT/two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh" \
  >> "logs/exp_bc_llm_phaseb50_${RUN_ID}.log" 2>&1

wait_gpu
log ">>> 4/4 multi-view"
env SEED="$SEED" GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
  bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b.sh" \
  >> "logs/exp_bc_mv_phaseb50_${RUN_ID}.log" 2>&1

log "======== BC four-config phaseb50 complete ========"
