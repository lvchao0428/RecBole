#!/usr/bin/env bash
# 5090 排队：等 GPU 空闲后跑 book-crossing MV（Phase-B=50ep）
#
# ⚠️  Food 为当前 P0 主线。Food 四配置完成前请勿启动本脚本。
#     误启会抢占 GPU，导致 Food TF-IDF/LLM/MV 中断。
#
# Usage:
#   nohup bash run_5090_queue_bc_mv_phaseb50.sh > logs/queue_bc_mv_phaseb50_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d)_bc_mv_phaseb50}"
LOG="${LOG:-logs/queue_bc_mv_phaseb50.log}"
POLL_SEC="${POLL_SEC:-120}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
METRIC_BASELINE="${METRIC_BASELINE:-0.028}"

mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$RUN_ID] $*" | tee -a "$LOG"
}

gpu_busy() {
  pgrep -f "two_phase_train.py|run_recbole.py" >/dev/null 2>&1
}

log "======== Queue: book-crossing MV phaseb50 ========"
log "Wait until no training process (poll ${POLL_SEC}s)..."
log "Target: SEED=$SEED METRIC_BASELINE=$METRIC_BASELINE GPU=$GPU_ID PHASE_B=50"

while gpu_busy; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null || echo "?")
  log "GPU busy (util=${util}%) — sleeping ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

log "GPU idle. Launching MV phaseb50..."
export PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
export PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
export CHECKPOINT_TAG="${CHECKPOINT_TAG:-phaseb50}"

env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$METRIC_BASELINE" \
  bash "$ROOT/two_phase_run_multiview_v3_book_crossing_stratified_7b_phaseb50.sh" \
  >> "logs/exp_bc_mv_phaseb50_${RUN_ID}.log" 2>&1

log "======== book-crossing MV phaseb50 complete ========"
