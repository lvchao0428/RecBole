#!/usr/bin/env bash
# log10 紧急切换：当前 Toys ID 2024 跑完后，若旧 bash 即将进入 2025，用本脚本替代。
# 仅补 Toys TF-IDF 2024（Beauty 2024 已完成），然后退出。
#
# Usage (on log10, after killing old run_log10_gru4rec_baselines_from_2024 loop):
#   bash run_log10_gru4rec_baselines_finish_2024_only.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
METRIC_BASELINE="${METRIC_BASELINE:-0.0272}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

LOG="${LOG10_GRU4REC_LOG:-logs/log10_gru4rec_baselines.log}"
mkdir -p logs saved run_metrics

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== log10 finish Toys TF-IDF seed=2024 only ========"
t0=$SECONDS
env MACHINE=log10 DATASET=toys SEED=2024 GPU_ID="$GPU_ID" METRIC_BASELINE="$METRIC_BASELINE" \
  SKIP_ID=1 bash "$ROOT/scripts/gru4rec_baseline_run_one.sh" | tee -a "$LOG"
log "  ✅ toys TF-IDF seed=2024 done ($((SECONDS - t0))s)"
touch logs/log10_gru4rec_through_2024.done
log "======== log10 through 2024 complete (5090 owns 2025/2026) ========"
