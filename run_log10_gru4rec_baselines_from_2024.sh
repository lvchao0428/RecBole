#!/usr/bin/env bash
# log10: GRU4Rec ID+TF-IDF seed 2024→2026（可重复跑，5090 侧 skip 已有 ckpt）。
# 多跑无妨；5090 收尾脚本会覆盖/补齐全部 baseline。
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
export GPU_ID METRIC_BASELINE

LOG="${LOG10_GRU4REC_LOG:-logs/log10_gru4rec_baselines.log}"
mkdir -p logs saved run_metrics

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_one() {
  local dataset="$1"
  local seed="$2"
  env MACHINE=log10 DATASET="$dataset" SEED="$seed" GPU_ID="$GPU_ID" \
    METRIC_BASELINE="$METRIC_BASELINE" \
    bash "$ROOT/scripts/gru4rec_baseline_run_one.sh" | tee -a "$LOG"
}

log "======== log10 GRU4Rec RESUME (2024→2026; duplicates OK) ========"

for s in 2024 2025 2026; do
  run_one beauty "$s"
  run_one toys "$s"
done

log "======== log10 GRU4Rec complete (2024→2026) ========"
