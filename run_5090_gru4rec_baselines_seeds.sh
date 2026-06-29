#!/usr/bin/env bash
# 5090: GRU4Rec ID+TF-IDF for given seeds (offload from log10).
#
# Usage:
#   bash run_5090_gru4rec_baselines_seeds.sh 2026
#   bash run_5090_gru4rec_baselines_seeds.sh 2025 2026
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
METRIC_BASELINE="${METRIC_BASELINE:-0.0272}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(2026)
fi

LOG="${GRU4REC_5090_BASELINE_LOG:-logs/5090_gru4rec_baselines.log}"
mkdir -p logs saved run_metrics paper_recsys

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== 5090 GRU4Rec baselines seeds=${SEEDS[*]} ========"
printf '%s\n' "${SEEDS[@]}" > paper_recsys/.gru4rec_owner_5090_seeds

for SEED in "${SEEDS[@]}"; do
  for ds in beauty toys; do
    env MACHINE=5090 DATASET="$ds" SEED="$SEED" GPU_ID="$GPU_ID" \
      METRIC_BASELINE="$METRIC_BASELINE" \
      bash "$ROOT/scripts/gru4rec_baseline_run_one.sh" >> "$LOG" 2>&1
  done
done

log "======== 5090 GRU4Rec baselines complete (seeds=${SEEDS[*]}) ========"
