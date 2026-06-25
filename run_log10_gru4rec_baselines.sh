#!/usr/bin/env bash
# log10 (1080Ti 11GB): GRU4Rec ID-only + TF-IDF for Beauty/Toys.
# Text-heavy LLM/MV stay on 5090.
#
# Usage (on log10):
#   SEED=2024 GPU_ID=0 bash run_log10_gru4rec_baselines.sh
# All seeds:
#   bash run_log10_gru4rec_baselines.sh --all-seeds

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
METRIC_BASELINE="${METRIC_BASELINE:-0.0272}"
# 11GB VRAM: smaller batch than 5090 defaults
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
export GPU_ID SEED METRIC_BASELINE

LOG="${LOG10_GRU4REC_LOG:-logs/log10_gru4rec_baselines.log}"
mkdir -p logs saved run_metrics

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_one() {
  local dataset="$1"  # beauty | toys
  local seed="$2"
  log "--- log10 GRU4Rec ID+TF-IDF ${dataset} seed=${seed} batch=${TRAIN_BATCH_SIZE} ---"
  t0=$SECONDS
  if [[ "$dataset" == "beauty" ]]; then
    env GPU_ID="$GPU_ID" SEED="$seed" METRIC_BASELINE="$METRIC_BASELINE" \
      bash "$ROOT/run50epBase_gru4rec_v3_beauty_stratified.sh" \
      >> "logs/log10_gru4rec_beauty_id_seed${seed}.log" 2>&1
    env GPU_ID="$GPU_ID" SEED="$seed" METRIC_BASELINE="$METRIC_BASELINE" \
      bash "$ROOT/two_phase_run_gru4rec_tfidf_v3_beauty_stratified.sh" \
      >> "logs/log10_gru4rec_beauty_tfidf_seed${seed}.log" 2>&1
  else
    env GPU_ID="$GPU_ID" SEED="$seed" METRIC_BASELINE="$METRIC_BASELINE" \
      bash "$ROOT/run50epBase_gru4rec_v3_toys_stratified.sh" \
      >> "logs/log10_gru4rec_toys_id_seed${seed}.log" 2>&1
    env GPU_ID="$GPU_ID" SEED="$seed" METRIC_BASELINE="$METRIC_BASELINE" \
      bash "$ROOT/two_phase_run_gru4rec_tfidf_v3_toys_stratified.sh" \
      >> "logs/log10_gru4rec_toys_tfidf_seed${seed}.log" 2>&1
  fi
  log "  ✅ ${dataset} seed=${seed} done ($((SECONDS - t0))s)"
}

log "======== log10 GRU4Rec baselines (ID + TF-IDF) ========"
log "GPU=$GPU_ID"

if [[ "${1:-}" == "--all-seeds" ]]; then
  for s in 42 2024 2025 2026; do
    run_one beauty "$s"
    run_one toys "$s"
  done
else
  run_one beauty "$SEED"
  run_one toys "$SEED"
fi

log "======== log10 GRU4Rec baselines complete ========"
