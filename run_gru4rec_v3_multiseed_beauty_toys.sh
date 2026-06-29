#!/usr/bin/env bash
# GRU4Rec split: log10 = ID+TF-IDF (parallel), 5090 = LLM+MV.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEEDS=(42 2024 2025 2026)
BEAUTY_METRIC_BASELINE="${BEAUTY_METRIC_BASELINE:-0.0272}"
TOYS_METRIC_BASELINE="${TOYS_METRIC_BASELINE:-0.0272}"
USE_LOG10="${USE_LOG10:-1}"

LOG="${GRU4REC_LOG:-logs/gru4rec_multiseed_beauty_toys.log}"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== GRU4Rec V3 multiseed (5090+log10 split) ========"
log "GPU=$GPU_ID  seeds=${SEEDS[*]}  USE_LOG10=$USE_LOG10"

if [[ "$USE_LOG10" == "1" ]]; then
  log ">>> Launch log10 baselines (ID + TF-IDF × 4 seeds × 2 datasets)"
  bash "$ROOT/scripts/remote_start_log10_gru4rec_baselines.sh" >> "$LOG" 2>&1 || {
    log "⚠️ log10 start failed — falling back to full 5090 run"
    USE_LOG10=0
  }
  sleep 2
fi

if [[ "$USE_LOG10" == "1" ]]; then
  log ">>> 5090: LLM + MV only (16 runs)"
  for SEED in "${SEEDS[@]}"; do
    log "--- Beauty seed=$SEED (LLM+MV) ---"
    t0=$SECONDS
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$BEAUTY_METRIC_BASELINE" \
      bash "$ROOT/run_gru4rec_v3_batch_beauty_text.sh" \
      >> "logs/gru4rec_beauty_text_seed${SEED}.log" 2>&1
    log "  ✅ Beauty text seed=$SEED ($((SECONDS - t0))s)"

    log "--- Toys seed=$SEED (LLM+MV) ---"
    t0=$SECONDS
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$TOYS_METRIC_BASELINE" \
      bash "$ROOT/run_gru4rec_v3_batch_toys_text.sh" \
      >> "logs/gru4rec_toys_text_seed${SEED}.log" 2>&1
    log "  ✅ Toys text seed=$SEED ($((SECONDS - t0))s)"
  done

  log ">>> Wait log10 + pull baseline results"
  bash "$ROOT/scripts/wait_and_pull_log10_gru4rec.sh" >> "$LOG" 2>&1
else
  log ">>> Full 5090 run (all 4 configs × seeds)"
  for SEED in "${SEEDS[@]}"; do
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$BEAUTY_METRIC_BASELINE" \
      bash "$ROOT/run_gru4rec_v3_batch_beauty.sh" >> "logs/gru4rec_beauty_seed${SEED}.log" 2>&1
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$TOYS_METRIC_BASELINE" \
      bash "$ROOT/run_gru4rec_v3_batch_toys.sh" >> "logs/gru4rec_toys_seed${SEED}.log" 2>&1
  done
fi

log "======== GRU4Rec multiseed split complete ========"
