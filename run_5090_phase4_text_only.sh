#!/usr/bin/env bash
# Phase 4 5090-only: LLM+MV text runs, then wait/pull log10 baselines.
# Use when log10 ID+TF-IDF is already running.
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
LOG="${PHASE4_LOG:-logs/post_main_pipeline_20260625.log}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Phase 4 RESUME (5090 text only) ========"
log "log10 baselines assumed running — skipping remote_start"

log ">>> 5090: LLM + MV (16 runs)"
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

log "======== Phase 4 complete ========"
