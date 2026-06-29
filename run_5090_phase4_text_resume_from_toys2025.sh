#!/usr/bin/env bash
# Resume Phase 4 after power outage.
# Assumes Beauty GRU4Rec LLM+MV for seeds 42/2024/2025 are done (incl. MV 2025).
# Continues: Toys 2025 → Beauty/Toys 2026 → rebalanced tail (UniSRec + 5090 baseline 2026 + pull log10).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
BEAUTY_METRIC_BASELINE="${BEAUTY_METRIC_BASELINE:-0.0272}"
TOYS_METRIC_BASELINE="${TOYS_METRIC_BASELINE:-0.0272}"
LOG="${PHASE4_LOG:-logs/post_main_pipeline_20260625.log}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Phase 4 RESUME (from Toys seed=2025; power-outage checkpoint) ========"

log "--- Toys seed=2025 (LLM+MV) ---"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2025 METRIC_BASELINE="$TOYS_METRIC_BASELINE" \
  bash "$ROOT/run_gru4rec_v3_batch_toys_text.sh" \
  >> "logs/gru4rec_toys_text_seed2025.log" 2>&1
log "  ✅ Toys text seed=2025 ($((SECONDS - t0))s)"

for SEED in 2026; do
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

log "======== Phase 4 text (5090 LLM+MV) complete ========"
bash "$ROOT/run_5090_phase4_tail_rebalanced.sh" >> "$LOG" 2>&1

log "======== Phase 4 + Phase 5 complete (rebalanced) ========"
