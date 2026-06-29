#!/usr/bin/env bash
# Resume Phase 4 text from Toys seed=42 (Beauty seed=42 already done 2026-06-26).
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
SKIP_BEAUTY_SEEDS="${SKIP_BEAUTY_SEEDS:-42}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== Phase 4 RESUME (from Toys; Beauty ${SKIP_BEAUTY_SEEDS} done) ========"

for SEED in "${SEEDS[@]}"; do
  if [[ "$SEED" != "$SKIP_BEAUTY_SEEDS" ]]; then
    log "--- Beauty seed=$SEED (LLM+MV) ---"
    t0=$SECONDS
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE="$BEAUTY_METRIC_BASELINE" \
      bash "$ROOT/run_gru4rec_v3_batch_beauty_text.sh" \
      >> "logs/gru4rec_beauty_text_seed${SEED}.log" 2>&1
    log "  ✅ Beauty text seed=$SEED ($((SECONDS - t0))s)"
  else
    log "--- Beauty seed=$SEED skipped (already done) ---"
  fi

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

# --- Phase 5: UniSRec +Align / +MV (main-table balanced weights) ---
log ">>> Phase 5: UniSRec Align/MV balanced (Beauty seed=2024)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 \
  bash "$ROOT/run_unisrec_v3_align_balanced_batch_beauty.sh" \
  >> "logs/unisrec_align_balanced_batch_beauty_nohup.log" 2>&1
log "  ✅ Phase 5 UniSRec balanced done ($((SECONDS - t0))s)"

log "======== Phase 4 + Phase 5 complete ========"
