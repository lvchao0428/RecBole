#!/usr/bin/env bash
# Resume from UniSRecAlignV3 (Exp 1 UniSRec Base already done)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_ID="${GPU_ID:-0}"
LOG="logs/main_pipeline_20260625.log"
mkdir -p logs
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log ">>> Resume: UniSRecAlignV3 [2/3]"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 bash "$ROOT/run_unisrec_align_v3_beauty_stratified.sh" \
  >> "logs/unisrec_align_v3_beauty_seed2024.log" 2>&1
log "  ✅ UniSRecAlignV3 done ($((SECONDS - t0))s)"

log ">>> UniSRecAlignMultiViewV3 [3/3]"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 bash "$ROOT/run_unisrec_align_multiview_v3_beauty_stratified.sh" \
  >> "logs/unisrec_align_multiview_v3_beauty_seed2024.log" 2>&1
log "  ✅ UniSRecAlignMultiViewV3 done ($((SECONDS - t0))s)"

log ">>> Phase 2: Grocery SASRec Multiseed"
GROCERY_SEEDS=(2025 2026 42)
GROCERY_SCRIPTS=(
  "run50epBase_grocery_stratified.sh"
  "two_phase_run_tfidf_v3_grocery_stratified.sh"
  "two_phase_run_tfidf_llm_v3_grocery_stratified.sh"
  "two_phase_run_multiview_v3_grocery_stratified_7b.sh"
)
GROCERY_NAMES=("ID-only" "TF-IDF" "TF-IDF+LLM" "MV")
for SEED in "${GROCERY_SEEDS[@]}"; do
  log "--- Grocery seed=$SEED ---"
  for i in "${!GROCERY_SCRIPTS[@]}"; do
    log "  ▶ ${GROCERY_NAMES[$i]} (seed=$SEED)"
    t0=$SECONDS
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE=0.0204 \
      bash "$ROOT/${GROCERY_SCRIPTS[$i]}" >> "logs/grocery_${GROCERY_NAMES[$i]// /_}_seed${SEED}.log" 2>&1
    log "  ✅ ${GROCERY_NAMES[$i]} seed=$SEED done ($((SECONDS - t0))s)"
  done
done
log "======== Pipeline resume complete ========"

log ">>> Chaining Phase 3: mechanism + supplemental analysis"
bash "$ROOT/run_5090_post_main_pipeline_20260625.sh"
