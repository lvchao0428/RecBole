#!/usr/bin/env bash
# 5090 主实验流水线（2026-06-25 起）
#
# 优先级队列：
#   1. UniSRec Beauty 三配置（seed=2024，主表）
#   2. Grocery SASRec multiseed（2025/2026/42 × 四配置，2024已完成）
#
# Usage:
#   nohup bash run_5090_main_pipeline_20260625.sh \
#     > logs/main_pipeline_20260625_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
LOG="logs/main_pipeline_20260625.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "======== Main Pipeline 20260625 ========"
log "GPU=$GPU_ID"

# ============================================================
# Phase 1: UniSRec Beauty 三配置 (最高优先级)
# ============================================================
log ">>> Phase 1: UniSRec Beauty (3 configs, seed=2024)"

# 1a. UniSRec Base (ID + MoE, no cross/align)
log "  [1/3] UniSRec Base"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 \
  bash "$ROOT/run_unisrec_beauty_stratified.sh" \
  >> "logs/unisrec_base_beauty_seed2024.log" 2>&1
log "  ✅ UniSRec Base done ($((SECONDS - t0))s)"

# 1b. UniSRecAlignV3 (+ Cross + Align)
log "  [2/3] UniSRecAlignV3 (+ Cross + Align)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 \
  bash "$ROOT/run_unisrec_align_v3_beauty_stratified.sh" \
  >> "logs/unisrec_align_v3_beauty_seed2024.log" 2>&1
log "  ✅ UniSRecAlignV3 done ($((SECONDS - t0))s)"

# 1c. UniSRecAlignMultiViewV3 (+ Cross + Align + MV)
log "  [3/3] UniSRecAlignMultiViewV3 (+ Cross + Align + MV)"
t0=$SECONDS
env GPU_ID="$GPU_ID" SEED=2024 \
  bash "$ROOT/run_unisrec_align_multiview_v3_beauty_stratified.sh" \
  >> "logs/unisrec_align_multiview_v3_beauty_seed2024.log" 2>&1
log "  ✅ UniSRecAlignMultiViewV3 done ($((SECONDS - t0))s)"

log ">>> Phase 1 complete: UniSRec Beauty 三配置"

# ============================================================
# Phase 2: Grocery SASRec Multiseed
# ============================================================
log ">>> Phase 2: Grocery SASRec Multiseed (3 seeds × 4 configs)"

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
    SCRIPT="${GROCERY_SCRIPTS[$i]}"
    NAME="${GROCERY_NAMES[$i]}"
    log "  ▶ $NAME (seed=$SEED)"
    t0=$SECONDS
    env GPU_ID="$GPU_ID" SEED="$SEED" METRIC_BASELINE=0.0204 \
      bash "$ROOT/$SCRIPT" >> "logs/grocery_${NAME// /_}_seed${SEED}.log" 2>&1
    log "  ✅ $NAME seed=$SEED done ($((SECONDS - t0))s)"
  done
done

log ">>> Phase 2 complete: Grocery multiseed"
log "======== Main Pipeline 20260625 complete ========"
