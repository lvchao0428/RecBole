#!/usr/bin/env bash
# GTS Multi-seed Toys + Grocery — Text models on 5090
# 3 text configs × 3 seeds × 2 datasets = 18 runs
#
# Usage (on 5090):
#   nohup bash run_5090_ts_toys_grocery.sh \
#     > logs/ts_toys_grocery_5090_nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
LOG="logs/ts_toys_grocery_5090.log"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

SEEDS=(2024 2025 42)

run_text_model() {
  local DS="$1" SEED="$2" MODEL="$3" CONFIG="$4" TAG="$5" RUN="$6" TOTAL="$7"
  local EXTRA_DICT="$8"
  log ">>> [$RUN/$TOTAL] ${TAG} (${DS}, seed=$SEED)"
  local t0=$SECONDS
  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset "$DS" \
    --config_files "$CONFIG" \
    --config_dict "$EXTRA_DICT" \
    --gpu_id "$GPU_ID" \
    --phase_a_grid \
    --align_grid "0.10" \
    --tau_grid "0.05" \
    --backbone_burnin_epochs 0 \
    --burnin_eval_step 2 \
    --phase_a_epochs ${PHASE_A_EPOCHS} \
    --phase_a_eval_step 1 \
    --phase_a_valid_metric "MRR@10" \
    --metric_baseline 0.0 \
    --metric_gain_threshold 0.01 \
    --lr_text_head 2e-3 \
    --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b \
    --phase_b_epochs ${PHASE_B_EPOCHS} \
    --backbone_lr_scale 0.1 \
    --checkpoint_dir "./saved/ts_${DS,,}_${TAG}_seed${SEED}" \
    --seed "$SEED" \
    --variant_features "sasrec,${TAG},ts,seed${SEED}" \
    --watchdog_disable \
    --save \
    >> "logs/ts_${DS,,}_${TAG}_seed${SEED}.log" 2>&1
  log "  done ($((SECONDS - t0))s)"
}

log "======== GTS Toys + Grocery: Text models on 5090 ========"
log "Seeds: ${SEEDS[*]}"

RUN=0
TOTAL=18

# --- Toys ---
for SEED in "${SEEDS[@]}"; do
  RUN=$((RUN + 1))
  run_text_model "Amazon_Toys_and_Games" "$SEED" "SASRecAlignV3" \
    "sasrec_align_toys_base_stratified_v3_ts.yaml" "tfidf" \
    "$RUN" "$TOTAL" \
    "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}"

  RUN=$((RUN + 1))
  run_text_model "Amazon_Toys_and_Games" "$SEED" "SASRecAlignV3" \
    "sasrec_align_toys_qwen3_stratified_v3_ts.yaml" "tfidf_llm" \
    "$RUN" "$TOTAL" \
    "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}"

  RUN=$((RUN + 1))
  run_text_model "Amazon_Toys_and_Games" "$SEED" "SASRecAlignMultiViewV3" \
    "sasrec_align_multi_view_v3_toys_stratified_7b_ts.yaml" "mv_no_se" \
    "$RUN" "$TOTAL" \
    "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'use_text_view_senet': false}"
done

# --- Grocery ---
for SEED in "${SEEDS[@]}"; do
  RUN=$((RUN + 1))
  run_text_model "Amazon_Grocery_and_Gourmet_Food" "$SEED" "SASRecAlignV3" \
    "sasrec_align_grocery_base_stratified_v3_ts.yaml" "tfidf" \
    "$RUN" "$TOTAL" \
    "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}"

  RUN=$((RUN + 1))
  run_text_model "Amazon_Grocery_and_Gourmet_Food" "$SEED" "SASRecAlignV3" \
    "sasrec_align_grocery_qwen_stratified_v3_ts.yaml" "tfidf_llm" \
    "$RUN" "$TOTAL" \
    "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}"

  RUN=$((RUN + 1))
  run_text_model "Amazon_Grocery_and_Gourmet_Food" "$SEED" "SASRecAlignMultiViewV3" \
    "sasrec_align_multi_view_v3_grocery_stratified_7b_ts.yaml" "mv_no_se" \
    "$RUN" "$TOTAL" \
    "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10, 'use_text_view_senet': false}"
done

log "======== GTS Toys+Grocery complete ($RUN/$TOTAL runs) ========"
