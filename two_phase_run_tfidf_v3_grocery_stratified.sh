#!/usr/bin/env bash
# Two-phase TF-IDF V3 + stratified — Grocery

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-phaseb50}"
DS="Amazon_Grocery_and_Gourmet_Food"

python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset "$DS" \
  --config_files "sasrec_align_grocery_base_stratified_v3.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
  --gpu_id "$GPU_ID" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs ${PHASE_A_EPOCHS} \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline "$METRIC_BASELINE" \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs ${PHASE_B_EPOCHS} \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir "./saved/two_phase_run_tfidf_v3_grocery_stratified_${CHECKPOINT_TAG}_seed${SEED}" \
  --seed "$SEED" \
  --variant_features "sasrec,tfidf,v3,grocery,stratified,${CHECKPOINT_TAG},seed_${SEED}" \
  --watchdog_disable \
  --save
