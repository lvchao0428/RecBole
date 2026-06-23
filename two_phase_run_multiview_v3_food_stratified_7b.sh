#!/usr/bin/env bash
# Two-phase Multi-View V3 (7B) + stratified — Food

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
METRIC_BASELINE="${METRIC_BASELINE:-0.015}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-20}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-50}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-phaseb50}"

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Food \
  --config_files "sasrec_align_multi_view_v3_food_stratified_7b.yaml" \
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
  --checkpoint_dir "./saved/two_phase_run_multiview_v3_food_stratified_7b_${CHECKPOINT_TAG}_seed${SEED}" \
  --seed "$SEED" \
  --variant_features "sasrec,multiview_v3,7b,food,stratified,${CHECKPOINT_TAG},seed_${SEED}" \
  --watchdog_disable \
  --save
