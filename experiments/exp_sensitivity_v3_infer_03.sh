#!/usr/bin/env bash
#
# exp_sensitivity_v3_infer_03.sh - Sensitivity Analysis: infer_boost = 0.3
#
# 基准: align_weight=0.10, τ=0.05, cold_text_boost=3.0, infer_boost=0.6
# 变化: infer_boost=0.6 → 0.3 (弱推理增强)
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Sensitivity Analysis: infer_boost = 0.3"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - align_weight: 0.10 (baseline)"
echo "  - temperature: 0.05 (baseline)"
echo "  - cold_text_boost: 3.0 (baseline)"
echo "  - infer_boost: 0.6 → 0.3 (弱推理增强)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v3_toys_stratified_7b.yaml" \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.3, 'cold_threshold': 10}" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0272 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/sensitivity_v3_infer_03 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v3,7b,toys,infer_03" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/sensitivity_v3_infer_03/"
