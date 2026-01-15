#!/usr/bin/env bash
#
# exp_sensitivity_infer_10.sh - Sensitivity Analysis: inference_cold_text_boost = 1.0
#
# 目的: 补充敏感性分析的中间点
# 基准: λ=0.10, τ=0.05, cold=2.5, infer=1.5 (Aggressive)
# 变化: infer=1.5 → 1.0 (中等推理放大)
# 预期: HR 和 MRR 介于 infer=0.5 和 infer=1.5 之间
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Sensitivity Analysis: inference_cold_text_boost = 1.0"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - alignment_weight: 0.10 (baseline)"
echo "  - temperature: 0.05 (baseline)"
echo "  - cold_start_align_boost: 2.5 (aggressive baseline)"
echo "  - inference_cold_text_boost: 1.5 → 1.0 (中等推理放大)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0249 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.10 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.0}" \
  --checkpoint_dir ./saved/sensitivity_infer_10 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,agg,infer_10" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/sensitivity_infer_10/"
