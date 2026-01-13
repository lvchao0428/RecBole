#!/usr/bin/env bash
#
# exp_sensitivity_tau_003.sh - Sensitivity Analysis: temperature = 0.03
#
# 目的: 验证锐利对比学习对冷启动的影响
# 基准: λ=0.10, τ=0.05, cold=2.0, infer=1.0
# 变化: τ=0.05 → 0.03 (更锐利)
# 预期: HR_new↑ (冷启动物品受益), MRR/NDCG 稳定
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Sensitivity Analysis: temperature = 0.03"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - alignment_weight: 0.10 (baseline)"
echo "  - temperature: 0.05 → 0.03 (锐利对比学习)"
echo "  - cold_start_align_boost: 2.0 (baseline)"
echo "  - inference_cold_text_boost: 1.0 (baseline)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.03" \
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
  --config_dict "{'cold_start_align_boost': 2.0, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.0}" \
  --checkpoint_dir ./saved/sensitivity_tau_003 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,tau_003" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/sensitivity_tau_003/"
