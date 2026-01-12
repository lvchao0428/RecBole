#!/usr/bin/env bash
#
# exp_multiview_toys_32b_aggressive.sh - Toys Multi-view 32B + Aggressive
#
# 目的：验证 Toys 32B aggressive 能否展现 Scale Law
# 当前：Toys HR Scale Law 成立，但 MRR 32B = 7B
# 目标：验证 aggressive 配置能否让 32B 在 MRR 上也超越 7B/14B
#
# 理论依据：
# - Toys HR@10 已展现 Scale Law: 7B < 14B < 32B ✅
# - HR_new Scale Law 更明显: 7B (+5%) < 14B (+6%) < 32B (+9%) ✅
# - aggressive 配置可能让 32B 的语义优势在 MRR 上也体现

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Toys Multi-view 32B Aggressive"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.5"
echo "  - inference_cold_text_boost: 1.5"
echo "  - Model: 32B"
echo "  - Goal: 验证 Scale Law (7B < 14B < 32B)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_32b.yaml" \
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
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5}" \
  --checkpoint_dir ./saved/exp_multiview_toys_32b_aggressive \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,32b,toys,aggressive" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_multiview_toys_32b_aggressive/"
