#!/usr/bin/env bash
#
# exp_multiview_toys_14b_aggressive.sh - Toys Multi-view 14B + Aggressive
#
# 目的：验证 Toys 14B Scale Law 在 aggressive 配置下是否更明显
# 当前：7B (0.0368) < 14B (0.0373) > 32B (0.0368) ⚠️ 14B 峰值
# 目标：14B aggressive 能否拉开与 7B 的差距
#
# 理论依据：
# - Toys standard 配置下 14B 已显示峰值优势
# - aggressive 配置可能让 14B 的语义优势更明显
# - 与实验1 (7B aggressive) 对比，验证 Scale Law

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Toys Multi-view 14B Aggressive"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.5"
echo "  - inference_cold_text_boost: 1.5"
echo "  - Model: 14B"
echo "  - Goal: MRR@10 > 7B aggressive"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_14b.yaml" \
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
  --checkpoint_dir ./saved/exp_multiview_toys_14b_aggressive \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,14b,toys,aggressive" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_multiview_toys_14b_aggressive/"
