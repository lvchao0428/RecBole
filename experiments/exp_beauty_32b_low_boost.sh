#!/usr/bin/env bash
#
# exp_beauty_32b_low_boost.sh - Beauty 32B Low Boost + High Align
#
# 目的：验证 Beauty 32B 在"降低放大 + 增强对齐"策略下能否展现 Scale Law
#
# 参数变化：
# - cold_start_align_boost: 2.5 → 1.5 (降低)
# - inference_cold_text_boost: 1.5 → 0.5 (大幅降低)
# - alignment_weight: 0.10 → 0.15 (提高)
#
# 预期：
# - 如果 Hit_new@10 > 14B low_boost：Scale Law 成立
# - 如果 > 7B aggressive (0.0181)：证明策略有效
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Beauty 32B Low Boost + High Align"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 1.5 (降低)"
echo "  - inference_cold_text_boost: 0.5 (降低)"
echo "  - alignment_weight: 0.15 (提高)"
echo "  - Model: 32B"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_beauty_32b_low_boost.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.15" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0207 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.15 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --config_dict "{'cold_start_align_boost': 1.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 0.5}" \
  --checkpoint_dir ./saved/exp_beauty_32b_low_boost \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,32b,beauty,low_boost" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_beauty_32b_low_boost/"
