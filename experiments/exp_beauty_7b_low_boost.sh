#!/usr/bin/env bash
#
# exp_beauty_7b_low_boost.sh - Beauty 7B Low Boost + High Align
#
# 目的：作为 Beauty Scale Law 对比基线
#       与 14B/32B Low Boost 实验一起验证 Scale Law
#
# 参数配置 (Low Boost + High Align):
# - cold_start_align_boost: 1.5 (降低)
# - inference_cold_text_boost: 0.5 (降低)
# - alignment_weight: 0.15 (提高)
#
# 预期：
# - 作为 Beauty Scale Law 基线
# - 如果 32B > 14B > 7B (Hit_new@10)：Scale Law 在 Beauty 成立
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Beauty 7B Low Boost + High Align"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 1.5 (降低)"
echo "  - inference_cold_text_boost: 0.5 (降低)"
echo "  - alignment_weight: 0.15 (提高)"
echo "  - Model: 7B"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_stratified.yaml" \
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
  --checkpoint_dir ./saved/exp_beauty_7b_low_boost \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,beauty,low_boost" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_beauty_7b_low_boost/"
