#!/usr/bin/env bash
#
# exp_boost_32b_aggressive.sh - Beauty 32B + Aggressive Inference Boost
#
# 目的：验证 Beauty 32B 能否通过 aggressive 参数超越 7B/14B
# 配置：cold=2.5, infer=1.5 (aggressive)
#
# 当前问题：Beauty 32B (0.0314) < 7B (0.0320)，Scale Law 反转
#
# 预期：
# - 如果 32B aggressive > 7B aggressive：语义过饱和，需要更强 boost
# - 如果仍然落后：Beauty 数据集本身是 Scale Law 的瓶颈

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Beauty 32B + Aggressive Boost"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.5"
echo "  - inference_cold_text_boost: 1.5"
echo "  - Model: 32B"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_inference_boost_32b_aggressive.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
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
  --phase_b_alignment_weight 0.10 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/exp_boost_32b_aggressive \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,32b,beauty,aggressive" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_boost_32b_aggressive/"
