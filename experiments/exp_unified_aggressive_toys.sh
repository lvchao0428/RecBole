#!/usr/bin/env bash
#
# exp_unified_aggressive_toys.sh - Toys 使用 Beauty 的 aggressive 参数
#
# 目的：验证统一参数的可行性
# 配置：cold=2.5, infer=1.5 (来自 Beauty 最佳)
#
# 预期：
# - 如果成功 (MRR > 0.0376)：参数具有通用性
# - 如果失败：需要数据集特定的参数调优

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Unified Aggressive (Toys)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.5 (aggressive)"
echo "  - inference_cold_text_boost: 1.5 (aggressive)"
echo "  - Source: Beauty's best config"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_unified_aggressive.yaml" \
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
  --checkpoint_dir ./saved/exp_unified_aggressive_toys \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,unified_aggressive" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_unified_aggressive_toys/"
