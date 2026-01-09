#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Training - Toys AGGRESSIVE Squeeze
# ============================================================
# 目标: 最大化 text 特征学习，验证 Toys 的性能上限
# 方案: 更强对齐 + 更紧 InfoNCE + 更长训练
#
# 与普通 squeeze 的区别:
# - alignment_weight: 0.15 (从 0.10 提升)
# - temperature: 0.03 (从 0.05 降低)
# - multiview_align_scale: 2.0 (开启对齐放大)
# - phase_b_epochs: 50 (从 40 增加)
# ============================================================

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Multi-View V2 (7B) Toys AGGRESSIVE Squeeze"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo "LLM Model: Qwen2.5-7B-Instruct"
echo ""
echo "AGGRESSIVE Settings:"
echo "  - alignment_weight: 0.15 (stronger)"
echo "  - temperature: 0.03 (tighter InfoNCE)"
echo "  - multiview_align_scale: 2.0 (alignment amplification)"
echo "  - text_weight: 0.6"
echo "  - cross_dropout_prob: 0.1 (more stable)"
echo "  - phase_b_epochs: 50 (longer training)"
echo ""
echo "Goal: Find Toys performance upper bound"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b_squeeze_aggressive.yaml" \
  --phase_a_grid \
  --align_grid "0.15" \
  --tau_grid "0.03" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 25 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0272 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.15 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 0.6 \
  --phase_a_auto_to_b \
  --phase_b_epochs 50 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_v2_toys_aggressive \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,4views,toys,aggressive,strong_align,tight_tau" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Toys Aggressive Squeeze Done!"
echo "========================================="
echo "This is the upper bound test."
echo "Compare with regular squeeze to see if stronger config helps."
echo ""
