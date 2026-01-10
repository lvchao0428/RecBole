#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Balanced Training - Toys (14B)
# 用于验证 Scale Law

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Multi-View V2 Balanced (14B) - Toys"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo "LLM Model: Qwen2.5-14B-Instruct"
echo ""
echo "Balanced Configuration (same as 7B):"
echo "  - alignment_weight: 0.07"
echo "  - temperature: grid search [0.03, 0.05, 0.07]"
echo "  - text_weight: 0.8"
echo "  - cold_start_align_boost: 2.0"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_balanced_14b.yaml" \
  --phase_a_grid \
  --align_grid "0.07" \
  --tau_grid "0.03,0.05,0.07" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0272 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.07 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 0.8 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_v2_toys_balanced \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,14b,balanced,cold_boost_2.0,toys" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Multi-View V2 Balanced 14B (Toys) Done!"
echo "========================================="
