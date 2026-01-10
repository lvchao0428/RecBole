#!/usr/bin/env bash
#set -euo pipefail

# Two-phase TF-IDF baseline with Balanced Configuration (Toys Dataset)
# 与 Multi-View V2 Balanced 公平对比
#
# Balanced 参数:
# - alignment_weight: 0.07
# - temperature: 0.05
# - text_weight: 0.8
# - cold_start_align_boost: 2.0

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "TF-IDF Balanced - Toys"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo ""
echo "Balanced Configuration:"
echo "  - alignment_weight: 0.07"
echo "  - temperature: 0.05"
echo "  - text_weight: 0.8"
echo "  - cold_start_align_boost: 2.0"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_toys_base_balanced.yaml" \
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
  --checkpoint_dir ./saved/phase_runs_toys_balanced \
  --seed 2025 \
  --variant_features "sasrec,tfidf,balanced,toys" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ TF-IDF Balanced (Toys) Done!"
echo "========================================="
