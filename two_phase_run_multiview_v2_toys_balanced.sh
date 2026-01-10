#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Balanced Training - Toys (7B)
# ============================================================
# Balanced 配置设计原则:
# - alignment_weight: 0.07 (温和对齐，不过度压制高频)
# - temperature: 0.05 (Grid search)
# - text_weight: 0.8 (适中，不压制高频)
# - cold_start_boost: 2.0 (温和加权)
# - cross_dropout_prob: 0.15 (稳定正则化)
# ============================================================

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Multi-View V2 Balanced (7B) - Toys"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo "LLM Model: Qwen2.5-7B-Instruct"
echo ""
echo "Balanced Configuration:"
echo "  - alignment_weight: 0.07"
echo "  - temperature: grid search [0.03, 0.05, 0.07]"
echo "  - text_weight: 0.8"
echo "  - cold_start_align_boost: 2.0"
echo "  - cross_dropout_prob: 0.15"
echo "  - multiview_align_scale: 1.0"
echo ""
echo "Inference text weight formula:"
echo "  effective = alpha × text_weight × (1 + align_weight) × (0.07 / temp)"
echo "            ≈ 0.67 × 0.8 × 1.07 × 1.4 ≈ 0.80"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_balanced.yaml" \
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
  --variant_features "sasrec,multiview_v2,7b,balanced,cold_boost_2.0,toys" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Multi-View V2 Balanced 7B (Toys) Done!"
echo "========================================="
echo "Model Configuration:"
echo "  - Per-view L2 normalization"
echo "  - alignment_weight=0.07 (affects training + inference)"
echo "  - text_weight=0.8"
echo "  - cold_start_align_boost=2.0"
echo "  - cross_dropout_prob=0.15"
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - MRR_new@10, MRR_few@10, MRR_frequent@10"
echo ""
