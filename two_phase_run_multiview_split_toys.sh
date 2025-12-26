#!/usr/bin/env bash
#set -euo pipefail

# Multi-View Split Training with Per-View Alignment - Toys_and_Games Dataset
#
# Workflow:
#   1. Run tools/gen_multiview_4views_toys.sh to generate 4-view Qwen3 embeddings
#   2. Launch two-phase training with SASRecAlignMultiView
#
# Key Features:
#   - 4 views from different prompts (identity, function, audience, category)
#   - Per-view SENet enhancement
#   - Per-view alignment loss with learnable weights
#   - Gated fusion via concat → projection → cross network
#   - Based on successful Beauty dataset experiments (Recall@10: +12.28%, MRR@10: +50.24%)

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[Training] Launching two-phase SASRecAlignMultiView for Toys_and_Games..."
echo "  - Model: SASRecAlignMultiView (4 views)"
echo "  - Per-view alignment: Enabled (learnable weights)"
echo "  - Fusion: SENet → Gate → Concat → Projection → Cross Network"
echo "  - Based on Beauty dataset success: +12.28% Recall, +50.24% MRR"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_toys.yaml" \
  --phase_a_grid \
  --align_grid "0.03,0.05" \
  --tau_grid "0.05,0.07" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 15 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "mrr@10" \
  --metric_baseline 0.0272 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.05 \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_4views_toys \
  --seed 2025 \
  --variant_features "sasrec,multiview,4views,per_view_align,qwen3" \
  --watchdog_disable \
  --save

echo ""
echo "[Training] Done! Check saved/phase_runs_multiview_4views_toys/ for checkpoints."

