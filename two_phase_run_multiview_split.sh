#!/usr/bin/env bash
#set -euo pipefail

# Multi-View Split Training with Per-View Alignment
#
# Workflow:
#   1. Run tools/gen_multiview_4views_beauty.sh to generate 4-view Qwen3 embeddings
#   2. Launch two-phase training with SASRecAlignMultiView
#
# Key Features:
#   - 4 views from different prompts (identity, function, audience, category)
#   - Per-view SENet enhancement
#   - Per-view alignment loss with learnable weights
#   - Gated fusion via concat → projection → cross network
#   - Architecture aligned with two_phase_run_tfidf_llm.sh

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[Training] Launching two-phase SASRecAlignMultiView with per-view alignment..."
echo "  - Model: SASRecAlignMultiView (4 views)"
echo "  - Per-view alignment: Enabled (learnable weights)"
echo "  - Fusion: SENet → Gate → Concat → Projection → Cross Network"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view.yaml" \
  --phase_a_grid \
  --align_grid "0.01,0.03,0.05" \
  --tau_grid "0.05,0.07,0.1" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 6 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "Recall@10" \
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
  --checkpoint_dir ./saved/phase_runs_multiview_4views \
  --seed 2025 \
  --variant_features "sasrec,multiview,4views,per_view_align,qwen3" \
  --watchdog_disable \
  --save

echo ""
echo "[Training] Done! Check saved/phase_runs_multiview_4views/ for checkpoints."

