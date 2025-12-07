#!/usr/bin/env bash
#set -euo pipefail

# Multi-View Concat Training (Updated)
#
# Workflow:
#   1. Features already generated: item_text_emb.qwen3.multiview.npy (256-dim, 4 views concat)
#   2. Launch two-phase training with SASRec_Align using concatenated multi-view embeddings
#
# Key Features:
#   - 4 views concatenated into single 256-dim vector
#   - Qwen3 multi-view embeddings (Identity, Function, Audience, Category)
#   - Aligned with two_phase_run_multiview_split.sh parameters
#   - Uses item_text_emb.qwen3.multiview.npy
#
# Feature Structure:
#   - Base (TF-IDF): dataset/Amazon_Beauty/item_text_emb.base.npy
#   - Multi-view (Qwen3 concat): dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
#   - Whiten stats: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[Training] Launching two-phase SASRec_Align with multi-view concat embeddings..."
echo "  - Model: SASRec_Align"
echo "  - Features: item_text_emb.qwen3.multiview.npy (4 views, 256-dim)"
echo "  - Fusion: Concat → Cross Network"
echo ""

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multiview_concat.yaml" \
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
  --checkpoint_dir ./saved/phase_runs_multiview_concat \
  --seed 2025 \
  --variant_features "sasrec,multiview,4views,concat,qwen3" \
  --watchdog_disable \
  --save

echo ""
echo "[Training] Done! Check saved/phase_runs_multiview_concat/ for checkpoints."


