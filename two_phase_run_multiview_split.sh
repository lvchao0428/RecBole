#!/usr/bin/env bash
#set -euo pipefail

# Workflow:
#   1. Run tools/get_multiview_qwen3_embed.sh to regenerate per-view embeddings.
#   2. Launch two-phase training that consumes split views via SASRecAlignMultiView.

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "[Training] Launching two-phase SASRecAlignMultiView run (per-view residual cross)..."
python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view.yaml" \
  --phase_a_grid \
  --align_grid "0.05,0.1,0.2" \
  --tau_grid "0.05,0.07" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 2 \
  --phase_a_valid_metric NDCG@10 \
  --ndcg_baseline 0.0272 \
  --ndcg_gain_threshold 0.01 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.05 \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.02 \
  --phase_b_text_weight 0.6 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs \
  --seed 2025 \
  --save

