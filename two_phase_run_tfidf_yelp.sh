#!/usr/bin/env bash
#set -euo pipefail

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset yelp \
  --config_files "sasrec_align_yelp_base.yaml" \
  --phase_a_grid \
  --align_grid "0.05" \
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
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs \
  --seed 2025 \
  --save

