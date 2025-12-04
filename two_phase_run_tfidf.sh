#!/usr/bin/env bash
#set -euo pipefail

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Two-phase strong baseline: SASRec_Align + TF-IDF (base only) on Amazon_Beauty
# Phase-A: freeze backbone, grid over (alignment_weight, temperature) with gating
# Phase-B: unfreeze backbone, grouped LR with lr_backbone = lr_text_head * 0.1

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base.yaml" \
  --phase_a_grid \
  --align_grid "0.01,0.03,0.05" \
  --tau_grid "0.05,0.07,0.1" \
  --phase_a_epochs 6 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric Recall@10 \
  --ndcg_baseline 0.0562 \
  --ndcg_gain_threshold 0.002 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.05 \
  --phase_b_alignment_weight 0.05 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs \
  --seed 2025 \
  --variant_features "sasrec,tfidf,beauty" \
  --watchdog_interval 20 \
  --watchdog_log "run_metrics/watchdog_tfidf_beauty.log" \
  --watchdog_cpu_gb 40 \
  --watchdog_gpu_gb 28 \
  --save

