#!/usr/bin/env bash
cd /home/charlie/project/RecBole
export PYTHONPATH=$(pwd):$PYTHONPATH

# Phase-B Training Script
# Using checkpoint from Phase-A: ./saved/phase_runs/SASRec_Align-Nov-19-2025_23-39-22.pth
# Configuration matches guidance:
# - Phase-B epochs: 40
# - Learning rates: backbone = text_head * 0.1
# - Dropout & Gate: controlled via config (cross_dropout_prob, text_gate_reg_l2)
#   Note: Adjust cross_dropout_prob and text_gate_reg_l2 in overrides if needed, 
#   defaulted here via command line args or existing config.

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3.yaml overrides/stability_enhance.yaml" \
  --only_phase_b \
  --phase_b_epochs 40 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --backbone_lr_scale 0.1 \
  --resume_from ./saved/phase_runs/SASRec_Align-Nov-19-2025_23-39-22.pth \
  --save \
  --seed 2025

