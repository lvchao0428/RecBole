#!/bin/bash

# Two-phase training for whitened TF-IDF features
# Key changes:
# 1. Grid search over larger temperature values
# 2. Enable Phase-A early stopping with baseline gate
# 3. Shorter Phase-A epochs to avoid overfitting

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base_whiten.yaml" \
  \
  --phase_a_epochs 5 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  \
  --phase_b_epochs 40 \
  \
  --lr_text_head 0.001 \
  --lr_dnn_cross 0.0005 \
  --backbone_lr_scale 0.1 \
  \
  --phase_a_grid \
  --align_grid "0.03,0.05,0.08" \
  --tau_grid "0.1,0.15,0.2" \
  \
  --ndcg_baseline 0.0285 \
  --ndcg_gain_threshold 0.02 \
  --phase_a_auto_to_b \
  \
  --save \
  --checkpoint_dir "./saved/whiten_test"

