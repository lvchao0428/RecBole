#!/usr/bin/env bash
cd /home/charlie/project/RecBole
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3.yaml overrides/stability_enhance.yaml" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --only_phase_a \
  --phase_a_grid \
  --align_grid "0.05,0.1,0.2" \
  --tau_grid "0.05,0.07" \
  --phase_a_epochs 20 \
  --phase_a_eval_step 2 \
  --phase_a_valid_metric NDCG@10 \
  --ndcg_baseline 0.0272 \
  --ndcg_gain_threshold 0.01 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --checkpoint_dir ./saved/phase_runs \
  --seed 2025 \
  --save
