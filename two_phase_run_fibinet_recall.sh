#!/usr/bin/env bash
#set -euo pipefail

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Two-phase strong baseline: SASRec_Align + Multi-View LLM Amplifier (FiBiNET/SENet)
#
# Focused on RECALL Improvement:
# 1. Low alignment_weight (0.01) to protect ID space
# 2. High text_gate_reg_l2 (0.1) to prevent semantic overfitting
# 3. High Cross Dropout (0.3) for robustness
#
# Prerequisites:
# 1. Run tools/build_item_text_emb_qwen3_hf.py with --prompt_preset multiview --output_mode concat
#    to generate the amplified embeddings (e.g., item_text_emb_amplified.npy).

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3.yaml overrides/stability_enhance_with_id_ln.yaml" \
  --item_text_emb_path_llm "/home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_amplified.npy" \
  --num_text_views 5 \
  --text_use_senet True \
  --use_cross False \
  --phase_a_grid \
  --align_grid "0.0,0.01,0.05" \
  --tau_grid "0.07" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 2 \
  --phase_a_valid_metric Recall@20 \
  --ndcg_baseline 0.08 \
  --ndcg_gain_threshold 0.005 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --text_gate_reg_l2 0.1 \
  --cross_dropout_prob 0.3 \
  --checkpoint_dir ./saved/phase_runs_fibinet_recall \
  --seed 2025 \
  --save

