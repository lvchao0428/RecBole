#!/usr/bin/env bash
#
# exp_seed42_beauty_tfidf_llm.sh - Seed Experiment: TF-IDF+LLM on Beauty with seed=42
#
# 目的: 验证主表结果的统计显著性
# 配置: 与主表一致 (Aggressive: cold=2.5, infer=1.5)
# seed: 42
#
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Seed Experiment: TF-IDF+LLM on Beauty (seed=42)"
echo "Using GPU: $GPU_ID"
echo "========================================="

python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified.yaml" \
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5}" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0207 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.10 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/seed42_beauty_tfidf_llm \
  --seed 42 \
  --variant_features "sasrec,tfidf_llm,beauty,seed42" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/seed42_beauty_tfidf_llm/"
