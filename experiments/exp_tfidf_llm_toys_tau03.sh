#!/usr/bin/env bash
#
# exp_tfidf_llm_toys_tau03.sh - TF-IDF+LLM Toys with Lower Temperature
#
# 目的：修复 Toys 上 TF-IDF+LLM < TF-IDF 的层级反转问题
# 配置：temperature 从 0.05 降低到 0.03
#
# 理论依据：
# - 低温度 = 更尖锐的 softmax = 更强的正负样本区分
# - LLM 嵌入可能需要更强的对比信号
#
# 预期：
# - 如果 MRR > 0.0373：温度是 LLM 效果的关键
# - 如果失败：需要其他方案

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: TF-IDF+LLM Toys (tau=0.03)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - temperature: 0.03 (reduced from 0.05)"
echo "  - Goal: Fix LLM < TF-IDF hierarchy issue"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlign \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_qwen3_toys_tau03.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.03" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0249 \
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
  --checkpoint_dir ./saved/exp_tfidf_llm_toys_tau03 \
  --seed 2025 \
  --variant_features "sasrec,tfidf_llm,toys,tau03" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_tfidf_llm_toys_tau03/"
