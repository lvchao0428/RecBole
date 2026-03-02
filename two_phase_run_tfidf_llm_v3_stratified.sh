#!/usr/bin/env bash
#set -euo pipefail

# Two-phase TF-IDF+LLM V3 with Stratified Evaluation
# 使用简化的 V3 权重配置
#
# V3 简化权重配置：
# ============================================================
# align_weight: 0.1       全局对齐权重
# cold_text_boost: 2.5    冷启动训练增强（训练时给低频商品更高对齐权重）
# infer_boost: 1.5        推理增强（推理时给低频商品更强文本信号）
# cold_threshold: 10      冷启动阈值
# ============================================================

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU ID support - can be set via environment variable or defaults to 0
GPU_ID=${GPU_ID:-0}
echo "Using GPU: $GPU_ID"

echo "========================================="
echo "TF-IDF+LLM V3 with Stratified Metrics"
echo "========================================="
echo ""
echo "V3 Simplified Weight Config:"
echo "  - align_weight: 0.1 (global alignment weight)"
echo "  - cold_text_boost: 2.5 (training cold-start boost)"
echo "  - infer_boost: 1.5 (inference cold-start boost)"
echo "  - cold_threshold: 10 (cold-start threshold)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified_v3.yaml" \
  --gpu_id $GPU_ID \
  --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0272 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 5e-4 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_v3_stratified \
  --seed 2025 \
  --variant_features "sasrec,tfidf,llm,v3,beauty,stratified" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "V3 Model Features:"
echo "  - Simplified 3-weight config (align_weight, cold_text_boost, infer_boost)"
echo "  - Removed redundant IPW, gate regularization, etc."
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - MRR_new@10, MRR_few@10, MRR_frequent@10"
echo ""
