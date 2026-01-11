#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Training with Stratified Evaluation
# 增强版Multi-View模型
#
# 改动点汇总（相比 two_phase_run_multiview_split_stratified.sh）：
# ============================================================
# 1. 模型: SASRecAlignMultiViewV2 (使用V2版本)
# 2. 每个view独立L2归一化（而不是跨view权重归一化）
# 3. multiview_align_scale: 1.0 (训练对齐损失)
# 4. text_view_senet_ratio: 2 (减少信息压缩)
# 5. alignment_weight: 0.10 (折中值，同时影响训练+推理)
# 6. text_weight: 0.7 (补偿CHANGE-7带来的推理放大)
# 7. cold_start_align_boost: 3.0 (开启冷启动加权)
# 8. cross_dropout_prob: 0.15 (降低以稳定排序)
# ============================================================

#cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU ID support - can be set via environment variable or defaults to 0
GPU_ID=${GPU_ID:-0}
echo "Using GPU: $GPU_ID"

echo "========================================="
echo "Multi-View V2 with Stratified Metrics"
echo "========================================="
echo "V2 Enhancements (with CHANGE-7: alignment affects inference):"
echo "  - Per-view L2 normalization (no cross-view weight normalization)"
echo "  - alignment_weight: 0.10 (affects both training loss AND inference fusion)"
echo "  - text_weight: 0.7 (compensate for align_scale × temp_scale amplification)"
echo "  - cold_start_align_boost: 3.0 (boost alignment for new/few items)"
echo "  - cross_dropout_prob: 0.15 (stabilize ranking)"
echo ""
echo "Inference text weight formula:"
echo "  effective = alpha × text_weight × (1 + align_weight) × (0.07 / temp)"
echo "            ≈ 0.67 × 0.7 × 1.10 × 1.4 ≈ 0.72"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_stratified.yaml" \
  --gpu_id $GPU_ID \
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
  --phase_a_text_gate_reg_l2 0.01 \
  --phase_b_alignment_weight 0.10 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_v2_stratified \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,4views,per_view_l2_norm,cold_boost,beauty,stratified" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "V2 Model Changes (with CHANGE-7):"
echo "  - Per-view L2 normalization (each view normalized independently)"
echo "  - alignment_weight=0.10 (affects training + inference)"
echo "  - text_weight=0.7 (compensate for inference amplification)"
echo "  - cold_start_align_boost=3.0 (boost new/few items)"
echo "  - cross_dropout_prob=0.15 (stabilize ranking)"
echo "  - text_view_senet_ratio=2 (128-dim reduction)"
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - MRR_new@10, MRR_few@10, MRR_frequent@10"
echo ""
