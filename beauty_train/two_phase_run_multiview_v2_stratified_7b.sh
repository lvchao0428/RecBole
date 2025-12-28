#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Training with Stratified Evaluation (Amazon_Beauty Dataset)
# 增强版Multi-View模型 - 使用 Qwen2.5-7B embeddings
#
# 改动点汇总（相比 two_phase_run_multiview_split_stratified.sh）：
# ============================================================
# 1. 模型: SASRecAlignMultiViewV2 (使用V2版本)
# 2. 每个view独立L2归一化（而不是跨view权重归一化）
# 3. multiview_align_scale: 2.0 (对齐损失放大)
# 4. text_view_senet_ratio: 2 (减少信息压缩)
# 5. alignment_weight: 0.15 (从0.05提升到0.15，3x)
# ============================================================

#base_dir='/home/charlie/project/RecBole'
base_dir='/home/ubuntu/own/RecBole/'
cd ${base_dir}
#yaml_dir='/home/charlie/project/RecBole/beauty_train'
yaml_dir='/home/ubuntu/own/RecBole/beauty_train'
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Multi-View V2 (7B) with Stratified Metrics (Beauty)"
echo "========================================="
echo "Dataset: Amazon_Beauty"
echo "LLM Model: Qwen2.5-7B-Instruct"
echo ""
echo "V2 Enhancements:"
echo "  - Per-view L2 normalization (no cross-view weight normalization)"
echo "  - multiview_align_scale: 2.0 (alignment loss amplification)"
echo "  - text_view_senet_ratio: 2 (less compression)"
echo "  - alignment_weight: 0.15 (3x stronger)"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python ${base_dir}/scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "${yaml_dir}/sasrec_align_multi_view_v2_stratified_7b.yaml" \
  --phase_a_grid \
  --align_grid "0.03" \
  --tau_grid "0.1" \
  --backbone_burnin_epochs 10 \
  --burnin_eval_step 2 \
  --phase_a_epochs 15 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0272 \
  --metric_gain_threshold 0.01 \
  --lr_text_head 2e-3 \
  --lr_dnn_cross 1e-3 \
  --phase_a_text_gate_reg_l2 0.05 \
  --phase_b_alignment_weight 0.10 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_v2_beauty_stratified \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,4views,per_view_l2_norm,align_scale_2x,beauty,stratified" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Training Done!"
echo "========================================="
echo "V2 Model Changes:"
echo "  - Per-view L2 normalization (each view normalized independently)"
echo "  - No cross-view weight normalization (each view contributes 0~1)"
echo "  - multiview_align_scale=2.0 (alignment loss 2x stronger)"
echo "  - alignment_weight=0.15 (3x vs original 0.05)"
echo "  - text_view_senet_ratio=2 (128-dim reduction vs 64-dim)"
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - Coverage_new@10, Coverage_few@10, Coverage_frequent@10"
echo ""
