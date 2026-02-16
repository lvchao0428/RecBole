#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Training with Stratified Evaluation (Amazon_Toys_and_Games Dataset)
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

base_dir='/home/charlie/project/RecBole'
cd ${base_dir}
yaml_dir='/home/charlie/project/RecBole/toy_train'
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Multi-View V2 (7B) with Stratified Metrics (Toys)"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
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

python scripts/two_phase_train.py \
--model SASRecAlignMultiViewV3 \
--dataset Amazon_Toys_and_Games \
--config_files "${yaml_dir}/sasrec_align_multi_view_v3_toys_stratified_7b_nosenet.yaml" \
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
--checkpoint_dir ./saved/phase_runs_multiview_v3_toys_stratified_7b \
--seed 2025 \
--variant_features "sasrec,multiview_v2,7b,4views,per_view_l2_norm,align_scale_2x,toys,stratified" \
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

