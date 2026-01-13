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
# 6. [IPW] use_ipw_weighting: true (平滑的逆倾向加权，替代 cold_start)
#    - ipw_threshold: 10 (与分层评估 frequent 阈值对齐)
#    - ipw_alpha: 0.5 (曲线形状)
#    - ipw_max_weight: 4.0 (最大权重)
# ============================================================

#cd /home/ubuntu/own/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU ID support - can be set via environment variable or defaults to 0
GPU_ID=${GPU_ID:-0}
echo "Using GPU: $GPU_ID"

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
echo "IPW (Inverse Propensity Weighting):"
echo "  - use_ipw_weighting: true (smoother than cold_start)"
echo "  - ipw_threshold: 10 (pop >= 10 → weight ≈ 1.0)"
echo "  - ipw_alpha: 0.5, ipw_max_weight: 4.0"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
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
  --config_dict "{'cold_start_align_boost': 2.0, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.0}" \
  --checkpoint_dir ./saved/phase_runs_multiview_v2_toys_stratified_final \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,stratified,final" \
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
echo "IPW alignment weighting:"
echo "  - threshold=10, alpha=0.5, max_weight=4.0"
echo "  - pop >= 10: weight ≈ 1.0 (frequent items)"
echo "  - pop < 10: weight smoothly increases (new/few items)"
echo ""
echo "Stratified metrics in results:"
echo "  - Recall_new@10, Recall_few@10, Recall_frequent@10"
echo "  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10"
echo "  - Coverage_new@10, Coverage_few@10, Coverage_frequent@10"
echo ""

