#!/usr/bin/env bash
# SASRecAlignMultiViewV3 Baseline (40 epochs) with Stratified Evaluation
# 用于验证 MultiViewV3 模型在禁用文本特征后能还原 run50epBase_stratified 的效果
#
# 验证目的：
#   1. 确保 V3 架构的改动不影响纯 ID 基准性能
#   2. 验证 MultiViewV3 在禁用文本后与 SASRecAlignV3 baseline 效果一致
#
# 对比：
#   run50epBase_stratified.sh              -> SASRecAlign (原版基准)
#   run50epBase_v3_stratified.sh           -> SASRecAlignV3 (V3简化版基准)
#   run_multiview_v3_baseline_stratified.sh -> SASRecAlignMultiViewV3 (多视图禁用文本基准)
#
# 三者都是纯ID模型，禁用所有文本特征，效果应该一致

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# GPU ID support - can be set via environment variable or defaults to 0
GPU_ID=${GPU_ID:-0}
echo "Using GPU: $GPU_ID"

echo "========================================="
echo "SASRecAlignMultiViewV3 Baseline (40ep)"
echo "========================================="
echo "Dataset: Amazon_Beauty"
echo "Model: SASRecAlignMultiViewV3 (no text features, pure ID)"
echo ""
echo "V3 Config (all disabled for baseline):"
echo "  - align_weight: 0.0"
echo "  - cold_text_boost: 0.0"
echo "  - infer_boost: 0.0"
echo "  - use_text_view_split: false"
echo "  - disable_text_feature: true"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_baseline_stratified.yaml" \
  --gpu_id $GPU_ID

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""
echo "Expected: Results should match:"
echo "  - run50epBase_stratified.sh (SASRecAlign)"
echo "  - run50epBase_v3_stratified.sh (SASRecAlignV3)"
echo "All are pure ID models without text features."
echo ""
