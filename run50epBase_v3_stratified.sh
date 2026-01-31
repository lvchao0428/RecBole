#!/usr/bin/env bash
# SASRecAlignV3 Baseline (40 epochs) with Stratified Evaluation
# 用于还原 run50epBase_stratified.sh 的效果，但使用 V3 模型架构
# 
# 对比：
#   run50epBase_stratified.sh     -> SASRecAlign (原版)
#   run50epBase_v3_stratified.sh  -> SASRecAlignV3 (V3简化版)
#
# 两者都是纯ID模型，禁用所有文本特征，效果应该一致

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# GPU ID support - can be set via environment variable or defaults to 0
GPU_ID=${GPU_ID:-0}
echo "Using GPU: $GPU_ID"

echo "========================================="
echo "SASRecAlignV3 Baseline (40ep) with Stratified Metrics"
echo "========================================="
echo "Dataset: Amazon_Beauty"
echo "Model: SASRecAlignV3 (no text features, pure ID)"
echo ""
echo "V3 Config (all disabled for baseline):"
echo "  - align_weight: 0.0"
echo "  - cold_text_boost: 0.0"
echo "  - infer_boost: 0.0"
echo "  - disable_text_feature: true"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

python run_recbole.py \
  --model SASRecAlignV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_baseline_v3_50ep_stratified.yaml" \
  --gpu_id $GPU_ID

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""
echo "Expected: Results should match run50epBase_stratified.sh"
echo "Both are pure ID models without text features."
echo ""
