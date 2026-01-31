#!/usr/bin/env bash
# SASRecAlignMultiViewV3 Baseline (40 epochs) with Stratified Evaluation
# 用于还原 run50epBase_stratified.sh 的效果
# 
# 这个脚本禁用所有多视图文本特征，让 SASRecAlignMultiViewV3 退化为纯 ID 模型
# 验证模型架构在禁用文本时是否能达到与 SASRecAlign 一致的基准效果
#
# 对比：
#   run50epBase_stratified.sh                     -> SASRecAlign, pure ID
#   run50epBase_v3_stratified.sh                  -> SASRecAlignV3, pure ID
#   two_phase_run_multiview_v3_baseline_stratified.sh -> SASRecAlignMultiViewV3, pure ID (本脚本)
#
# 期望：三者效果应该一致

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# GPU ID support - can be set via environment variable or defaults to 0
GPU_ID=${GPU_ID:-0}
echo "Using GPU: $GPU_ID"

echo "========================================="
echo "SASRecAlignMultiViewV3 Baseline (40ep)"
echo "========================================="
echo "Dataset: Amazon_Beauty"
echo "Model: SASRecAlignMultiViewV3 (text features DISABLED)"
echo ""
echo "Disabled Features:"
echo "  - use_text_view_split: false"
echo "  - use_multiview_text_cross: false"
echo "  - disable_text_feature: true"
echo "  - fuse_text_feature: false"
echo "  - align_weight: 0.0"
echo ""
echo "V3 Config (all disabled for baseline):"
echo "  - align_weight: 0.0"
echo "  - cold_text_boost: 0.0"
echo "  - infer_boost: 0.0"
echo ""
echo "Item Stratification:"
echo "  - new:      [1, 3)   interactions"
echo "  - few:      [3, 10)  interactions"
echo "  - frequent: [10, +∞) interactions"
echo ""

# 直接使用 run_recbole.py 进行训练（无需 two_phase_train.py，因为是纯 ID 模型）
python run_recbole.py \
  --model SASRecAlignMultiViewV3 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v3_baseline_stratified.yaml" \
  --gpu_id $GPU_ID

echo ""
echo "✅ Training Done! Check results for stratified metrics."
echo ""
echo "Expected: Results should match:"
echo "  - run50epBase_stratified.sh (SASRecAlign baseline)"
echo "  - run50epBase_v3_stratified.sh (SASRecAlignV3 baseline)"
echo ""
echo "All three are pure ID models without text features."
echo ""
