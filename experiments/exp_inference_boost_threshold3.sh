#!/bin/bash
# exp_inference_boost_threshold3.sh
# 目的：测试 threshold=3 精准对齐 "new" 分层定义 [0,3)
# 
# Threshold 对比：
# - threshold=10: 覆盖 new + few (当前默认)
# - threshold=5:  覆盖 new + 部分 few (正在测试)
# - threshold=3:  精准覆盖 new [0,3) (本实验)
#
# 预期：
# - 更精准的 boost 定义可能提升 HR_new 效果
# - 避免对 few 项目的过度干预

GPU_ID=${GPU_ID:-0}

CONFIG_FILE="sasrec_align_multi_view_v2_inference_boost_threshold3.yaml"

echo "=========================================="
echo " exp_inference_boost_threshold3"
echo " Beauty 7B + threshold=3"
echo " GPU: $GPU_ID"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 1.0"
echo "  - cold_start_align_threshold: 3  <-- 精准匹配 new [0,3)"
echo ""

python scripts/two_phase_train.py \
    --config_file ${CONFIG_FILE} \
    --gpu_id ${GPU_ID} \
    --backbone_burnin_epochs 0 \
    --phase_a_grid \
    --tau_grid "0.05" \
    --align_grid "0.10" \
    --phase_a_epochs 20 \
    --phase_b_epochs 40 \
    --phase_b_text_weight 1.0 \
    --phase_b_alignment_weight 0.10 \
    --variant_features "inference_boost_threshold3"

echo ""
echo "Experiment completed!"
