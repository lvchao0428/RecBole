#!/bin/bash
# exp_inference_boost_32b.sh
# 目的：测试 CHANGE-9 能否修复 Beauty 32B 的 Scale Law 失效问题
# 
# 当前发现：
# - Beauty 7B: MRR@10=0.0320, 14B: 0.0322, 32B: 0.0314 (Scale Law 失效)
# - inference_boost 在 7B 上成功修复 HR_new
#
# 预期：
# - 32B + CHANGE-9 能否提升 MRR 到超过 14B？
# - 32B + CHANGE-9 的 HR_new 能否转正？

GPU_ID=${GPU_ID:-0}

CONFIG_FILE="sasrec_align_multi_view_v2_inference_boost_32b.yaml"

echo "=========================================="
echo " exp_inference_boost_32b"
echo " Beauty 32B + CHANGE-9"
echo " GPU: $GPU_ID"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 1.0"
echo "  - cold_start_align_threshold: 10"
echo "  - Model: 32B"
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
    --variant_features "inference_boost_32b"

echo ""
echo "Experiment completed!"
