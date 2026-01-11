#!/bin/bash
# exp_inference_boost_toys_14b.sh
# 目的：验证 Toys 14B + CHANGE-9 的 Scale Law
# 
# 当前发现：
# - Toys HR_new 呈现 Scale Law: 7B(+5%) < 14B(+6%) < 32B(+9%)
# - Toys MV-14B MRR@10=0.0373 (最好)
#
# 预期：
# - CHANGE-9 能否进一步提升 Toys 14B 的 HR_new？
# - MRR 是否能保持或提升？

GPU_ID=${GPU_ID:-0}

CONFIG_FILE="sasrec_align_multi_view_v2_toys_inference_boost_14b.yaml"

echo "=========================================="
echo " exp_inference_boost_toys_14b"
echo " Toys 14B + CHANGE-9"
echo " GPU: $GPU_ID"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 1.0"
echo "  - cold_start_align_threshold: 10"
echo "  - Dataset: Amazon_Toys_and_Games"
echo "  - Model: 14B"
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
    --variant_features "inference_boost_toys_14b"

echo ""
echo "Experiment completed!"
