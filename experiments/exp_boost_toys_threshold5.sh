#!/usr/bin/env bash
#
# exp_boost_toys_threshold5.sh - Toys + Inference Boost (threshold=5)
#
# 目的：测试 Toys 上 threshold=5 是否优于 threshold=10
#
# 配置对比：
# - threshold=10: 覆盖 new + few (当前默认)
# - threshold=5: 覆盖 new + 少量 few (本实验)
# - threshold=3: 精准覆盖 new (待测)
#
# 预期：
# - 如果 threshold=5 更好：过大的 threshold 会误伤 frequent 项目
# - 如果 threshold=10 更好：更广泛的覆盖有益

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Toys + Inference Boost (threshold=5)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.0"
echo "  - inference_cold_text_boost: 1.0"
echo "  - cold_start_align_threshold: 5"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_inference_boost_threshold5.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0249 \
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
  --checkpoint_dir ./saved/exp_boost_toys_threshold5 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,threshold5" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_boost_toys_threshold5/"
