#!/usr/bin/env bash
#
# exp_unified_standard_beauty.sh - Beauty 使用 Toys 的标准参数
#
# 目的：验证统一参数的可行性 (对称实验)
# 配置：cold=2.0, infer=1.0 (来自 Toys 最佳)
#
# 理论依据：
# - Beauty 当前最佳是 aggressive (cold=2.5, infer=1.5)
# - 如果 standard 参数也能达到接近效果，说明参数鲁棒性强
# - 这是统一参数方案的必要验证

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Experiment: Unified Standard (Beauty)"
echo "Using GPU: $GPU_ID"
echo "========================================="
echo "  - cold_start_align_boost: 2.0 (standard)"
echo "  - inference_cold_text_boost: 1.0 (standard)"
echo "  - Source: Toys' best config"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_beauty_unified_standard.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --burnin_eval_step 2 \
  --phase_a_epochs 20 \
  --phase_a_eval_step 1 \
  --phase_a_valid_metric "MRR@10" \
  --metric_baseline 0.0207 \
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
  --checkpoint_dir ./saved/exp_unified_standard_beauty \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,beauty,unified_standard" \
  --watchdog_disable \
  --save

echo "✅ Done! Check: saved/exp_unified_standard_beauty/"
