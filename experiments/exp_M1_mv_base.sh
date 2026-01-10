#!/usr/bin/env bash
# Exp M1: Multi-View 7B 基础 - tau=0.05, align=0.10, text_weight=1.0, cold_boost=2.0
# 目标：Multi-View 基准配置
# 用法: bash experiments/exp_M1_mv_base.sh [GPU_ID]

GPU_ID=${1:-4}  # 默认 GPU 4

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Exp M1: Multi-View 7B 基础 (align=0.10)"
echo "Using GPU: $GPU_ID"
echo "========================================="

python scripts/two_phase_train.py \
  --gpu_id "$GPU_ID" \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_stratified.yaml" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 10 \
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
  --phase_b_temperature 0.05 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/grid_exp_M1 \
  --seed 2025 \
  --variant_features "exp_M1,mv_base,multiview,7b,beauty" \
  --watchdog_disable \
  --save

echo "✅ Exp M1 Done!"

# 运行: CUDA_VISIBLE_DEVICES=4 bash experiments/exp_M1_mv_base.sh
