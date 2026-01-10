#!/usr/bin/env bash
# Exp Burn0: 无 Burn-in 测试 - 验证 Phase-A 是否能在无 burn-in 情况下正常涨点
# 假设：10 epoch burn-in 导致 ID 过拟合，Phase-A 无法追上
# 用法: bash experiments/exp_burn0_no_burnin.sh [GPU_ID]

GPU_ID=${1:-0}  # 默认 GPU 0

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Exp Burn0: 无 Burn-in 测试"
echo "测试假设: Phase-A 在无 burn-in 时能正常涨点"
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
  --phase_b_temperature 0.05 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/grid_exp_burn0 \
  --seed 2025 \
  --variant_features "exp_burn0,no_burnin,multiview,7b,beauty" \
  --watchdog_disable \
  --save

echo "✅ Exp Burn0 Done!"
