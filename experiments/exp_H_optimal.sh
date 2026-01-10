#!/usr/bin/env bash
# Exp H: 综合最优 - tau=0.05, align=0.12, text_weight=1.0, cold_boost=2.5
# 目标：预期最佳组合，平衡所有指标
# 用法: bash experiments/exp_H_optimal.sh [GPU_ID]

GPU_ID=${1:-3}  # 默认 GPU 3

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Exp H: 综合最优 (align=0.12, cold_boost=2.5)"
echo "Using GPU: $GPU_ID"
echo "========================================="

python scripts/two_phase_train.py \
  --gpu_id "$GPU_ID" \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3_stratified.yaml" \
  --phase_a_grid \
  --align_grid "0.12" \
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
  --phase_b_alignment_weight 0.12 \
  --phase_b_temperature 0.05 \
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 1.0 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/grid_exp_H \
  --seed 2025 \
  --variant_features "exp_H,optimal,llm,beauty" \
  --watchdog_disable \
  --save

echo "✅ Exp H Done!"

# 运行: CUDA_VISIBLE_DEVICES=3 bash experiments/exp_H_optimal.sh
