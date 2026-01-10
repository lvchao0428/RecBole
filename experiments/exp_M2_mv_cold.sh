#!/usr/bin/env bash
# Exp M2: Multi-View 7B 冷启动增强 - tau=0.05, align=0.12, text_weight=1.0, cold_boost=3.0
# 目标：增强冷启动商品推荐
# 用法: bash experiments/exp_M2_mv_cold.sh [GPU_ID]

GPU_ID=${1:-5}  # 默认 GPU 5 (折中方案: burn-in=2)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Exp M2: Multi-View 7B 冷启动 (cold_boost=3.0)"
echo "Using GPU: $GPU_ID"
echo "========================================="

python scripts/two_phase_train.py \
  --gpu_id "$GPU_ID" \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_stratified.yaml" \
  --phase_a_grid \
  --align_grid "0.12" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 2 \
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
  --checkpoint_dir ./saved/grid_exp_M2 \
  --seed 2025 \
  --variant_features "exp_M2,mv_cold,multiview,7b,beauty" \
  --watchdog_disable \
  --save

echo "✅ Exp M2 Done!"

# 运行: CUDA_VISIBLE_DEVICES=5 bash experiments/exp_M2_mv_cold.sh
# 注意: 需要在 YAML 中设置 cold_start_align_boost=3.0
