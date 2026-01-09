#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Training - Toys "Squeeze" Version
# ============================================================
# 目标: 压榨 Toys 性能，让 no-whiten 趋势与 Beauty 一致
# 假设: Toys 的 text 特征没有充分学习
# 方案: 对齐 Beauty 的成功配置
#
# 关键修改（对比原 toys_stratified_7b）:
# 1. alignment_weight: 0.10 (从 0.05 提升)
# 2. temperature: 0.05 (从 0.07 降低)
# 3. text_weight: 0.7 (从 1.0 降低)
# 4. cross_dropout_prob: 0.15 (从 0.2 降低)
# 5. cold_start_align_boost: 3.0 (开启，替代 IPW)
# 6. use_ipw_weighting: false (关闭 IPW)
# ============================================================

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Multi-View V2 (7B) Toys SQUEEZE Version"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo "LLM Model: Qwen2.5-7B-Instruct"
echo ""
echo "Key Changes (aligned with Beauty success):"
echo "  - alignment_weight: 0.10 (was 0.05)"
echo "  - temperature: 0.05 (was 0.07)"
echo "  - text_weight: 0.7 (was 1.0)"
echo "  - cross_dropout_prob: 0.15 (was 0.2)"
echo "  - cold_start_align_boost: 3.0 (was 0, using IPW)"
echo "  - use_ipw_weighting: false (was true)"
echo ""
echo "Hypothesis: Toys text features not fully learned"
echo "Goal: Make Toys no-whiten trend match Beauty"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b_squeeze.yaml" \
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
  --phase_b_text_gate_reg_l2 0.03 \
  --phase_b_text_weight 0.7 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_v2_toys_squeeze \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,4views,toys,squeeze,cold_boost,beauty_aligned" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Toys Squeeze Training Done!"
echo "========================================="
echo "Next steps:"
echo "  1. Compare with original Toys results"
echo "  2. Run no-whiten version to verify trend"
echo "  3. If successful, update paper analysis"
echo ""
