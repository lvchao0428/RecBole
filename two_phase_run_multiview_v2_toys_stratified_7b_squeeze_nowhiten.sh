#!/usr/bin/env bash
#set -euo pipefail

# Multi-View V2 Training - Toys Squeeze + NO WHITEN
# ============================================================
# 目标: 验证 Toys 的 no-whiten 趋势是否与 Beauty 一致
# 预期结果: HR↑, MRR↓ (和 Beauty 趋势一致)
#
# 如果结果符合预期，说明:
# 1. Toys 的 text 特征确实没有充分学习
# 2. 对齐 Beauty 配置后，text 学习充分
# 3. Whitening 的 trade-off 在两个数据集上一致
# ============================================================

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================="
echo "Multi-View V2 (7B) Toys SQUEEZE + NO WHITEN"
echo "========================================="
echo "Dataset: Amazon_Toys_and_Games"
echo "LLM Model: Qwen2.5-7B-Instruct"
echo ""
echo "Key Settings (aligned with Beauty):"
echo "  - alignment_weight: 0.10"
echo "  - temperature: 0.05"
echo "  - text_weight: 0.7"
echo "  - cold_start_align_boost: 3.0"
echo ""
echo "NO WHITEN:"
echo "  - Using center_only embeddings (no whitening)"
echo ""
echo "Expected: HR↑, MRR↓ (same trend as Beauty)"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b_squeeze_nowhiten.yaml" \
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
  --checkpoint_dir ./saved/phase_runs_multiview_v2_toys_squeeze_nowhiten \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,4views,toys,squeeze,no_whiten,center_only" \
  --watchdog_disable \
  --save

echo ""
echo "========================================="
echo "✅ Toys Squeeze + No Whiten Done!"
echo "========================================="
echo "Compare with Toys Squeeze (w/ whiten) to verify:"
echo "  Expected: HR↑, MRR↓ (same as Beauty trend)"
echo ""
