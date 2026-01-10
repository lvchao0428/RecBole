#!/usr/bin/env bash
# ============================================================
# Exp M5: 最小 Phase-A 测试 (5090 专用)
# ============================================================
# 
# 核心假设：Phase-A 长时间 freeze backbone 导致 ID-Text 不兼容
# 测试方案：最小化 Phase-A，让 Phase-B 承担主要学习任务
#
# 对比实验 (总训练量相同 = 62ep)：
#   - exp_burn2: Burn-in(2) → Phase-A(20, freeze) → Phase-B(40)
#   - exp_M5:    Burn-in(2) → Phase-A(5, freeze)  → Phase-B(55)
#
# 设计依据：
#   - 业界 warmup 通常 5-10 epochs
#   - 5 epochs 足够文本层建立初步表征
#   - 不至于让 backbone 被锁定太久 (20ep → 5ep)
#
# 预期：
#   - 如果 M5 效果更好 → Phase-A 20ep 太久是问题
#   - 如果 M5 效果差 → Phase-A 20ep 有其必要性
# ============================================================

GPU_ID=${1:-0}  # 5090 卡

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 5090 优化：增大 batch size
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "Exp M5: 短 Phase-A 测试 (5090)"
echo "========================================="
echo "方案：Burn-in(2ep) → Phase-A(5ep) → Phase-B(55ep)"
echo "对照：exp_burn2 (Phase-A 20ep + Phase-B 40ep)"
echo "GPU: $GPU_ID"
echo "========================================="

python scripts/two_phase_train.py \
  --gpu_id "$GPU_ID" \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view_v2_stratified.yaml" \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 2 \
  --burnin_eval_step 1 \
  --phase_a_epochs 5 \
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
  --phase_b_epochs 55 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/grid_exp_M5 \
  --seed 2025 \
  --variant_features "exp_M5,short_phaseA_5ep,long_phaseB_55ep,multiview,7b,beauty,5090" \
  --watchdog_disable \
  --save

echo ""
echo "✅ Exp M5 Done!"
echo ""
echo "============================================"
echo "关键对比 (与 exp_burn2):"
echo "============================================"
echo "| 阶段 | exp_burn2 | exp_M5 |"
echo "|------|-----------|--------|"
echo "| Burn-in | 2ep | 2ep |"
echo "| Phase-A (freeze) | 20ep | 5ep |"
echo "| Phase-B (unfreeze) | 40ep | 55ep |"
echo "| 总训练 | 62ep | 62ep |"
echo "============================================"
echo ""
echo "设计依据："
echo "- 业界 warmup 通常 5-10 epochs"
echo "- 5 epochs 足够文本层建立初步表征"
echo "- 20ep → 5ep: 减少 backbone 锁定时间"
echo ""
echo "观察重点："
echo "1. Phase-A 5ep 后 valid_score 是否合理"
echo "2. Phase-B 收敛速度是否更快"
echo "3. 最终 MRR@10, MRR_new@10 对比"
