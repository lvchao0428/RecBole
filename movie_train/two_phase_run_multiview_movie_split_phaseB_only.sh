#!/usr/bin/env bash
#set -euo pipefail

# Multi-View Split Training - Phase B Only
#
# 用途：从 Phase A checkpoint 直接启动 Phase B（适用于 OOM 场景）
#
# 使用方法：
#   1. 确保 Phase A 已经完成并保存了 checkpoint
#   2. 修改下面的 PHASE_A_CHECKPOINT 为实际的 checkpoint 路径
#   3. 运行本脚本
#
# 内存优化建议：
#   - 如果仍然 OOM，可以减小 eval_batch_size
#   - 可以在 config 中调整其他批次大小参数

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==========================================
# 重要：设置 Phase A checkpoint 路径
# ==========================================
# 示例路径（需要根据实际情况修改）：
# PHASE_A_CHECKPOINT="./saved/phase_runs_multiview_4views/SASRecAlignMultiView-Amazon_Beauty-Dec06-2024-10h30m15s-phase_a.pth"
# 
# 如何找到 checkpoint：
#   1. 查看 saved/phase_runs_multiview_4views/ 目录
#   2. 找到最新的 *-phase_a.pth 文件
#   3. 或者查看 Phase A 运行日志中的 "[Phase-A] Saved checkpoint: ..." 输出

PHASE_A_CHECKPOINT="./saved/phase_runs_multiview_4views_movies_stratified/SASRecAlignMultiView-Dec-16-2025_06-51-03.pth"

# 检查 checkpoint 是否存在
if [ ! -f "$PHASE_A_CHECKPOINT" ]; then
    echo "❌ Error: Phase A checkpoint not found at: $PHASE_A_CHECKPOINT"
    echo ""
    echo "请检查："
    echo "  1. Phase A 是否已经运行完成"
    echo "  2. Phase A 运行时是否使用了 --save 参数"
    echo "  3. Checkpoint 路径是否正确"
    echo ""
    echo "查找 checkpoint："
    echo "  ls -lht saved/phase_runs_multiview_4views/*-phase_a.pth"
    echo ""
    exit 1
fi

echo "========================================"
echo "Phase B Only - Multi-View Split Training"
echo "========================================"
echo "从 Phase A checkpoint 恢复："
echo "  $PHASE_A_CHECKPOINT"
echo ""
echo "内存优化："
echo "  - 跳过 Phase A（节省内存）"
echo "  - 直接加载已训练的权重"
echo "  - CUDA 使用可扩展段"
echo ""

python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Movies_and_TV \
  --config_files "sasrec_align_multi_view_movies_stratified.yaml" \
  --only_phase_b \
  --resume_from "$PHASE_A_CHECKPOINT" \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --phase_b_epochs 40 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_4views_movies_stratified \
  --seed 2025 \
  --variant_features "sasrec,multiview,4views,per_view_align,qwen3,movies,stratified" \
  --watchdog_disable \
  --save

echo ""
echo "========================================"
echo "✅ Phase B 完成！"
echo "========================================"
echo "检查结果："
echo "  - 日志目录: saved/phase_runs_multiview_4views/"
echo "  - Phase B checkpoint: *-phase_b.pth"
echo ""

