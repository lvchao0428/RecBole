#!/usr/bin/env bash
#set -euo pipefail

# Multi-View Split Training - Phase B Only (Low Memory Version)
#
# 用途：从 Phase A checkpoint 直接启动 Phase B，使用低内存配置
#
# 内存优化策略：
#   1. 跳过 Phase A（节省重复加载数据的内存）
#   2. 减小批次大小（train_batch_size 和 eval_batch_size）
#   3. 减少 DataLoader workers
#   4. 使用 CUDA 可扩展段
#
# 使用方法：
#   1. 运行 tools/find_phase_a_checkpoint.sh 查找 checkpoint
#   2. 修改下面的 PHASE_A_CHECKPOINT
#   3. 运行本脚本

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ==========================================
# CUDA 内存优化
# ==========================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 可选：限制 PyTorch 缓存大小
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# ==========================================
# 设置 Phase A Checkpoint
# ==========================================
# 运行 tools/find_phase_a_checkpoint.sh 查找最新的 checkpoint
PHASE_A_CHECKPOINT="./saved/phase_runs_multiview_4views/SASRecAlignMultiView-Amazon_Beauty-phase_a.pth"

# 检查 checkpoint
if [ ! -f "$PHASE_A_CHECKPOINT" ]; then
    echo "❌ Error: Phase A checkpoint not found: $PHASE_A_CHECKPOINT"
    echo ""
    echo "运行以下命令查找 checkpoint："
    echo "  bash tools/find_phase_a_checkpoint.sh"
    echo ""
    exit 1
fi

echo "========================================"
echo "Phase B Only - Low Memory Configuration"
echo "========================================"
echo "Checkpoint: $PHASE_A_CHECKPOINT"
echo ""
echo "内存优化措施："
echo "  ✅ 跳过 Phase A"
echo "  ✅ 减小批次大小（512）"
echo "  ✅ 减少 worker 数量（0）"
echo "  ✅ CUDA 可扩展段"
echo ""

# ==========================================
# 启动 Phase B
# ==========================================
python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view.yaml sasrec_align_multi_view_low_mem.yaml" \
  --only_phase_b \
  --resume_from "$PHASE_A_CHECKPOINT" \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --phase_b_epochs 40 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_4views \
  --seed 2025 \
  --variant_features "sasrec,multiview,4views,per_view_align,qwen3,phase_b_only,low_mem" \
  --watchdog_disable \
  --save

echo ""
echo "========================================"
echo "✅ Phase B 完成！"
echo "========================================"
echo ""
echo "如果仍然遇到 OOM 问题："
echo "  1. 进一步减小 train_batch_size 和 eval_batch_size"
echo "     编辑 sasrec_align_multi_view_low_mem.yaml"
echo ""
echo "  2. 使用梯度检查点（如果模型支持）"
echo "     在配置中添加 gradient_checkpointing: True"
echo ""
echo "  3. 减少序列长度"
echo "     在配置中设置 MAX_ITEM_LIST_LENGTH: 30"
echo ""
echo "  4. 监控 GPU 内存使用"
echo "     nvidia-smi --loop=1"
echo ""

