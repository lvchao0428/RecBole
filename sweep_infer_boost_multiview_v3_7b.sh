#!/usr/bin/env bash
# Infer Boost Sweep - Multi-View V3 (7B) Beauty
#
# 用途：加载已训练的 checkpoint，在 validation 上扫描 infer_boost 参数
# 找到最优 γ* 后在 test 上报告最终结果
#
# 使用前：
#   1. 确保 checkpoint 路径正确
#   2. 修改 CHECKPOINT 变量为实际路径

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ==========================================
# 配置区域 - 根据实际情况修改
# ==========================================

# Checkpoint 路径（需要修改为实际路径）
# 查找命令: ls -lht saved/phase_runs_multiview_v3_stratified/*.pth
CHECKPOINT="./saved/phase_runs_multiview_v3_stratified/SASRecAlignMultiViewV3-Jan-29-2026_18-35-38.pth"

# 模型和数据集
MODEL="SASRecAlignMultiViewV3"
DATASET="Amazon_Beauty"
CONFIG_FILES="sasrec_align_multi_view_v3_stratified.yaml"

# 网格搜索参数
# 粗扫: 6-8 个点
#INFER_BOOST_GRID="0.0,0.5,0.8,1.0,1.2,1.5,2.0"
INFER_BOOST_GRID="0.6"

# 验证指标
VALID_METRIC="MRR@10"

# 变体标签（用于日志和结果文件命名）
VARIANT_LABEL="multiview_v3_7b_beauty"

# GPU
GPU_ID=${GPU_ID:-0}

# ==========================================
# 检查 checkpoint
# ==========================================
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "请检查："
    echo "  1. 训练是否已完成并保存了 checkpoint"
    echo "  2. Checkpoint 路径是否正确"
    echo ""
    echo "查找 checkpoint:"
    echo "  ls -lht saved/phase_runs_multiview_v3_stratified/*.pth"
    exit 1
fi

# ==========================================
# 运行 sweep
# ==========================================
echo "=========================================="
echo "Infer Boost Sweep"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Grid: $INFER_BOOST_GRID"
echo "Metric: $VALID_METRIC"
echo "GPU: $GPU_ID"
echo "=========================================="
echo ""

python scripts/infer_boost_sweep.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --config_files "$CONFIG_FILES" \
  --checkpoint "$CHECKPOINT" \
  --infer_boost_grid "$INFER_BOOST_GRID" \
  --valid_metric "$VALID_METRIC" \
  --variant_label "$VARIANT_LABEL" \
   --skip_fine_sweep \
  --gpu_id "$GPU_ID" \
  --output_dir "./run_metrics/infer_boost_sweep"

echo ""
echo "=========================================="
echo "✅ Sweep Done!"
echo "=========================================="
echo "Results saved to: ./run_metrics/infer_boost_sweep/"
echo ""
