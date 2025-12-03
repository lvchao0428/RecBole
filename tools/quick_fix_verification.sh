#!/bin/bash
# 快速验证白化修复效果的脚本

set -e  # 遇到错误立即退出

echo "======================================================================="
echo "  Whitening Fix Verification Script"
echo "  验证 pre_svd_l2=False 修复后的白化效果"
echo "======================================================================="
echo ""

# 配置
DATASET=${1:-"Amazon_Beauty"}
OUTPUT_DIR="dataset/${DATASET}"
OUTPUT_FILE="${OUTPUT_DIR}/item_text_emb.base.fixed.npy"

echo "📋 配置信息:"
echo "   数据集: ${DATASET}"
echo "   输出路径: ${OUTPUT_FILE}"
echo ""

# 检查数据集是否存在
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "❌ 数据集目录不存在: ${OUTPUT_DIR}"
    echo "   请先准备数据集或指定正确的数据集名称"
    echo ""
    echo "用法: bash tools/quick_fix_verification.sh [dataset_name]"
    echo "示例: bash tools/quick_fix_verification.sh Amazon_Beauty"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 1/3: 生成新的 TF-IDF embeddings (pre_svd_l2=False)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python tools/build_item_text_emb_base.py \
  --dataset ${DATASET} \
  --output ${OUTPUT_FILE} \
  --svd_dim 256 \
  --dtype float16

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 生成embeddings失败"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 2/3: 验证白化效果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python tools/verify_whiten.py ${OUTPUT_FILE}

VERIFY_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 3/3: 结果总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "✅ 白化修复验证通过！"
    echo ""
    echo "📊 期望改进:"
    echo "   修复前: 对角线均值≈0.84, 非对角线均值≈0.07"
    echo "   修复后: 对角线均值≈0.95-1.05, 非对角线均值<0.05"
    echo ""
    echo "📁 生成的文件:"
    echo "   - Embeddings: ${OUTPUT_FILE}"
    echo "   - 统计量: ${OUTPUT_FILE%.npy}_whiten_stats.npz"
    echo ""
    echo "🎯 下一步:"
    echo "   1. 替换原有的 item_text_emb.base.npy"
    echo "   2. 重新训练模型验证性能提升"
else
    echo "⚠️  验证未完全通过，但这可能是正常的"
    echo ""
    echo "💡 说明:"
    echo "   - 对角线均值在0.9-1.1之间仍然可接受"
    echo "   - 非对角线均值<0.1表示白化有效"
    echo "   - 请检查上方的详细输出判断"
    echo ""
    echo "📁 生成的文件:"
    echo "   - Embeddings: ${OUTPUT_FILE}"
    echo "   - 统计量: ${OUTPUT_FILE%.npy}_whiten_stats.npz"
fi

echo ""
echo "======================================================================="
echo "  验证完成"
echo "======================================================================="

