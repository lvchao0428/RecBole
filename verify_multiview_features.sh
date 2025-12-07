#!/usr/bin/env bash
# 验证 Multi-View 特征文件是否存在

echo "========================================"
echo "验证 Multi-View 特征文件"
echo "========================================"
echo ""

DATASET_DIR="/home/charlie/project/RecBole/dataset/Amazon_Beauty"

# 检查必需的特征文件
echo "检查 Concat 模式所需文件："
echo ""

FILES=(
    "item_text_emb.base.npy"
    "item_text_emb.base_whiten_stats.npz"
    "item_text_emb.qwen3.multiview.npy"
    "item_text_emb.qwen3.multiview_whiten_stats.npz"
)

MISSING=0

for file in "${FILES[@]}"; do
    filepath="$DATASET_DIR/$file"
    if [ -f "$filepath" ]; then
        size=$(ls -lh "$filepath" | awk '{print $5}')
        echo "  ✅ $file ($size)"
    else
        echo "  ❌ $file (缺失)"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "检查 Split 模式所需文件："
echo ""

SPLIT_DIR="$DATASET_DIR/qwen3_4views"
if [ -d "$SPLIT_DIR" ]; then
    echo "  ✅ qwen3_4views/ 目录存在"
    
    SPLIT_FILES=("view_0.npy" "view_1.npy" "view_2.npy" "view_3.npy" "views.json")
    for file in "${SPLIT_FILES[@]}"; do
        filepath="$SPLIT_DIR/$file"
        if [ -f "$filepath" ]; then
            size=$(ls -lh "$filepath" | awk '{print $5}')
            echo "    ✅ $file ($size)"
        else
            echo "    ❌ $file (缺失)"
            MISSING=$((MISSING + 1))
        fi
    done
else
    echo "  ❌ qwen3_4views/ 目录不存在"
    MISSING=$((MISSING + 5))
fi

echo ""
echo "========================================"
echo "验证结果"
echo "========================================"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "✅ 所有特征文件都存在！"
    echo ""
    echo "可以运行："
    echo "  - Concat 模式: bash two_phase_run_multiview_concat.sh"
    echo "  - Split 模式:  bash two_phase_run_multiview_split.sh"
    echo ""
else
    echo "⚠️  缺失 $MISSING 个文件"
    echo ""
    echo "请运行以下命令生成特征："
    echo "  bash tools/gen_text_emb_beauty_full_fast.sh"
    echo ""
fi

echo "========================================"

