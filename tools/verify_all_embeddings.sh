#!/usr/bin/env bash
set -euo pipefail

# 批量验证所有生成的文本特征

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================"
echo "批量验证所有文本特征的白化效果"
echo "========================================"
echo ""

# 收集所有需要验证的文件
files_to_verify=()

# TF-IDF基线
if [ -f "dataset/Amazon_Beauty/item_text_emb.base.npy" ]; then
    files_to_verify+=("dataset/Amazon_Beauty/item_text_emb.base.npy")
fi

# Qwen3单视图
if [ -f "dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy" ]; then
    files_to_verify+=("dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy")
fi

# Qwen3多视图（拼接版本）
if [ -f "dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy" ]; then
    files_to_verify+=("dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy")
fi

# Qwen3分视图
for view_file in dataset/Amazon_Beauty/qwen3_4views/view_*.npy; do
    if [ -f "$view_file" ]; then
        files_to_verify+=("$view_file")
    fi
done

# 检查是否有文件需要验证
if [ ${#files_to_verify[@]} -eq 0 ]; then
    echo "❌ 未找到任何embedding文件"
    echo ""
    echo "请先运行以下命令生成特征："
    echo "  bash tools/gen_text_emb_beauty_full.sh"
    echo ""
    exit 1
fi

echo "找到 ${#files_to_verify[@]} 个文件待验证："
for f in "${files_to_verify[@]}"; do
    echo "  - $f"
done
echo ""

# 运行验证
python tools/verify_whiten.py "${files_to_verify[@]}"

echo ""
echo "========================================"
echo "验证完成"
echo "========================================"

