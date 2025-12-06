#!/usr/bin/env bash
set -euo pipefail

# 重新生成 Amazon_Beauty 的 Qwen3 Multi-view 特征（修复白化问题后）

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================"
echo "重新生成 Qwen3 Multi-view 特征（修复版）"
echo "========================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 备份旧文件（如果存在）
if [ -d dataset/Amazon_Beauty/qwen3_4views ]; then
    backup_dir="dataset/Amazon_Beauty/qwen3_4views.backup_$(date +%s)"
    mv dataset/Amazon_Beauty/qwen3_4views "$backup_dir"
    echo "✅ 备份旧 multi-view 目录到: $backup_dir"
fi

if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
       dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy.backup_$(date +%s)
    echo "✅ 备份旧拼接文件"
fi

if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz \
       dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz.backup_$(date +%s)
    echo "✅ 备份统计文件"
fi

echo ""
echo "开始生成 Multi-view 特征..."
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Beauty/qwen3_4views \
  --view_project_dim 64 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "========================================"
echo "✅ Multi-view 生成完成！"
echo "========================================"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "生成的文件："
echo "  - 拼接特征: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy (shape: [N, 256])"
echo "  - 统计文件: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz"
echo "  - 视图0 (Identity):  dataset/Amazon_Beauty/qwen3_4views/view_0.npy (64维)"
echo "  - 视图1 (Function):  dataset/Amazon_Beauty/qwen3_4views/view_1.npy (64维)"
echo "  - 视图2 (Audience):  dataset/Amazon_Beauty/qwen3_4views/view_2.npy (64维)"
echo "  - 视图3 (Category):  dataset/Amazon_Beauty/qwen3_4views/view_3.npy (64维)"
echo "  - 元数据: dataset/Amazon_Beauty/qwen3_4views/views.json"
echo ""

echo "现在验证各视图的白化效果..."
echo ""

for i in 0 1 2 3; do
    echo "========================================"
    echo "验证视图 $i"
    echo "========================================"
    python tools/verify_whiten.py \
      dataset/Amazon_Beauty/qwen3_4views/view_${i}.npy \
      --dataset Amazon_Beauty
    echo ""
done

echo "========================================"
echo "验证拼接特征"
echo "========================================"
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
  --dataset Amazon_Beauty

echo ""
echo "========================================"
echo "全部完成！"
echo "========================================"

