#!/usr/bin/env bash
set -euo pipefail

# 仅生成Qwen3多视图特征（4 views, center + whiten全开）

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================"
echo "生成Qwen3多视图特征（4 views）"
echo "========================================"
echo ""

# 检查mapping文件是否存在
if [ ! -f "dataset/Amazon_Beauty/item_index_mapping.csv" ]; then
  echo "[准备] 生成item_index_mapping.csv..."
  python tools/export_internal_item_mapping.py \
    --dataset Amazon_Beauty \
    --config sasrec_base_plain.yaml \
    --output dataset/Amazon_Beauty/item_index_mapping.csv
  echo ""
fi

echo "[生成] Qwen3多视图特征（4个视图，每个64维，拼接为256维）..."
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
echo "✅ 生成完成！"
echo ""
echo "输出文件："
echo "  拼接特征: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy"
echo "  统计文件: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz"
echo "  视图目录: dataset/Amazon_Beauty/qwen3_4views/"
echo "    ├─ view_0.npy (Identity, 64-dim)"
echo "    ├─ view_1.npy (Function, 64-dim)"
echo "    ├─ view_2.npy (Audience, 64-dim)"
echo "    ├─ view_3.npy (Category, 64-dim)"
echo "    └─ views.json (元数据)"
echo ""
echo "4个视图提示词："
echo "  1. Identity:  'Identify the item: [TITLE] {text}'"
echo "  2. Function:  'What are the main functions and features of [TITLE] {text}?'"
echo "  3. Audience:  'Who is the target audience or user group for [TITLE] {text}?'"
echo "  4. Category:  'Categorize the item [TITLE] {text} and describe its context.'"
echo ""

