#!/usr/bin/env bash
set -euo pipefail

# 重新生成 Amazon_Beauty 的 Qwen3 特征（修复白化问题后）

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================"
echo "重新生成 Qwen3 单视图特征（修复版）"
echo "========================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 备份旧文件（如果存在）
if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
       dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy.backup_$(date +%s)
    echo "✅ 备份旧文件"
fi

if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz \
       dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz.backup_$(date +%s)
    echo "✅ 备份统计文件"
fi

echo ""
echo "开始生成..."

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
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
echo "✅ 生成完成！"
echo "========================================"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "现在验证白化效果..."
echo ""

python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --dataset Amazon_Beauty

echo ""
echo "========================================"
echo "全部完成！"
echo "========================================"

