#!/usr/bin/env bash
set -euo pipefail

# 仅生成TF-IDF基线特征（center + whiten全开）

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================"
echo "生成TF-IDF基线特征（center+whiten）"
echo "========================================"
echo ""

python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --title_field title \
  --svd_dim 256 \
  --svd_random_state 42 \
  --ngram_min 1 \
  --ngram_max 2 \
  --min_df 2 \
  --dtype float16

echo ""
echo "✅ 生成完成！"
echo "   - 特征: dataset/Amazon_Beauty/item_text_emb.base.npy"
echo "   - 统计: dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz"
echo ""

