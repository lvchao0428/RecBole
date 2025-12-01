#!/usr/bin/env bash
#set -euo pipefail

cd /home/charlie/project/RecBole

echo "[Toys] Generating TF-IDF (base) embeddings..."
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_align_base.yaml \
  --output dataset/Amazon_Toys_and_Games/item_text_emb.base.npy \
  --title_field title \
  --svd_dim 256 \
  --svd_random_state 42 \
  --dtype float16

echo "[Toys] Exporting item index mapping for LLM embedding..."
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_align_base.yaml \
  --output dataset/Amazon_Toys_and_Games/item_index_mapping.csv

echo "[Toys] Generating Qwen3 base-prompt embeddings..."
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Toys_and_Games/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Toys_and_Games/item_text_emb.qwen3.npy \
  --batch_size 16 \
  --max_length 0 \
  --project_dim 256 \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_align_base.yaml recbole/properties/overall.yaml \
  --dtype float16 \
  --prompt_preset base \
  --output_mode mean

echo "[Toys] Done! Generated embeddings:"
echo "  - TF-IDF: dataset/Amazon_Toys_and_Games/item_text_emb.base.npy"
echo "  - Qwen3: dataset/Amazon_Toys_and_Games/item_text_emb.qwen3.npy"
echo "  - Mapping: dataset/Amazon_Toys_and_Games/item_index_mapping.csv"
echo ""
echo "Run tools/get_multiview_qwen3_embed.sh for multi-view features if needed."


