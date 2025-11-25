#!/usr/bin/env bash
set -euo pipefail

cd /home/charlie/project/RecBole

echo "[Beauty] Generating TF-IDF (base) embeddings..."
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --svd_random_state 42 \
  --dtype float16

echo "[Beauty] Exporting item index mapping for LLM embedding..."
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Beauty/item_index_mapping.csv

echo "[Beauty] Generating Qwen3 multi-view embeddings..."
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
  --batch_size 16 \
  --max_length 0 \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --dtype float16 \
  --prompt_preset multiview \
  --output_mode concat \
  --view_project_dim 64 \
  --split_output_dir dataset/Amazon_Beauty/item_text_emb_amplified_views

echo "[Beauty] Done."

