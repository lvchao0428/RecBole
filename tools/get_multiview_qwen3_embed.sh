#!/usr/bin/env bash
#set -euo pipefail

# Generate multi-view Qwen3 embeddings (concat + optional SVD + per-view splits)

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_amplified.npy \
  --batch_size 16 --max_length 128 --dtype float16 \
  --dataset Amazon_Beauty \
  --config /home/charlie/project/RecBole/sasrec_base_plain.yaml \
  --prompt_preset multiview \
  --output_mode concat \
  --view_project_dim 64 \
  --split_output_dir /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_amplified_views


