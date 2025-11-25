#!/usr/bin/env bash
#set -euo pipefail

# Multi-view Qwen3 workflow for Yelp dataset.
# Assumes tools/gen_text_emb_yelp.sh has already produced item_index_mapping.csv.

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/yelp/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/yelp/item_text_emb_amplified.npy \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --dataset yelp \
  --config yelp_config/yelp_sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --prompt_preset multiview \
  --output_mode concat \
  --view_project_dim 64 \
  --split_output_dir dataset/yelp/item_text_emb_amplified_views

