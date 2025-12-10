#!/usr/bin/env bash
#set -euo pipefail

# Generate 4-view Qwen3 embeddings with per-view splits for Amazon_Toys_and_Games
# This script creates per-view embeddings for the multi-view alignment architecture.
#
# Prerequisites:
#   1. item_index_mapping.csv must exist (run gen_text_emb_toys.sh first if needed)
#   2. Qwen model must be available

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "[Multi-View 4-Views] Generating per-view Qwen3 embeddings for Toys_and_Games..."
echo "  - Output mode: split (per-view files)"
echo "  - View projection: 128 dims per view"
echo "  - Total views: 4"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Toys_and_Games/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Toys_and_Games/item_text_emb.qwen3.multiview.npy \
  --prompt_preset multiview-opt \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Toys_and_Games/qwen3_4views \
  --view_project_dim 64 \
  --dataset Amazon_Toys_and_Games \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "[Multi-View 4-Views] Done!"
echo "  - Concat embedding: dataset/Amazon_Toys_and_Games/item_text_emb_qwen3_4views.npy"
echo "  - Split views dir: dataset/Amazon_Toys_and_Games/item_text_emb_qwen3_4views_split/"
echo "  - Metadata: dataset/Amazon_Toys_and_Games/item_text_emb_qwen3_4views_split/views.json"
echo ""
echo "Next: Update sasrec_align_multi_view_toys.yaml to use:"
echo "  item_text_emb_split_dir: dataset/Amazon_Toys_and_Games/item_text_emb_qwen3_4views_split"
echo ""

