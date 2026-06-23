#!/usr/bin/env bash
set -euo pipefail

# Grocery: title + categories → TF-IDF + Qwen2.5-7B single + 4-view（同 Beauty/Toys）

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_NAME="Amazon_Grocery_and_Gourmet_Food"
MODEL_PATH="${QWEN_MODEL:-/home/charlie/project/qwen/Model}"
MODEL_NAME="qwen2.5_7b"
MAPPING_FILE="dataset/${DATASET_NAME}/item_index_mapping.csv"
PLAIN="sasrec_grocery_plain.yaml"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export TF_CPP_MIN_LOG_LEVEL=2

echo "========================================"
echo "Grocery 文本特征 (title+categories, center+whiten)"
echo "Dataset: $DATASET_NAME"
echo "Qwen: $MODEL_PATH"
echo "========================================"

echo "[0/4] TF-IDF..."
python tools/build_item_text_emb_base.py \
  --dataset "$DATASET_NAME" \
  --config "$PLAIN" \
  --output "dataset/${DATASET_NAME}/item_text_emb.base.npy" \
  --title_field title \
  --svd_dim 256 \
  --svd_random_state 42 \
  --ngram_min 1 \
  --ngram_max 2 \
  --min_df 2 \
  --dtype float16 \
  --center \
  --whiten
echo ""

echo "[1/4] item_index_mapping.csv..."
python tools/export_internal_item_mapping.py \
  --dataset "$DATASET_NAME" \
  --config "$PLAIN" \
  --output "$MAPPING_FILE"
echo ""

echo "[2/4] Qwen2.5-7B single-view..."
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output "dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.base.npy" \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset "$DATASET_NAME" \
  --config "$PLAIN" recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template \
  --center \
  --whiten
echo ""

echo "[3/4] Qwen2.5-7B 4-view..."
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output "dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.multiview.npy" \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir "dataset/${DATASET_NAME}/${MODEL_NAME}_4views" \
  --view_project_dim 64 \
  --dataset "$DATASET_NAME" \
  --config "$PLAIN" recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template \
  --center \
  --whiten
echo ""

echo "✅ Grocery embeddings done"
