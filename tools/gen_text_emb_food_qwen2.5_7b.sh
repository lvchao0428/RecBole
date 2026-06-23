#!/usr/bin/env bash
set -euo pipefail

# Food 文本特征：name + nutrition + tags 全量拼接 → TF-IDF + Qwen2.5-7B single/MV
# 4-view prompt 与 Beauty/Toys 相同；{text} 为统一全字段文本，center+whiten

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_NAME="Food"
MODEL_PATH="${QWEN_MODEL:-/home/charlie/project/qwen/Model}"
MODEL_NAME="qwen2.5_7b"
MAPPING_FILE="dataset/${DATASET_NAME}/item_index_mapping.csv"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export TF_CPP_MIN_LOG_LEVEL=2

echo "========================================"
echo "Food 文本特征 (name+tags full, center+whiten)"
echo "========================================"
echo "Dataset: $DATASET_NAME"
echo "Qwen model: $MODEL_PATH"
echo ""

echo "[0/4] Prepare item_index_mapping.csv (name + nutrition + tags)..."
python tools/prepare_food_item_mapping.py \
  --dataset "$DATASET_NAME" \
  --config sasrec_food_plain.yaml \
  --output "$MAPPING_FILE"
echo ""

echo "[1/4] TF-IDF (mapping_csv=name+nutrition+tags)..."
python tools/build_item_text_emb_base.py \
  --dataset "$DATASET_NAME" \
  --config sasrec_food_plain.yaml \
  --mapping_csv "$MAPPING_FILE" \
  --output "dataset/${DATASET_NAME}/item_text_emb.base.npy" \
  --svd_dim 256 \
  --svd_random_state 42 \
  --ngram_min 1 \
  --ngram_max 2 \
  --min_df 2 \
  --dtype float16 \
  --center \
  --whiten
echo ""

echo "[2/4] Qwen2.5-7B single-view (prompt_preset=base)..."
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output "dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.base.npy" \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset "$DATASET_NAME" \
  --config sasrec_food_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template \
  --center \
  --whiten
echo ""

echo "[3/4] Qwen2.5-7B 4-view (prompt_preset=multiview, same as Beauty/Toys)..."
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output "dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.multiview.npy" \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir "dataset/${DATASET_NAME}/${MODEL_NAME}_4views" \
  --view_project_dim 64 \
  --dataset "$DATASET_NAME" \
  --config sasrec_food_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template \
  --center \
  --whiten
echo ""

echo "========================================"
echo "✅ Food embeddings done"
echo "  TF-IDF:  dataset/Food/item_text_emb.base.npy"
echo "  LLM 1v:  dataset/Food/item_text_emb.${MODEL_NAME}.base.npy"
echo "  LLM 4v:  dataset/Food/qwen2.5_7b_4views/view_{0..3}.np"
echo "========================================"
