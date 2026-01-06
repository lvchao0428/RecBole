#!/usr/bin/env bash
# set -euo pipefail

# ==============================================
# 使用 Qwen2.5-14B-Instruct 生成 Toys 文本特征
# 非量化版本，使用 HuggingFace Transformers
# 支持多GPU推理 (device_map=auto)
# [推荐配置] 仅启用 center，不使用 whiten
# ==============================================

# 项目根目录（根据实际情况修改）
PROJECT_ROOT="/home/ubuntu/own/RecBole"
cd "$PROJECT_ROOT"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==========================================
# 数据集配置
# ==========================================
DATASET_NAME="Amazon_Toys_and_Games"

# ==========================================
# 模型路径配置
# ==========================================
MODEL_PATH="/data/model/Qwen2.5-14B-Instruct"
MODEL_NAME="qwen2.5_14b"

# ==========================================
# GPU 配置 - 使用多张GPU (根据实际情况修改)
# ==========================================
export CUDA_VISIBLE_DEVICES=2,3,4,5,6
#export CUDA_VISIBLE_DEVICES=6

# ==========================================
# 性能优化环境变量
# ==========================================
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

# 禁用TensorFlow警告
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# HuggingFace 缓存
export HF_HUB_CACHE="/tmp/hf_hub_cache"

echo "========================================"
echo "Qwen2.5-14B-Instruct 文本特征生成 (多GPU)"
echo "[推荐配置] Center Only (无Whiten)"
echo "========================================"
echo "项目路径: $PROJECT_ROOT"
echo "数据集: $DATASET_NAME"
echo "模型路径: $MODEL_PATH"
echo "CPU cores: $(nproc)"
echo "GPU count: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# ==========================================
# 1. TF-IDF基线特征 (如果已存在则跳过)
# ==========================================
TFIDF_FILE="dataset/${DATASET_NAME}/item_text_emb.base.center_only.npy"
if [ ! -f "$TFIDF_FILE" ]; then
    echo "[1/4] Generating TF-IDF (base) embeddings (center only)..."
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    python tools/build_item_text_emb_base.py \
      --dataset ${DATASET_NAME} \
      --config sasrec_base_plain.yaml \
      --output dataset/${DATASET_NAME}/item_text_emb.base.center_only.npy \
      --title_field title \
      --svd_dim 256 \
      --svd_random_state 42 \
      --ngram_min 1 \
      --ngram_max 2 \
      --min_df 2 \
      --dtype float16 \
      --center

    echo ""
    echo "✅ TF-IDF特征生成完成 (center only)"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "[1/4] TF-IDF特征已存在: $TFIDF_FILE (跳过)"
fi
echo ""

# ==========================================
# 2. 导出item mapping
# ==========================================
MAPPING_FILE="dataset/${DATASET_NAME}/item_index_mapping.csv"
if [ ! -f "$MAPPING_FILE" ]; then
    echo "[2/4] Exporting item index mapping for LLM embedding..."
    python tools/export_internal_item_mapping.py \
      --dataset ${DATASET_NAME} \
      --config sasrec_base_plain.yaml \
      --output "$MAPPING_FILE"
    echo "✅ Mapping文件生成完成: $MAPPING_FILE"
else
    echo "[2/4] Mapping文件已存在: $MAPPING_FILE"
fi
echo ""

# ==========================================
# 3. Qwen2.5-14B 单视图特征 (多GPU, center only)
# ==========================================
echo "[3/4] Generating ${MODEL_NAME} single-view embeddings (Multi-GPU, center only)..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output "dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.base.center_only.npy" \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset ${DATASET_NAME} \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 24 \
  --max_length 0 \
  --dtype float16 \
  --device_map auto \
  --svd_random_state 42 \
  --use_chat_template \
  --center

echo ""
echo "✅ ${MODEL_NAME} 单视图特征生成完成 (center only)"
echo "   - 特征文件: dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.base.center_only.npy"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 4. Qwen2.5-14B 多视图特征（4个视图, center only）
# ==========================================
echo "[4/4] Generating ${MODEL_NAME} multi-view embeddings (4 views, Multi-GPU, center only)..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output "dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.multiview.center_only.npy" \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir "dataset/${DATASET_NAME}/${MODEL_NAME}_4views_center_only" \
  --view_project_dim 64 \
  --dataset ${DATASET_NAME} \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 24 \
  --max_length 0 \
  --dtype float16 \
  --device_map auto \
  --svd_random_state 42 \
  --use_chat_template \
  --center

echo ""
echo "✅ ${MODEL_NAME} 多视图特征生成完成 (center only)"
echo "   - 拼接特征: dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.multiview.center_only.npy (shape: [N, 256])"
echo "   - 视图0 (Identity):  dataset/${DATASET_NAME}/${MODEL_NAME}_4views_center_only/view_0.npy"
echo "   - 视图1 (Function):  dataset/${DATASET_NAME}/${MODEL_NAME}_4views_center_only/view_1.npy"
echo "   - 视图2 (Audience):  dataset/${DATASET_NAME}/${MODEL_NAME}_4views_center_only/view_2.npy"
echo "   - 视图3 (Category):  dataset/${DATASET_NAME}/${MODEL_NAME}_4views_center_only/view_3.npy"
echo "   - 元数据: dataset/${DATASET_NAME}/${MODEL_NAME}_4views_center_only/views.json"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "========================================"
echo "✅ 全部生成完成！[推荐配置: Center Only]"
echo "========================================"
echo ""
echo "生成的特征文件："
echo "  [TF-IDF基线]"
echo "    - dataset/${DATASET_NAME}/item_text_emb.base.center_only.npy"
echo ""
echo "  [${MODEL_NAME}单视图]"
echo "    - dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.base.center_only.npy"
echo ""
echo "  [${MODEL_NAME}多视图]"
echo "    - dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.multiview.center_only.npy"
echo "    - dataset/${DATASET_NAME}/${MODEL_NAME}_4views_center_only/ (分视图)"
echo ""
echo "下一步："
echo "  1. 验证特征维度:"
echo "     python -c \"import numpy as np; x=np.load('dataset/${DATASET_NAME}/item_text_emb.${MODEL_NAME}.base.center_only.npy'); print(f'shape={x.shape}, dtype={x.dtype}')\""
echo "  2. 运行实验 (使用center_only版yaml配置)"
echo ""


