#!/usr/bin/env bash
# set -euo pipefail

# ==============================================
# 使用 vLLM + Qwen2.5-72B-Instruct-GPTQ-Int8 生成文本特征
# 适用于 8x4090 服务器 (tensor parallel)
# ==============================================

# 项目根目录（根据实际情况修改）
PROJECT_ROOT="/home/charlie/project/RecBole"
cd "$PROJECT_ROOT"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==========================================
# 模型路径配置（根据实际情况修改）
# ==========================================
MODEL_PATH="/data/model/Qwen2.5-72B-Instruct-GPTQ-Int8"

# Embedding 模型（用于从生成文本中提取 embedding）
# 可选: BAAI/bge-large-en-v1.5, BAAI/bge-large-zh-v1.5, etc.
EMBED_MODEL="BAAI/bge-large-en-v1.5"

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

# vLLM 缓存目录
export VLLM_CACHE_DIR="/tmp/vllm_cache"
export HF_HUB_CACHE="/tmp/hf_hub_cache"

# CUDA 配置 - 使用所有8张GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "========================================"
echo "vLLM + Qwen2.5-72B 文本特征生成 (GPTQ-Int8)"
echo "========================================"
echo "项目路径: $PROJECT_ROOT"
echo "模型路径: $MODEL_PATH"
echo "Embedding模型: $EMBED_MODEL"
echo "CPU cores: $(nproc)"
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo ""

# ==========================================
# 1. 导出 item mapping (如果不存在)
# ==========================================
MAPPING_FILE="dataset/Amazon_Beauty/item_index_mapping.csv"
if [ ! -f "$MAPPING_FILE" ]; then
    echo "[1/3] Exporting item index mapping..."
    python tools/export_internal_item_mapping.py \
      --dataset Amazon_Beauty \
      --config sasrec_base_plain.yaml \
      --output "$MAPPING_FILE"
    echo "✅ Mapping文件生成完成: $MAPPING_FILE"
else
    echo "[1/3] Mapping文件已存在: $MAPPING_FILE"
fi
echo ""

# ==========================================
# 2. Qwen2.5-72B 单视图特征
# ==========================================
echo "[2/3] Generating Qwen2.5-72B single-view embeddings with vLLM..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen2.5_72b.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.base.npy \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --tensor_parallel_size 8 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 4096 \
  --batch_size 32 \
  --gen_max_new_tokens 128 \
  --gen_temperature 0.7 \
  --gen_top_p 0.95 \
  --dtype float16 \
  --enforce_eager \
  --svd_random_state 42 \
  --use_chat_template \
  --embed_model_name_or_path "$EMBED_MODEL"

echo ""
echo "✅ Qwen2.5-72B 单视图特征生成完成"
echo "   - 特征文件: dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.base.npy"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 3. Qwen2.5-72B 多视图特征（4个视图）
# ==========================================
echo "[3/3] Generating Qwen2.5-72B multi-view embeddings (4 views) with vLLM..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen2.5_72b.py \
  --mapping "$MAPPING_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --output dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.multiview.npy \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Beauty/qwen2.5_72b_4views \
  --view_project_dim 64 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --tensor_parallel_size 8 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 4096 \
  --batch_size 32 \
  --gen_max_new_tokens 128 \
  --gen_temperature 0.7 \
  --gen_top_p 0.95 \
  --dtype float16 \
  --enforce_eager \
  --svd_random_state 42 \
  --use_chat_template \
  --embed_model_name_or_path "$EMBED_MODEL" \
  --save_generated_texts dataset/Amazon_Beauty/qwen2.5_72b_generated_texts.json

echo ""
echo "✅ Qwen2.5-72B 多视图特征生成完成"
echo "   - 拼接特征: dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.multiview.npy (shape: [N, 256])"
echo "   - 视图0 (Identity):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_0.npy"
echo "   - 视图1 (Function):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_1.npy"
echo "   - 视图2 (Audience):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_2.npy"
echo "   - 视图3 (Category):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_3.npy"
echo "   - 元数据: dataset/Amazon_Beauty/qwen2.5_72b_4views/views.json"
echo "   - 生成文本: dataset/Amazon_Beauty/qwen2.5_72b_generated_texts.json"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "========================================"
echo "✅ 全部生成完成！"
echo "========================================"
echo ""
echo "生成的特征文件："
echo "  [Qwen2.5-72B 单视图]"
echo "    - dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.base.npy"
echo ""
echo "  [Qwen2.5-72B 多视图]"
echo "    - dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.multiview.npy"
echo "    - dataset/Amazon_Beauty/qwen2.5_72b_4views/ (分视图)"
echo ""
echo "下一步："
echo "  1. 验证特征维度:"
echo "     python -c \"import numpy as np; x=np.load('dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.base.npy'); print(f'shape={x.shape}, dtype={x.dtype}')\""
echo "  2. 查看生成文本示例:"
echo "     python -c \"import json; d=json.load(open('dataset/Amazon_Beauty/qwen2.5_72b_generated_texts.json')); print(json.dumps(d['items'][1], indent=2, ensure_ascii=False))\""
echo ""
