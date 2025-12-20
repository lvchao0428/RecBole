#!/usr/bin/env bash
# set -euo pipefail

# ==============================================
# 使用 Qwen2.5-72B-Instruct-GPTQ-Int8 生成文本特征
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

# CUDA 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "========================================"
echo "Qwen2.5-72B 文本特征生成 (GPTQ-Int8)"
echo "========================================"
echo "项目路径: $PROJECT_ROOT"
echo "模型路径: $MODEL_PATH"
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
echo "[2/3] Generating Qwen2.5-72B single-view embeddings..."
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
  --batch_size 4 \
  --max_length 0 \
  --dtype float16 \
  --device_map auto \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "✅ Qwen2.5-72B 单视图特征生成完成"
echo "   - 特征文件: dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.base.npy"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 3. Qwen2.5-72B 多视图特征（4个视图）
# ==========================================
echo "[3/3] Generating Qwen2.5-72B multi-view embeddings (4 views)..."
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
  --batch_size 4 \
  --max_length 0 \
  --dtype float16 \
  --device_map auto \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "✅ Qwen2.5-72B 多视图特征生成完成"
echo "   - 拼接特征: dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.multiview.npy (shape: [N, 256])"
echo "   - 视图0 (Identity):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_0.npy"
echo "   - 视图1 (Function):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_1.npy"
echo "   - 视图2 (Audience):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_2.npy"
echo "   - 视图3 (Category):  dataset/Amazon_Beauty/qwen2.5_72b_4views/view_3.npy"
echo "   - 元数据: dataset/Amazon_Beauty/qwen2.5_72b_4views/views.json"
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
echo "  1. 验证特征维度: python -c \"import numpy as np; x=np.load('dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.base.npy'); print(f'shape={x.shape}, dtype={x.dtype}')\""
echo "  2. 运行实验使用生成的特征"
echo ""

