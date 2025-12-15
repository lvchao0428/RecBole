#!/usr/bin/env bash
#set -euo pipefail

# 生成Amazon_Beauty的生成式文本特征
# 使用LLM先生成回答，再提取embedding

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==========================================
# 性能优化环境变量
# ==========================================
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

# 禁用TensorFlow警告输出
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

echo "========================================"
echo "Amazon_Beauty 生成式文本特征生成"
echo "========================================"
echo "CPU cores: $(nproc)"
echo "OMP threads: $OMP_NUM_THREADS"
echo ""
echo "⚠️  生成式模式说明："
echo "   - 先让LLM回答prompt问题，再对回答提取embedding"
echo "   - 使用temperature=0保证确定性（每次运行结果一致）"
echo "   - 速度较慢（需要autoregressive生成）"
echo ""

# ==========================================
# 1. 导出item mapping（如果不存在）
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
    echo "[1/3] Mapping文件已存在: $MAPPING_FILE (跳过)"
fi
echo ""

# ==========================================
# 2. Qwen3生成式单视图特征
# ==========================================
echo "[2/3] Generating Qwen3 generative single-view embeddings..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "配置: temperature=0, max_new_tokens=128, prompt=base"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.generative.base.npy \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 8 \
  --max_length 512 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template \
  --generative \
  --gen_max_new_tokens 128 \
  --gen_temperature 0.0

echo ""
echo "✅ Qwen3生成式单视图特征生成完成"
echo "   - 特征文件: dataset/Amazon_Beauty/item_text_emb.qwen3.generative.base.npy"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 3. Qwen3生成式多视图特征（4视图）
# ==========================================
echo "[3/3] Generating Qwen3 generative multi-view embeddings (4 views)..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "配置: temperature=0, max_new_tokens=128, prompt=multiview-universal"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.generative.multiview.npy \
  --prompt_preset multiview-universal \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Beauty/qwen3_generative_4views \
  --view_project_dim 64 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 8 \
  --max_length 512 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template \
  --generative \
  --gen_max_new_tokens 128 \
  --gen_temperature 0.0

echo ""
echo "✅ Qwen3生成式多视图特征生成完成"
echo "   - 拼接特征: dataset/Amazon_Beauty/item_text_emb.qwen3.generative.multiview.npy (shape: [N, 256])"
echo "   - 视图0 (WHAT):       dataset/Amazon_Beauty/qwen3_generative_4views/view_0.npy"
echo "   - 视图1 (WHO):        dataset/Amazon_Beauty/qwen3_generative_4views/view_1.npy"
echo "   - 视图2 (WHEN/WHERE): dataset/Amazon_Beauty/qwen3_generative_4views/view_2.npy"
echo "   - 视图3 (HOW):        dataset/Amazon_Beauty/qwen3_generative_4views/view_3.npy"
echo "   - 元数据: dataset/Amazon_Beauty/qwen3_generative_4views/views.json"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "========================================"
echo "✅ 生成式特征全部生成完成！"
echo "========================================"
echo ""
echo "生成的特征文件："
echo "  [Qwen3生成式单视图]"
echo "    - dataset/Amazon_Beauty/item_text_emb.qwen3.generative.base.npy"
echo ""
echo "  [Qwen3生成式多视图]"
echo "    - dataset/Amazon_Beauty/item_text_emb.qwen3.generative.multiview.npy"
echo "    - dataset/Amazon_Beauty/qwen3_generative_4views/ (分视图)"
echo ""
echo "对比实验建议："
echo "  - 非生成式 vs 生成式："
echo "    item_text_emb.qwen3.base.npy vs item_text_emb.qwen3.generative.base.npy"
echo "    item_text_emb.qwen3.multiview.npy vs item_text_emb.qwen3.generative.multiview.npy"
echo ""
echo "关键参数说明："
echo "  --generative           : 启用生成式模式"
echo "  --gen_temperature 0.0  : 确定性生成（greedy decoding）"
echo "  --gen_max_new_tokens   : 生成的最大token数"
echo ""

