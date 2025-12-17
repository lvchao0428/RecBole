#!/usr/bin/env bash
#set -euo pipefail

# 生成Amazon_Beauty的生成式文本特征（仅4视图）
# 使用LLM先生成回答，再提取embedding

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# PyTorch 2.x uses PYTORCH_ALLOC_CONF instead of PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="expandable_segments:True"

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
echo "Amazon_Beauty 生成式文本特征生成（4视图）"
echo "========================================"
echo "CPU cores: $(nproc)"
echo "OMP threads: $OMP_NUM_THREADS"
echo ""
echo "⚠️  生成式模式说明："
echo "   - 先让LLM回答prompt问题，再对回答提取embedding"
echo "   - 使用temperature=0保证确定性（每次运行结果一致）"
echo ""
echo "🚀 性能优化已启用（32GB显存最大化配置）："
echo "   - Qwen 2.5 模型（支持批量生成）"
echo "   - bfloat16 精度（无量化，最佳质量）"
echo "   - batch_size=128（最大化吞吐量）"
echo "   - Flash Attention 2"
echo "   - 预计显存占用：~28-30GB"
echo ""

# ==========================================
# 1. 导出item mapping（如果不存在）
# ==========================================
MAPPING_FILE="dataset/Amazon_Beauty/item_index_mapping.csv"
if [ ! -f "$MAPPING_FILE" ]; then
    echo "[1/2] Exporting item index mapping..."
    python tools/export_internal_item_mapping.py \
      --dataset Amazon_Beauty \
      --config sasrec_base_plain.yaml \
      --output "$MAPPING_FILE"
    echo "✅ Mapping文件生成完成: $MAPPING_FILE"
else
    echo "[1/2] Mapping文件已存在: $MAPPING_FILE (跳过)"
fi
echo ""

# ==========================================
# 2. Qwen3生成式多视图特征（4视图）
# ==========================================
echo "[2/2] Generating Qwen3 generative multi-view embeddings (4 views)..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "配置: temperature=0, max_new_tokens=128, prompt=multiview-universal"
echo ""

# ==========================================
# 性能优化参数说明：
#   --batch_size 4       : 减小批量大小节省显存
#   --gen_max_new_tokens : 减少可显著加速（64 vs 128）
#   --load_in_4bit       : INT4量化，显存减少~75%（需要bitsandbytes）
#   --load_in_8bit       : INT8量化，显存减少~50%（需要bitsandbytes）
#   --flash_attn         : 启用Flash Attention 2（需要安装flash-attn）
# ==========================================

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Qwen2.5-7B-Instruct \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.generative.multiview.npy \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Beauty/qwen3_generative_4views_multiview \
  --view_project_dim 64 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 64 \
  --max_length 512 \
  --dtype bfloat16 \
  --svd_random_state 42 \
  --use_chat_template \
  --generative \
  --gen_max_new_tokens 64 \
  --gen_temperature 0.0 \
  --flash_attn \
  --center \
  --whiten \
  --save_generated_texts dataset/Amazon_Beauty/qwen3_generative_4views_multiview/generated_texts.json

echo ""
echo "✅ Qwen3生成式多视图特征生成完成"
echo "   - 拼接特征: dataset/Amazon_Beauty/item_text_emb.qwen3.generative.multiview.npy (shape: [N, 256])"
echo "   - 视图0 (WHAT):       dataset/Amazon_Beauty/qwen3_generative_4views/view_0.npy"
echo "   - 视图1 (WHO):        dataset/Amazon_Beauty/qwen3_generative_4views/view_1.npy"
echo "   - 视图2 (WHEN/WHERE): dataset/Amazon_Beauty/qwen3_generative_4views/view_2.npy"
echo "   - 视图3 (HOW):        dataset/Amazon_Beauty/qwen3_generative_4views/view_3.npy"
echo "   - 元数据: dataset/Amazon_Beauty/qwen3_generative_4views/views.json"
echo "   - 生成文本: dataset/Amazon_Beauty/qwen3_generative_4views/generated_texts.json"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "========================================"
echo "✅ 生成式特征全部生成完成！"
echo "========================================"
echo ""
echo "生成的特征文件："
echo "  [Qwen3生成式多视图]"
echo "    - dataset/Amazon_Beauty/item_text_emb.qwen3.generative.multiview.npy"
echo "    - dataset/Amazon_Beauty/qwen3_generative_4views/ (分视图)"
echo "    - dataset/Amazon_Beauty/qwen3_generative_4views/generated_texts.json (生成的中间文本)"
echo ""
echo "关键参数说明："
echo "  --generative             : 启用生成式模式"
echo "  --gen_temperature 0.0    : 确定性生成（greedy decoding）"
echo "  --gen_max_new_tokens     : 生成的最大token数"
echo "  --save_generated_texts   : 保存LLM生成的中间文本到JSON文件"
echo ""
echo "🚀 显存优化技巧："
echo "  --load_in_4bit      : INT4量化，显存减少~75%"
echo "  --load_in_8bit      : INT8量化，显存减少~50%"
echo "  --batch_size N      : 减小N以节省显存"
echo "  --gen_max_new_tokens: 减少以节省显存和加速"
echo ""
echo "🚀 安装依赖："
echo "  pip install bitsandbytes  # 量化支持"
echo "  pip install flash-attn --no-build-isolation  # Flash Attention（可选）"
echo ""
