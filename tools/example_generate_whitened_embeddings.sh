#!/usr/bin/env bash
# 生成带 center + whiten 预处理的文本embedding示例脚本

set -e

DATASET="Amazon_Beauty"
OUTPUT_DIR="dataset/${DATASET}"

echo "=================================================="
echo "生成 Center + Whiten 预处理的文本Embedding"
echo "Dataset: ${DATASET}"
echo "=================================================="
echo ""

# 1. 生成 TF-IDF + SVD (Base) 特征
echo "[1/2] 生成 TF-IDF + SVD 特征..."
python tools/build_item_text_emb_base.py \
  --dataset ${DATASET} \
  --output ${OUTPUT_DIR}/item_text_emb.base.npy \
  --svd_dim 256 \
  --title_field title \
  --dtype float16
  # 默认启用 center + whiten
  # 若要关闭，添加 --no_whiten

echo ""
echo "✅ Base特征生成完成"
echo "   - Embedding: ${OUTPUT_DIR}/item_text_emb.base.npy"
echo "   - 统计量: ${OUTPUT_DIR}/item_text_emb.base_whiten_stats.npz"
echo ""

# 验证Base特征
echo "验证 Base 特征..."
python tools/verify_whiten.py ${OUTPUT_DIR}/item_text_emb.base.npy
echo ""

# 2. 生成 Qwen3 (LLM) 特征
echo "[2/2] 生成 Qwen3 LLM 特征..."
echo "⚠️  注意：需要先运行 export_internal_item_mapping.py 生成 mapping CSV"
echo ""

# 检查是否存在 mapping 文件
MAPPING_FILE="${OUTPUT_DIR}/item_index_mapping.csv"
if [ ! -f "${MAPPING_FILE}" ]; then
  echo "❌ 未找到 mapping 文件: ${MAPPING_FILE}"
  echo "   请先运行："
  echo "   python tools/export_internal_item_mapping.py --dataset ${DATASET}"
  echo ""
  echo "跳过 Qwen3 特征生成"
else
  # 注释掉Qwen3生成（因为需要大模型，仅作示例）
  echo "# python tools/build_item_text_emb_qwen3_hf.py \\"
  echo "#   --mapping ${MAPPING_FILE} \\"
  echo "#   --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\"
  echo "#   --output ${OUTPUT_DIR}/item_text_emb.qwen3.npy \\"
  echo "#   --dataset ${DATASET} \\"
  echo "#   --output_mode mean \\"
  echo "#   --batch_size 16 \\"
  echo "#   --dtype float16"
  echo "#   # 默认启用 center + whiten"
  echo "#   # 若要关闭，添加 --no_whiten"
  echo ""
  echo "⚠️  Qwen3 生成已注释（需要大模型）"
fi

echo ""
echo "=================================================="
echo "✅ 全部完成"
echo "=================================================="
echo ""
echo "使用方法："
echo "1. 在模型配置YAML中设置："
echo "   item_text_emb_path_base: ${OUTPUT_DIR}/item_text_emb.base.npy"
echo "   item_text_emb_path_llm: ${OUTPUT_DIR}/item_text_emb.qwen3.npy"
echo ""
echo "2. 运行训练："
echo "   python two_phase_run_tfidf_llm.sh"
echo ""

