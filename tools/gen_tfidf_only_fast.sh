#!/usr/bin/env bash
#set -euo pipefail

# 仅生成TF-IDF基线特征（优化版：多核加速）

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ==========================================
# 性能优化环境变量
# ==========================================
# 启用NumPy/OpenBLAS多线程
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

# 禁用TensorFlow警告
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

echo "========================================"
echo "生成TF-IDF基线特征（优化版）"
echo "========================================"
echo "CPU cores: $(nproc)"
echo "OMP threads: $OMP_NUM_THREADS"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --title_field title \
  --svd_dim 256 \
  --svd_random_state 42 \
  --ngram_min 1 \
  --ngram_max 2 \
  --min_df 2 \
  --dtype float16

echo ""
echo "========================================"
echo "✅ 生成完成！"
echo "========================================"
echo "   - 特征: dataset/Amazon_Beauty/item_text_emb.base.npy"
echo "   - 统计: dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

