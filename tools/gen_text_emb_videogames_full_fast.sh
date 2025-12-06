#!/usr/bin/env bash
#set -euo pipefail

# 生成Amazon_Video_Games的所有文本特征（优化版：多核加速）
# 对齐Amazon_Beauty的center+whiten格式

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==========================================
# 性能优化环境变量
# ==========================================
# 启用NumPy/OpenBLAS多线程（根据CPU核心数调整）
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

# 禁用TensorFlow警告输出（加快启动速度）
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

echo "========================================"
echo "Amazon_Video_Games 文本特征生成（优化版）"
echo "========================================"
echo "CPU cores: $(nproc)"
echo "OMP threads: $OMP_NUM_THREADS"
echo ""

# ==========================================
# 1. TF-IDF基线特征
# ==========================================
echo "[1/4] Generating TF-IDF (base) embeddings with center+whiten..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_base.py \
  --dataset Amazon_Video_Games \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Video_Games/item_text_emb.base.npy \
  --title_field title \
  --svd_dim 256 \
  --svd_random_state 42 \
  --ngram_min 1 \
  --ngram_max 2 \
  --min_df 2 \
  --dtype float16

echo ""
echo "✅ TF-IDF特征生成完成"
echo "   - 特征文件: dataset/Amazon_Video_Games/item_text_emb.base.npy"
echo "   - 统计文件: dataset/Amazon_Video_Games/item_text_emb.base_whiten_stats.npz"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 2. 导出item mapping
# ==========================================
echo "[2/4] Exporting item index mapping for LLM embedding..."
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Video_Games \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Video_Games/item_index_mapping.csv

echo "✅ Mapping文件生成完成"
echo "   - dataset/Amazon_Video_Games/item_index_mapping.csv"
echo ""

# ==========================================
# 3. Qwen3单视图特征（用于对比）
# ==========================================
echo "[3/4] Generating Qwen3 single-view embeddings with center+whiten..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Video_Games/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Video_Games/item_text_emb.qwen3.base.npy \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset Amazon_Video_Games \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "✅ Qwen3单视图特征生成完成"
echo "   - 特征文件: dataset/Amazon_Video_Games/item_text_emb.qwen3.base.npy"
echo "   - 统计文件: dataset/Amazon_Video_Games/item_text_emb.qwen3.base_whiten_stats.npz"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 4. Qwen3多视图特征（4个视图）
# ==========================================
echo "[4/4] Generating Qwen3 multi-view embeddings (4 views) with per-view projection..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Video_Games/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Video_Games/item_text_emb.qwen3.multiview.npy \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Video_Games/qwen3_4views \
  --view_project_dim 64 \
  --dataset Amazon_Video_Games \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "✅ Qwen3多视图特征生成完成"
echo "   - 拼接特征: dataset/Amazon_Video_Games/item_text_emb.qwen3.multiview.npy (shape: [N, 256])"
echo "   - 统计文件: dataset/Amazon_Video_Games/item_text_emb.qwen3.multiview_whiten_stats.npz"
echo "   - 视图0 (Identity):  dataset/Amazon_Video_Games/qwen3_4views/view_0.npy"
echo "   - 视图1 (Function):  dataset/Amazon_Video_Games/qwen3_4views/view_1.npy"
echo "   - 视图2 (Audience):  dataset/Amazon_Video_Games/qwen3_4views/view_2.npy"
echo "   - 视图3 (Category):  dataset/Amazon_Video_Games/qwen3_4views/view_3.npy"
echo "   - 元数据: dataset/Amazon_Video_Games/qwen3_4views/views.json"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "========================================"
echo "✅ 全部生成完成！"
echo "========================================"
echo ""
echo "生成的特征文件："
echo "  [TF-IDF基线]"
echo "    - dataset/Amazon_Video_Games/item_text_emb.base.npy"
echo ""
echo "  [Qwen3单视图]"
echo "    - dataset/Amazon_Video_Games/item_text_emb.qwen3.base.npy"
echo ""
echo "  [Qwen3多视图]"
echo "    - dataset/Amazon_Video_Games/item_text_emb.qwen3.multiview.npy"
echo "    - dataset/Amazon_Video_Games/qwen3_4views/ (分视图)"
echo ""
echo "下一步："
echo "  1. 验证特征: bash tools/verify_all_embeddings.sh"
echo "  2. 运行TF-IDF实验: bash two_phase_run_tfidf_videogames.sh"
echo "  3. 运行多视图实验: bash two_phase_run_multiview_split_videogames.sh"
echo ""

