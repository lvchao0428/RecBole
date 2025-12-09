#!/usr/bin/env bash
#set -euo pipefail

# 重新生成 Amazon_Beauty 的所有 Qwen3 特征（修复白化问题后）
# 包括：单视图 + Multi-view（4个视图）

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========================================"
echo "重新生成所有 Qwen3 特征（修复版）"
echo "========================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "说明: 修复了白化后 L2 归一化的问题"
echo "修复前: 对角线均值 ≈ 1/D (错误)"
echo "修复后: 对角线均值 ≈ 1.0 (正确)"
echo ""

# ==========================================
# 1. 备份旧文件
# ==========================================
echo "========================================"
echo "备份旧文件"
echo "========================================"

backup_time=$(date +%s)

if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
       dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy.backup_$backup_time
    echo "✅ 备份 item_text_emb.qwen3.base.npy"
fi

if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz \
       dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz.backup_$backup_time
    echo "✅ 备份 item_text_emb.qwen3.base_whiten_stats.npz"
fi

if [ -d dataset/Amazon_Beauty/qwen3_4views ]; then
    backup_dir="dataset/Amazon_Beauty/qwen3_4views.backup_$backup_time"
    mv dataset/Amazon_Beauty/qwen3_4views "$backup_dir"
    echo "✅ 备份 qwen3_4views/ 到 $backup_dir"
fi

if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
       dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy.backup_$backup_time
    echo "✅ 备份 item_text_emb.qwen3.multiview.npy"
fi

if [ -f dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz ]; then
    mv dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz \
       dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz.backup_$backup_time
    echo "✅ 备份 item_text_emb.qwen3.multiview_whiten_stats.npz"
fi

echo ""

# ==========================================
# 2. 生成 Qwen3 单视图特征
# ==========================================
echo "========================================"
echo "[1/2] 生成 Qwen3 单视图特征（256维）"
echo "========================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "✅ 单视图特征生成完成"
echo "   - 特征文件: dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy"
echo "   - 统计文件: dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 3. 生成 Qwen3 Multi-view 特征
# ==========================================
echo "========================================"
echo "[2/2] 生成 Qwen3 Multi-view 特征（4视图×64维）"
echo "========================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Beauty/qwen3_4views \
  --view_project_dim 64 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --device cuda:0 \
  --svd_random_state 42 \
  --use_chat_template

echo ""
echo "✅ Multi-view 特征生成完成"
echo "   - 拼接特征: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy (256维)"
echo "   - 统计文件: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz"
echo "   - 视图0 (Identity):  dataset/Amazon_Beauty/qwen3_4views/view_0.npy (64维)"
echo "   - 视图1 (Function):  dataset/Amazon_Beauty/qwen3_4views/view_1.npy (64维)"
echo "   - 视图2 (Audience):  dataset/Amazon_Beauty/qwen3_4views/view_2.npy (64维)"
echo "   - 视图3 (Category):  dataset/Amazon_Beauty/qwen3_4views/view_3.npy (64维)"
echo "   - 元数据: dataset/Amazon_Beauty/qwen3_4views/views.json"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 4. 验证白化效果
# ==========================================
echo "========================================"
echo "验证白化效果"
echo "========================================"
echo ""

echo "----------------------------------------"
echo "验证单视图特征 (256维)"
echo "----------------------------------------"
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --dataset Amazon_Beauty

echo ""
echo "----------------------------------------"
echo "验证 Multi-view 各视图 (64维)"
echo "----------------------------------------"
for i in 0 1 2 3; do
    echo ""
    echo "=== 视图 $i ==="
    python tools/verify_whiten.py \
      dataset/Amazon_Beauty/qwen3_4views/view_${i}.npy \
      --dataset Amazon_Beauty
done

echo ""
echo "----------------------------------------"
echo "验证 Multi-view 拼接特征 (256维)"
echo "----------------------------------------"
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
  --dataset Amazon_Beauty

echo ""
echo "========================================"
echo "✅ 全部完成！"
echo "========================================"
echo "总耗时: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "期望的验证结果："
echo "  ✅ Embedding未L2归一化 (这是正常的)"
echo "  ✅ Embedding已中心化"
echo "  ✅ 对角线均值 ≈ 0.95~1.05 (期望≈1.0)"
echo "  ✅ 白化效果优秀/良好"
echo ""

