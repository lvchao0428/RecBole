#!/usr/bin/env bash
#set -euo pipefail

# 一键生成所有数据集的文本特征（对齐 Beauty 格式）
# 包括：Beauty, Toys, VideoGames, Yelp

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "========================================"
echo "批量生成所有数据集的文本特征"
echo "========================================"
echo "数据集列表："
echo "  1. Amazon_Beauty"
echo "  2. Amazon_Toys_and_Games"
echo "  3. Amazon_Video_Games"
echo "  4. Yelp"
echo ""
echo "每个数据集将生成："
echo "  - TF-IDF 特征（256维）"
echo "  - Qwen3 单视图特征（256维）"
echo "  - Qwen3 多视图特征（4×64=256维）"
echo "  - Center+Whiten 统计文件"
echo ""
echo "总开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# ==========================================
# 1. Amazon Beauty
# ==========================================
echo ""
echo "████████████████████████████████████████"
echo "█  [1/4] Amazon Beauty                 █"
echo "████████████████████████████████████████"
echo ""

if [ -f "tools/gen_text_emb_beauty_full_fast.sh" ]; then
    bash tools/gen_text_emb_beauty_full_fast.sh
    BEAUTY_STATUS="✅ 完成"
else
    echo "⚠️  脚本不存在: tools/gen_text_emb_beauty_full_fast.sh"
    BEAUTY_STATUS="⚠️  跳过"
fi

# ==========================================
# 2. Amazon Toys and Games
# ==========================================
echo ""
echo "████████████████████████████████████████"
echo "█  [2/4] Amazon Toys and Games         █"
echo "████████████████████████████████████████"
echo ""

if [ -f "tools/gen_text_emb_toys_full_fast.sh" ]; then
    bash tools/gen_text_emb_toys_full_fast.sh
    TOYS_STATUS="✅ 完成"
else
    echo "⚠️  脚本不存在: tools/gen_text_emb_toys_full_fast.sh"
    TOYS_STATUS="⚠️  跳过"
fi

# ==========================================
# 3. Amazon Video Games
# ==========================================
echo ""
echo "████████████████████████████████████████"
echo "█  [3/4] Amazon Video Games            █"
echo "████████████████████████████████████████"
echo ""

if [ -f "tools/gen_text_emb_videogames_full_fast.sh" ]; then
    bash tools/gen_text_emb_videogames_full_fast.sh
    VIDEOGAMES_STATUS="✅ 完成"
else
    echo "⚠️  脚本不存在: tools/gen_text_emb_videogames_full_fast.sh"
    VIDEOGAMES_STATUS="⚠️  跳过"
fi

# ==========================================
# 4. Yelp
# ==========================================
echo ""
echo "████████████████████████████████████████"
echo "█  [4/4] Yelp                          █"
echo "████████████████████████████████████████"
echo ""

if [ -f "tools/gen_text_emb_yelp_full_fast.sh" ]; then
    bash tools/gen_text_emb_yelp_full_fast.sh
    YELP_STATUS="✅ 完成"
else
    echo "⚠️  脚本不存在: tools/gen_text_emb_yelp_full_fast.sh"
    YELP_STATUS="⚠️  跳过"
fi

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# ==========================================
# 总结
# ==========================================
echo ""
echo "========================================"
echo "✅ 批量生成完成！"
echo "========================================"
echo ""
echo "执行状态："
echo "  1. Amazon Beauty:          $BEAUTY_STATUS"
echo "  2. Amazon Toys and Games:  $TOYS_STATUS"
echo "  3. Amazon Video Games:     $VIDEOGAMES_STATUS"
echo "  4. Yelp:                   $YELP_STATUS"
echo ""
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "生成的文件位置："
echo "  Beauty:      dataset/Amazon_Beauty/"
echo "  Toys:        dataset/Amazon_Toys_and_Games/"
echo "  VideoGames:  dataset/Amazon_Video_Games/"
echo "  Yelp:        dataset/yelp/"
echo ""
echo "下一步："
echo "  1. 验证特征维度："
echo "     python tools/verify_all_embeddings.py"
echo ""
echo "  2. 运行实验："
echo "     - TF-IDF:    bash two_phase_run_tfidf_*.sh"
echo "     - 多视图:    bash two_phase_run_multiview_split_*.sh"
echo ""
echo "========================================"

