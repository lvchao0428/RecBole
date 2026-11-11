#!/bin/bash
# 清理 5090 saved/ 目录 — 只保留 WSDM V4 相关 checkpoint
# 执行前请确认保留列表正确！
#
# 保留 (9 个, ~19.4G):
#   ps_beauty_llm_cross_noboost_seed2025
#   ps_beauty_llm_nc_noboost_seed2025
#   ps_beauty_mv_cross_noboost_seed2025
#   ps_beauty_mv_nc_noboost_seed2025
#   ps_beauty_tf_cross_noboost_seed2025
#   ps_beauty_tf_nc_noboost_seed2025
#   ts_beauty_mv_cross_tfviews_seed2025
#   ts_beauty_mv_nc_boost_tfviews_seed2025
#   ts_beauty_mv_nc_tfviews_seed2025
#
# 删除: 其余全部 (~215 个, ~1080G)

set -euo pipefail
SAVED_DIR="/home/charlie/project/RecBole/saved"
cd "$SAVED_DIR"

KEEP=(
    "ps_beauty_llm_cross_noboost_seed2025"
    "ps_beauty_llm_nc_noboost_seed2025"
    "ps_beauty_mv_cross_noboost_seed2025"
    "ps_beauty_mv_nc_noboost_seed2025"
    "ps_beauty_tf_cross_noboost_seed2025"
    "ps_beauty_tf_nc_noboost_seed2025"
    "ts_beauty_mv_cross_tfviews_seed2025"
    "ts_beauty_mv_nc_boost_tfviews_seed2025"
    "ts_beauty_mv_nc_tfviews_seed2025"
)

echo "=== Checkpoint Cleanup ==="
echo "Saved dir: $SAVED_DIR"
echo "Keep list: ${#KEEP[@]} directories"
echo ""

is_keep() {
    local dir="$1"
    for k in "${KEEP[@]}"; do
        if [[ "$dir" == "$k" ]]; then
            return 0
        fi
    done
    return 1
}

echo "--- BEFORE ---"
du -sh "$SAVED_DIR"
echo ""

DELETED=0
KEPT=0
FREED=0

for d in */; do
    d="${d%/}"
    if is_keep "$d"; then
        KEPT=$((KEPT + 1))
        echo "[KEEP] $d"
    else
        SIZE=$(du -sm "$d" 2>/dev/null | cut -f1)
        echo "[DELETE] $d (${SIZE}MB)"
        rm -rf "$d"
        DELETED=$((DELETED + 1))
        FREED=$((FREED + SIZE))
    fi
done

echo ""
echo "--- Cleaning loose .pth files in saved/ root ---"
PTH_COUNT=0
PTH_FREED=0
for f in *.pth; do
    if [[ -f "$f" ]]; then
        SIZE=$(du -sm "$f" 2>/dev/null | cut -f1)
        echo "[DELETE] $f (${SIZE}MB)"
        rm -f "$f"
        PTH_COUNT=$((PTH_COUNT + 1))
        PTH_FREED=$((PTH_FREED + SIZE))
    fi
done

echo ""
echo "--- AFTER ---"
du -sh "$SAVED_DIR"
echo ""
echo "=== Summary ==="
echo "Directories: Kept=$KEPT, Deleted=$DELETED, Freed=${FREED}MB (~$((FREED / 1024))GB)"
echo "Loose .pth:  Deleted=$PTH_COUNT, Freed=${PTH_FREED}MB (~$((PTH_FREED / 1024))GB)"
echo "Total freed: $(( (FREED + PTH_FREED) / 1024 ))GB"
