#!/bin/bash
# 统计 rebuttal 文件词数（200词限制）
FILE="${1:-rebuttal_polished.txt}"
DIR="$(cd "$(dirname "$0")" && pwd)"
FILEPATH="$DIR/$FILE"

if [ ! -f "$FILEPATH" ]; then
    echo "文件不存在: $FILEPATH"
    exit 1
fi

COUNT=$(wc -w < "$FILEPATH")
LIMIT=200

echo "文件: $FILE"
echo "词数: $COUNT / $LIMIT"

if [ "$COUNT" -le "$LIMIT" ]; then
    echo "状态: 通过 (剩余 $((LIMIT - COUNT)) 词)"
else
    echo "状态: 超限! 需删减 $((COUNT - LIMIT)) 词"
fi
