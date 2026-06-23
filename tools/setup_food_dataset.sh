#!/usr/bin/env bash
set -euo pipefail

# 从 ProcessedDatasets 解压 Food 到 dataset/Food/（仅 inter + item）

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${FOOD_ZIP:-/home/charlie/project/RecSysDatasets/RecBole/ProcessedDatasets/Food/Food.zip}"
DST="${ROOT}/dataset/Food"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: Food zip not found: $SRC"
  exit 1
fi

mkdir -p "$DST"
echo "Extracting Food from $SRC -> $DST"
unzip -o -j "$SRC" "Food/Food.inter" "Food/Food.item" -d "$DST"
for f in Food.inter Food.item; do
  if [[ -f "$DST/$f" ]]; then
    :
  elif [[ -f "$DST/Food/$f" ]]; then
    mv "$DST/Food/$f" "$DST/"
  fi
done
# unzip -j 可能直接落文件名
[[ -f "$DST/Food.inter" ]] || { echo "ERROR: Food.inter missing after unzip"; exit 1; }
[[ -f "$DST/Food.item" ]] || { echo "ERROR: Food.item missing after unzip"; exit 1; }

echo "OK: $(wc -l < "$DST/Food.inter") lines inter, $(wc -l < "$DST/Food.item") lines item"
