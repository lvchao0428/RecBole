#!/usr/bin/env bash
set -euo pipefail

# 解压 Amazon Grocery → dataset/Amazon_Grocery_and_Gourmet_Food/

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${GROCERY_ZIP:-/home/charlie/project/RecSysDatasets/RecBole/ProcessedDatasets/Amazon_ratings/Amazon_Grocery_and_Gourmet_Food.zip}"
DST="${ROOT}/dataset/Amazon_Grocery_and_Gourmet_Food"
DS="Amazon_Grocery_and_Gourmet_Food"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: Grocery zip not found: $SRC"
  exit 1
fi

mkdir -p "$DST"
echo "Extracting Grocery from $SRC -> $DST"
unzip -o -j "$SRC" "${DS}.inter" "${DS}.item" -d "$DST"

[[ -f "$DST/${DS}.inter" ]] || { echo "ERROR: ${DS}.inter missing"; exit 1; }
[[ -f "$DST/${DS}.item" ]] || { echo "ERROR: ${DS}.item missing"; exit 1; }

echo "OK: $(wc -l < "$DST/${DS}.inter") lines inter, $(wc -l < "$DST/${DS}.item") lines item"
