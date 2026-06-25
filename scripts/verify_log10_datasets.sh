#!/usr/bin/env bash
# Preflight: log10 data for GRU4Rec baseline offload (ID + TF-IDF).
# Run on 5090: bash scripts/verify_log10_datasets.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/log10_env.sh"

check_remote() { ssh "$LOG10_SSH" "$1"; }

echo "======== log10 baseline preflight ($LOG10_SSH) ========"
FAIL=0

require_file() {
  local label="$1" path="$2"
  if check_remote "test -f '$path'"; then
    local sz
    sz=$(check_remote "ls -lh '$path' | awk '{print \$5}'")
    printf "  ✅ %-28s (%s)\n" "$label" "$sz"
  else
    printf "  ❌ %-28s MISSING\n" "$label"
    FAIL=1
  fi
}

warn_missing() {
  local label="$1" path="$2"
  if check_remote "test -e '$path'"; then
    printf "  ✅ %-28s (5090-only ok)\n" "$label"
  else
    printf "  ⚠️  %-28s not on log10 (5090 only)\n" "$label"
  fi
}

echo ""
echo "## ID-only: inter + item"
for ds in Amazon_Beauty Amazon_Toys_and_Games; do
  require_file "${ds}.inter" "/home/charlie/project/RecBole/dataset/${ds}/${ds}.inter"
  require_file "${ds}.item" "/home/charlie/project/RecBole/dataset/${ds}/${ds}.item"
  lines=$(check_remote "wc -l < '/home/charlie/project/RecBole/dataset/${ds}/${ds}.inter'")
  echo "       ${ds} lines: $lines"
done

echo ""
echo "## TF-IDF: item_text_emb.base.npy"
require_file "Beauty TF-IDF" "/home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy"
require_file "Toys TF-IDF" "/home/charlie/project/RecBole/dataset/Amazon_Toys_and_Games/item_text_emb.base.npy"

echo ""
echo "## LLM/MV (optional on log10 — run on 5090)"
warn_missing "Beauty qwen3.base" "/home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy"
warn_missing "Beauty MV 4views" "/home/charlie/project/RecBole/dataset/Amazon_Beauty/qwen2.5_7b_4views"
warn_missing "Toys qwen2.5 base" "/home/charlie/project/RecBole/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.npy"
warn_missing "Toys MV 4views" "/home/charlie/project/RecBole/dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views"

echo ""
echo "## Python"
check_remote "bash -lc 'source /home/charlie/project/RecBole/scripts/recbole_env.sh && python -c \"import torch; print(\\\"torch\\\", torch.__version__, \\\"cuda\\\", torch.cuda.is_available())\"'"

echo ""
if [[ "$FAIL" -eq 0 ]]; then
  echo "✅ log10 ready for GRU4Rec ID + TF-IDF offload (16 runs)"
  exit 0
fi
echo "❌ Missing required baseline data — run: bash scripts/sync_log10_gru4rec_data.sh"
exit 1
