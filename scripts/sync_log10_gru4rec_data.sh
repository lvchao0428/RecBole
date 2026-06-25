#!/usr/bin/env bash
# 5090 → log10: sync GRU4Rec baseline code + minimal text embeddings.
# ID-only needs inter/item only (already on log10).
# TF-IDF needs item_text_emb.base.npy (Toys must rsync from 5090).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/log10_env.sh"

REMOTE="${LOG10_SSH}:${LOG10_PROJECT}"
RSYNC=(rsync -avz)

echo ">>> sync GRU4Rec bundle 5090 → log10"

ssh "$LOG10_SSH" "mkdir -p ${LOG10_PROJECT}/{scripts,recbole,logs,saved,run_metrics,dataset/Amazon_Beauty,dataset/Amazon_Toys_and_Games}"

# Code (extends generic sync)
bash "$ROOT/scripts/sync_log10_code_from_5090.sh"

"${RSYNC[@]}" \
  "$ROOT/run_gru4rec_v3_batch_beauty.sh" \
  "$ROOT/run_gru4rec_v3_batch_toys.sh" \
  "$ROOT/run50epBase_gru4rec_v3_beauty_stratified.sh" \
  "$ROOT/run50epBase_gru4rec_v3_toys_stratified.sh" \
  "$ROOT/two_phase_run_gru4rec_tfidf_v3_beauty_stratified.sh" \
  "$ROOT/two_phase_run_gru4rec_tfidf_v3_toys_stratified.sh" \
  "$ROOT/run_log10_gru4rec_baselines.sh" \
  "$ROOT/gru4rec_"*.yaml \
  "${REMOTE}/"

# Toys TF-IDF embedding (~165MB)
echo ">>> rsync Toys item_text_emb.base.npy"
"${RSYNC[@]}" \
  "$ROOT/dataset/Amazon_Toys_and_Games/item_text_emb.base.npy" \
  "${REMOTE}/dataset/Amazon_Toys_and_Games/"

# Beauty LLM path fix: yaml expects qwen3.base.npy; log10 may only have qwen3.npy
ssh "$LOG10_SSH" "
  B=/home/charlie/project/RecBole/dataset/Amazon_Beauty
  if [[ ! -f \$B/item_text_emb.qwen3.base.npy && -f \$B/item_text_emb.qwen3.npy ]]; then
    ln -sf item_text_emb.qwen3.npy \$B/item_text_emb.qwen3.base.npy
    echo '  linked qwen3.npy -> qwen3.base.npy'
  fi
"

ssh "$LOG10_SSH" "chmod +x ${LOG10_PROJECT}/run_log10_gru4rec_baselines.sh"

echo "✅ GRU4Rec log10 data sync done"
echo "   Verify: bash scripts/verify_log10_datasets.sh"
