#!/usr/bin/env bash
# Scan saved/ + run_metrics for GRU4Rec Phase 4 completion matrix → markdown.
#
# Usage:
#   bash scripts/collect_gru4rec_phase4_results.sh
#   bash scripts/collect_gru4rec_phase4_results.sh --pull-log10   # rsync first

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${GRU4REC_COLLECT_OUT:-paper_recsys/gru4rec_phase4_matrix_$(date +%Y%m%d).md}"
SEEDS=(42 2024 2025 2026)
DATASETS=(beauty toys)

if [[ "${1:-}" == "--pull-log10" ]]; then
  source "$ROOT/scripts/log10_env.sh"
  rsync -avz "${LOG10_SSH}:${LOG10_PROJECT}/saved/gru4rec_"* "${ROOT}/saved/" 2>/dev/null || true
  rsync -avz "${LOG10_SSH}:${LOG10_PROJECT}/run_metrics/" "${ROOT}/run_metrics/" 2>/dev/null || true
fi

has_ckpt() {
  local dir="$1"
  [[ -d "$dir" ]] && compgen -G "${dir}/*.pth" >/dev/null && echo "✅" || echo "—"
}

latest_metric() {
  local pattern="$1"
  local f
  f=$(ls -t run_metrics/*${pattern}* 2>/dev/null | head -1)
  if [[ -z "$f" ]]; then
    echo "—"
    return
  fi
  grep -E "MRR@10|HR@10" "$f" 2>/dev/null | tail -2 | tr '\n' ' ' || echo "(ckpt only)"
}

mkdir -p paper_recsys

{
  echo "# GRU4Rec Phase 4 结果矩阵"
  echo ""
  echo "> 生成: $(date '+%Y-%m-%d %H:%M:%S') · 主机: $(hostname)"
  echo ""
  echo "## 5090 — LLM + MV"
  echo ""
  echo "| Seed | B-LLM | B-MV | T-LLM | T-MV |"
  echo "|------|-------|------|-------|------|"
  for s in "${SEEDS[@]}"; do
    bllm=$(has_ckpt "saved/gru4rec_tfidf_llm_v3_beauty_stratified_seed${s}")
    bmv=$(has_ckpt "saved/gru4rec_multiview_v3_beauty_stratified_seed${s}")
    tllm=$(has_ckpt "saved/gru4rec_tfidf_llm_v3_toys_stratified_seed${s}")
    tmv=$(has_ckpt "saved/gru4rec_multiview_v3_toys_stratified_seed${s}")
    echo "| $s | $bllm | $bmv | $tllm | $tmv |"
  done
  echo ""
  echo "## log10 / 5090 — ID + TF-IDF"
  echo ""
  echo "| Seed | B-ID | B-TF | T-ID | T-TF | 机器(推断) |"
  echo "|------|------|------|------|------|------------|"
  for s in "${SEEDS[@]}"; do
    bid=$(has_ckpt "saved/gru4rec_baseline_v3_beauty_stratified_seed${s}")
    btf=$(has_ckpt "saved/gru4rec_tfidf_v3_beauty_stratified_seed${s}")
    tid=$(has_ckpt "saved/gru4rec_baseline_v3_toys_stratified_seed${s}")
    ttf=$(has_ckpt "saved/gru4rec_tfidf_v3_toys_stratified_seed${s}")
    owner="log10"
    if [[ -f paper_recsys/.gru4rec_owner_5090_seeds ]] && grep -qw "$s" paper_recsys/.gru4rec_owner_5090_seeds; then
      owner="5090"
    fi
    echo "| $s | $bid | $btf | $tid | $ttf | $owner |"
  done
  echo ""
  echo "## 计数"
  n5090_text=0
  nbase=0
  for s in "${SEEDS[@]}"; do
    for cfg in gru4rec_tfidf_llm_v3 gru4rec_multiview_v3; do
      for ds in "${DATASETS[@]}"; do
        compgen -G "saved/${cfg}_${ds}_stratified_seed${s}/*.pth" >/dev/null && n5090_text=$((n5090_text + 1))
      done
    done
    for cfg in gru4rec_baseline_v3 gru4rec_tfidf_v3; do
      for ds in "${DATASETS[@]}"; do
        compgen -G "saved/${cfg}_${ds}_stratified_seed${s}/*.pth" >/dev/null && nbase=$((nbase + 1))
      done
    done
  done
  echo "- 5090 text (LLM+MV): **${n5090_text}/16**"
  echo "- baseline (ID+TF-IDF): **${nbase}/16**"
  echo "- 合计 checkpoint 块: **$((n5090_text + nbase))/32**"
  echo ""
  echo "## 日志路径"
  echo "- 5090 text: \`logs/gru4rec_*_text_seed*.log\`"
  echo "- log10 baseline: \`logs/log10_gru4rec_*.log\`"
  echo "- 5090 baseline: \`logs/5090_gru4rec_*.log\`"
  echo "- 汇总: \`logs/post_main_pipeline_20260625.log\`"
} | tee "$OUT"

echo "Wrote $OUT"
