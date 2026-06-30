#!/usr/bin/env bash
# Evaluate completed GRU4Rec text checkpoints (TF-IDF+LLM + MV) and
# write a markdown summary for Beauty/Toys across seeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEVICE="${DEVICE:-cuda}"
SEEDS=(42 2024 2025 2026)
DATASETS=(beauty toys)
CFGS=(tfidf_llm_v3 multiview_v3)
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
OUTDIR="${OUTDIR:-paper_recsys/gru4rec_text_eval_${DATE_TAG}}"
LOG="${EVAL_LOG:-logs/gru4rec_text_eval_${DATE_TAG}.log}"
SUMMARY_OUT="${SUMMARY_OUT:-paper_recsys/gru4rec_text_eval_summary_${DATE_TAG}.md}"

mkdir -p "$OUTDIR" logs paper_recsys

latest_ckpt() {
  local dir="$1"
  ls -t "$dir"/*.pth 2>/dev/null | head -1
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== GRU4Rec text checkpoint evaluation ========"
log "device=$DEVICE outdir=$OUTDIR"

for ds in "${DATASETS[@]}"; do
  for cfg in "${CFGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ckpt_dir="saved/gru4rec_${cfg}_${ds}_stratified_seed${seed}"
      ckpt="$(latest_ckpt "$ckpt_dir")"
      out_txt="${OUTDIR}/${ds}_${cfg}_seed${seed}.txt"
      if [[ -z "${ckpt:-}" ]]; then
        log "SKIP missing checkpoint: ${ckpt_dir}"
        continue
      fi
      log "EVAL ${ds} ${cfg} seed=${seed}"
      python scripts/standalone_test.py \
        --model_file "$ckpt" \
        --device "$DEVICE" \
        --no_progress \
        --output_file "$out_txt" \
        >> "$LOG" 2>&1
      log "  ✅ wrote ${out_txt}"
    done
  done
done

DATE_TAG_ENV="$DATE_TAG" SUMMARY_OUT_ENV="$SUMMARY_OUT" OUTDIR_ENV="$OUTDIR" python - <<'PY'
import re
import os
from pathlib import Path
from statistics import mean

date_tag = os.environ["DATE_TAG_ENV"]
outdir = Path(os.environ["OUTDIR_ENV"])
summary_out = Path(os.environ["SUMMARY_OUT_ENV"])

rows = []
for txt in sorted(outdir.glob("*.txt")):
    m = re.match(r"(beauty|toys)_(tfidf_llm_v3|multiview_v3)_seed(\d+)\.txt", txt.name)
    if not m:
        continue
    dataset, cfg, seed = m.group(1), m.group(2), int(m.group(3))
    metrics = {}
    raw = False
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip() == "Raw results:":
            raw = True
            continue
        if raw and ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                pass
    if metrics:
        rows.append({
            "dataset": dataset,
            "cfg": cfg,
            "seed": seed,
            "mrr@10": metrics.get("mrr@10"),
            "hit@10": metrics.get("hit@10"),
            "ndcg@10": metrics.get("ndcg@10"),
        })

def fmt_triplet(r):
    return f"{r['mrr@10']*100:.2f} / {r['hit@10']*100:.2f} / {r['ndcg@10']*100:.2f}"

def mean_triplet(items):
    return " / ".join(f"{mean([x[k] for x in items])*100:.2f}" for k in ["mrr@10", "hit@10", "ndcg@10"])

cfg_label = {"tfidf_llm_v3": "TF-IDF+LLM", "multiview_v3": "MV"}

lines = [
    f"# GRU4Rec text 评测汇总（{date_tag}）",
    "",
    "> 指标顺序: `MRR@10 / HR@10 / NDCG@10`，单位 `%`",
    "",
]
for dataset in ["beauty", "toys"]:
    lines.extend([f"## {dataset.title()}", "", "| Config | seed=42 | seed=2024 | seed=2025 | seed=2026 | 4-seed mean |", "|--------|---------|-----------|-----------|-----------|-------------|"])
    for cfg in ["tfidf_llm_v3", "multiview_v3"]:
        items = [r for r in rows if r["dataset"] == dataset and r["cfg"] == cfg]
        by_seed = {r["seed"]: r for r in items}
        cols = []
        for seed in [42, 2024, 2025, 2026]:
            cols.append(fmt_triplet(by_seed[seed]) if seed in by_seed else "—")
        mean_col = mean_triplet(items) if items else "—"
        lines.append(f"| {cfg_label[cfg]} | " + " | ".join(cols + [f"**{mean_col}**"]) + " |")
    lines.extend(["", ""])

summary_out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {summary_out}")
PY

log "======== GRU4Rec text checkpoint evaluation complete ========"
