#!/usr/bin/env bash
# Post-UniSRec overnight queue:
#   1) Warm-target-only re-eval on Beauty stopgate + Toys wrap ckpts
#   2) Re-run audits / consolidate paper tables for daytime analysis
#
# Usage:
#   nohup bash run_5090_post_unisrec_queue.sh > logs/post_unisrec_nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPU_ID="${GPU_ID:-0}"
QLOG="logs/post_unisrec_queue.log"
mkdir -p logs paper_recsys
: > "$QLOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

log "======== Post-UniSRec queue START ========"

# -------- ckpt list (GTS Main, no-boost) --------
LIST="logs/warm_eval_ckpt_list.txt"
cat > "$LIST" <<'EOF'
# Beauty ID-only 3-seed (sync these log10 ckpts to 5090 before running)
saved/wrap_beauty_id_lr1e4_do0.1_seed42
saved/wrap_beauty_id_lr1e4_do0.1_seed2024
saved/wrap_beauty_id_lr1e4_do0.1_seed2026
# Beauty stop-gate 3-seed
saved/sg_beauty_tf_lr5e4_do0.3_seed42
saved/sg_beauty_tf_lr5e4_do0.3_seed2024
saved/sg_beauty_tf_lr5e4_do0.3_seed2026
saved/sg_beauty_llm_lr1e4_do0.1_seed42
saved/sg_beauty_llm_lr1e4_do0.1_seed2024
saved/sg_beauty_llm_lr1e4_do0.1_seed2026
saved/sg_beauty_mv_lr5e4_do0.3_seed42
saved/sg_beauty_mv_lr5e4_do0.3_seed2024
saved/sg_beauty_mv_lr5e4_do0.3_seed2026
# Toys GTS wrap 3-seed
saved/wrap_toys_id_lr1e4_do0.1_seed42
saved/wrap_toys_id_lr1e4_do0.1_seed2024
saved/wrap_toys_id_lr1e4_do0.1_seed2026
saved/wrap_toys_tf_lr5e4_do0.3_seed42
saved/wrap_toys_tf_lr5e4_do0.3_seed2024
saved/wrap_toys_tf_lr5e4_do0.3_seed2026
saved/wrap_toys_llm_lr1e4_do0.1_seed42
saved/wrap_toys_llm_lr1e4_do0.1_seed2024
saved/wrap_toys_llm_lr1e4_do0.1_seed2026
saved/wrap_toys_mv_lr5e4_do0.3_seed42
saved/wrap_toys_mv_lr5e4_do0.3_seed2024
saved/wrap_toys_mv_lr5e4_do0.3_seed2026
EOF

log "======== Warm-only re-eval ========"
python scripts/eval_warm_targets_only.py \
  --ckpt_list "$LIST" \
  --out logs/warm_only_eval_min5.csv \
  --gpu_id "$GPU_ID" \
  --min_train_interactions 5 \
  --skip_valid \
  >> logs/warm_only_eval_min5_nohup.log 2>&1 || log "WARN: warm-only eval had errors (see logs/warm_only_eval_min5_nohup.log)"

log "======== Audits + paper table consolidate ========"
python scripts/audit_warm_targets.py --dataset Amazon_Beauty \
  --out paper_recsys/warm_target_audit_amazon_beauty.md || true
python scripts/audit_warm_targets.py --dataset Amazon_Toys_and_Games \
  --out paper_recsys/warm_target_audit_amazon_toys_and_games.md || true

python3 - <<'PY' || true
"""Consolidate CSVs into a diagnostic table pack pending protocol sanity gates."""
from pathlib import Path
from datetime import datetime
import csv, statistics as st

ROOT = Path(".")
out = ROOT / "paper_recsys" / "paper_tables_overnight.md"
lines = [
    "# Paper Tables Pack — overnight\n",
    f"> Generated: {datetime.now():%Y-%m-%d %H:%M}  ",
    "> **Diagnostic pack only.** Warm-only and UniSRec checkpoint reloads must "
    "reproduce `min_train_interactions=5` before they are merged into paper tables.\n",
]

def read_csv(p):
    p = ROOT / p
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))

def mean_std(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    if not vals:
        return "—", "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}", "—"
    return f"{st.mean(vals):.4f}", f"{st.stdev(vals):.4f}"

# Beauty LOO
rows = read_csv("logs/beauty_loo_results.csv")
lines.append("## Beauty Clean LOO (test MRR@10)\n")
lines.append("| Model | seed42 | seed2024 | seed2026 | mean ± std |\n|-------|-------:|---------:|---------:|-----------:|")
for key, name in [("_id_", "ID"), ("_tfidf_", "TF"), ("_llm_", "LLM"), ("_mv_", "MV")]:
    by = {r["seed"]: r["test_mrr"] for r in rows if key in r["tag"]}
    vals = [by.get(s, "") for s in ("42", "2024", "2026")]
    m, s = mean_std(vals)
    lines.append(f"| {name} | {vals[0] or '—'} | {vals[1] or '—'} | {vals[2] or '—'} | {m} ± {s} |")

# Toys LOO
rows = read_csv("logs/toys_loo_results.csv")
lines.append("\n## Toys Clean LOO (test MRR@10)\n")
lines.append("| Model | seed42 | seed2024 | seed2026 | mean ± std |\n|-------|-------:|---------:|---------:|-----------:|")
for key, name in [("_id_", "ID"), ("_tfidf_", "TF"), ("_llm_", "LLM"), ("_mv_", "MV")]:
    by = {r["seed"]: r["test_mrr"] for r in rows if key in r["tag"]}
    vals = [by.get(s, "") for s in ("42", "2024", "2026")]
    m, s = mean_std(vals)
    lines.append(f"| {name} | {vals[0] or '—'} | {vals[1] or '—'} | {vals[2] or '—'} | {m} ± {s} |")

# Toys GTS
rows = read_csv("logs/wsdm_wrapup_toys_gts.csv")
lines.append("\n## Toys GTS (test MRR@10)\n")
lines.append("| Model | seed42 | seed2024 | seed2026 | mean ± std |\n|-------|-------:|---------:|---------:|-----------:|")
for key, name in [("_id_", "ID"), ("_tf_", "TF"), ("_llm_", "LLM"), ("_mv_", "MV")]:
    by = {r["seed"]: r["test_mrr"] for r in rows if key in r["tag"]}
    vals = [by.get(s, "") for s in ("42", "2024", "2026")]
    m, s = mean_std(vals)
    lines.append(f"| {name} | {vals[0] or '—'} | {vals[1] or '—'} | {vals[2] or '—'} | {m} ± {s} |")

# UniSRec
rows = read_csv("logs/unisrec_gts_results_min5.csv")
if not rows:
    rows = read_csv("paper_recsys/unisrec_gts_results_min5.csv")
lines.append("\n## UniSRec GTS (min5 test MRR@10)\n")
lines.append("| Dataset | seed42 | seed2024 | seed2026 | mean |\n|---------|-------:|---------:|---------:|-----:|")
for ds, key in [("Beauty", "beauty"), ("Toys", "toys")]:
    by = {r["seed"]: r["test_mrr"] for r in rows if key in r["tag"]}
    vals = [by.get(s, "") for s in ("42", "2024", "2026")]
    m, _ = mean_std(vals)
    lines.append(f"| {ds} | {vals[0] or '—'} | {vals[1] or '—'} | {vals[2] or '—'} | {m} |")

# Warm-only
warm = read_csv("logs/warm_only_eval_min5.csv")
if warm:
    lines.append("\n## Warm-only vs All-target (test MRR@10)\n")
    lines.append("| tag | unseen_share | MRR all | MRR warm |\n|-----|-------------:|--------:|---------:|")
    for r in warm:
        if r.get("error"):
            continue
        lines.append(
            f"| {r.get('tag','')} | {r.get('test_unseen_share','')} | "
            f"{r.get('test_mrr@10_all','')} | {r.get('test_mrr@10_warm','')} |"
        )

out.write_text("\n".join(lines) + "\n")
print(f"Wrote {out}")
PY

cp -f logs/warm_only_eval_min5.csv paper_recsys/ 2>/dev/null || true
cp -f logs/unisrec_gts_results_min5.csv paper_recsys/ 2>/dev/null || true

log "======== Post-UniSRec queue COMPLETE ========"
touch logs/post_unisrec.DONE
log "Artifacts: logs/warm_only_eval_min5.csv paper_recsys/paper_tables_overnight.md"
