#!/usr/bin/env bash
# Correct GTS-Last / GTS-Random eval for Pure-LLM + Hybrid Phase-B checkpoints.
# Does not retrain. Safe to run after hybrid training finishes.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

LOG="$ROOT/logs/p3_fixed_0818.log"
CKPT_LIST="$ROOT/logs/supplement_phase_b_ckpts.txt"
GTS_LAST_CSV="$ROOT/paper_recsys/purellm_hybrid_gts_last.csv"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

collect_phase_b() {
  python3 - <<PY
from pathlib import Path
root = Path("$ROOT")
saved = root / "saved"
pure, hybrid = [], []

for d in sorted(saved.glob("purellm_*")):
    if not d.is_dir():
        continue
    pths = list(d.glob("*.pth"))
    if not pths:
        continue
    pth = max(pths, key=lambda p: p.stat().st_size)
    pure.append(str(pth.parent))

for pat in ("SASRecAlignMultiViewV3-Aug-18-*.pth",
            "SASRecAlignMultiViewV3-Aug-19-*.pth",
            "SASRecAlignMultiViewV3-Aug-20-*.pth"):
    for pth in sorted(saved.glob(pat)):
        if pth.stat().st_size >= 950 * 1024 * 1024:
            hybrid.append(str(pth))

eval_path = root / "logs" / "supplement_phase_b_ckpts.txt"
eval_path.write_text("\\n".join(pure) + ("\\n" if pure else ""))
hyb_path = root / "logs" / "hybrid_nonts_phase_b_ckpts.txt"
hyb_path.write_text("\\n".join(hybrid) + ("\\n" if hybrid else ""))
print(f"eval (purellm) {len(pure)} -> {eval_path}")
print(f"audit-only hybrid {len(hybrid)} -> {hyb_path}")
for x in pure + hybrid:
    print(" ", x)
PY
}

log "======== P3 fixed: collect Phase-B checkpoints ========"
collect_phase_b
n=$(grep -c . "$CKPT_LIST" || true)
log "n_ckpts=$n"
if [ "$n" -lt 6 ]; then
  log "WARN: expected >=6 PureLLM dirs; hybrid Phase-B may still be training"
fi

AUDIT_LIST="$ROOT/logs/protocol_audit_ckpts_0818.txt"
{
  cat "$CKPT_LIST"
  [ -f "$ROOT/logs/hybrid_nonts_phase_b_ckpts.txt" ] && cat "$ROOT/logs/hybrid_nonts_phase_b_ckpts.txt"
  for d in \
    saved/sg_beauty_tf_lr5e4_do0.3_seed42 \
    saved/sg_beauty_llm_lr1e4_do0.1_seed42 \
    saved/sg_beauty_mv_lr5e4_do0.3_seed42 \
    saved/wrap_toys_tf_lr5e4_do0.3_seed42 \
    saved/wrap_toys_llm_lr1e4_do0.1_seed42 \
    saved/wrap_toys_mv_lr5e4_do0.3_seed42 \
    saved/wrap_grocery_tf_lr5e4_do0.3_seed42 \
    saved/wrap_grocery_llm_lr1e4_do0.1_seed42 \
    saved/wrap_grocery_mv_lr5e4_do0.3_seed42 \
    saved/wrap_beauty_id_lr1e4_do0.1_seed42
  do
    [ -e "$ROOT/$d" ] && echo "$ROOT/$d"
  done
} > "$AUDIT_LIST"

log "======== Protocol audit (CPU, no GPU) ========"
python3 "$ROOT/scripts/audit_ckpt_protocol.py" \
  --ckpt_list "$AUDIT_LIST" \
  --out "$ROOT/paper_recsys/checkpoint_protocol_audit_0818.csv" \
  || log "WARN: protocol audit failed"

log "======== GTS-Last warm eval ========"
python3 "$ROOT/scripts/eval_gts_last.py" \
  --ckpt_list "$CKPT_LIST" \
  --gpu_id 0 \
  --min_train_interactions 5 \
  --out "$GTS_LAST_CSV"

log "======== GTS-Random warm eval (seeds 42/123/456) ========"
for SAMPLE_SEED in 42 123 456; do
  python3 "$ROOT/eval_gts_random.py" \
    --ckpt_list "$CKPT_LIST" \
    --gpu_id 0 \
    --min_train_interactions 5 \
    --sample_seed "$SAMPLE_SEED" \
    --out "$ROOT/paper_recsys/purellm_hybrid_gts_random_seed${SAMPLE_SEED}.csv"
done

log "======== P3 fixed DONE ========"
log "Last: $GTS_LAST_CSV"
log "Random: paper_recsys/purellm_hybrid_gts_random_seed{42,123,456}.csv"
