#!/usr/bin/env bash
#
# GTS-Random re-evaluation on 5090 (same 30 checkpoints as GTS-Last).
# No retraining. Expected ~60-90 min.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
QLOG="$LOG_DIR/gts_random_queue.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

if [ -f scripts/recbole_env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/recbole_env.sh
fi

log "======== GTS-Random START (GPU=0, sample_seed=42) ========"

CKPT_LIST="$LOG_DIR/gts_last_ckpt_list.txt"
if [ ! -f "$CKPT_LIST" ]; then
    cat > "$CKPT_LIST" << 'EOF'
saved/wrap_beauty_id_lr1e4_do0.1_seed42
saved/wrap_beauty_id_lr1e4_do0.1_seed2024
saved/wrap_beauty_id_lr1e4_do0.1_seed2026
saved/sg_beauty_tf_lr5e4_do0.3_seed42
saved/sg_beauty_tf_lr5e4_do0.3_seed2024
saved/sg_beauty_tf_lr5e4_do0.3_seed2026
saved/sg_beauty_llm_lr1e4_do0.1_seed42
saved/sg_beauty_llm_lr1e4_do0.1_seed2024
saved/sg_beauty_llm_lr1e4_do0.1_seed2026
saved/sg_beauty_mv_lr5e4_do0.3_seed42
saved/sg_beauty_mv_lr5e4_do0.3_seed2024
saved/sg_beauty_mv_lr5e4_do0.3_seed2026
saved/unisrec_beauty_gts_seed42
saved/unisrec_beauty_gts_seed2024
saved/unisrec_beauty_gts_seed2026
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
saved/unisrec_toys_gts_seed42
saved/unisrec_toys_gts_seed2024
saved/unisrec_toys_gts_seed2026
EOF
fi

OUT_CSV="$LOG_DIR/gts_random_eval.csv"
N_CKPT=$(grep -c '^saved/' "$CKPT_LIST" || true)
log "Running GTS-Random on ${N_CKPT} checkpoints → $OUT_CSV"

python3 scripts/eval_gts_random.py \
    --ckpt_list "$CKPT_LIST" \
    --out "$OUT_CSV" \
    --min_train_interactions 5 \
    --sample_seed 42 \
    --gpu_id 0 2>&1 | tee "$LOG_DIR/gts_random_eval.log"

cp "$OUT_CSV" paper_recsys/gts_random_eval.csv
touch "$LOG_DIR/gts_random.DONE"
log "======== GTS-Random COMPLETE → $OUT_CSV ========"
log "Copied to paper_recsys/gts_random_eval.csv"
