#!/usr/bin/env bash
# Overnight P0/P1 for stop-gate close + Cross seal (Beauty)
#
# P0 — Early temporal window (TS [0.6,0.1,0.3]):
#   1) Rebuild train-only TF-IDF / Qwen whitening at early cutoff
#   2) Freeze Main hparams: TF(5e-4/0.3), LLM(1e-4/0.1), MV(5e-4/0.3)
#   3) Run TF / LLM / MV — seeds default: 2025 42 2024 (override via SEEDS=)
#
# P1 — Cross Top-K head/tail share (offline on existing Main Cross ckpts)
# Optional — Early LLM ±Cross (1 seed) after P0 seed2025 trio
#
# Usage (5090):
#   nohup bash run_5090_early60_overnight.sh > logs/early60_overnight_nohup.log 2>&1 &
#
# Env knobs:
#   SEEDS="2025"              # only one seed (faster)
#   SEEDS="2025 42 2024"      # default
#   RUN_EARLY_CROSS=1         # also run Early LLM ±Cross (seed=2025)
#   SKIP_FEATURE_REBUILD=1    # skip TF/Qwen rebuild if already done
#   SKIP_TOPK_SHARE=1         # skip offline Top-K analysis
#   GPU_ID=0

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
MIN5=5
PA_EP=20
PB_EP=50
# Prefer seed2025 first so morning has the mandatory seed even if later seeds unfinished
SEEDS=(${SEEDS:-2025 42 2024})
RUN_EARLY_CROSS="${RUN_EARLY_CROSS:-1}"
SKIP_FEATURE_REBUILD="${SKIP_FEATURE_REBUILD:-0}"
SKIP_TOPK_SHARE="${SKIP_TOPK_SHARE:-0}"

TFIDF_E60="$ROOT/dataset/Amazon_Beauty/item_text_emb.base.ts_early60.npy"
QWEN_E60="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts_early60.npy"
VIEWS_E60="$ROOT/dataset/Amazon_Beauty/qwen2.5_7b_4views_ts_early60"
BASE_CFG="sasrec_baseline_50ep_stratified_ts_early60.yaml"

CSV="logs/early60_stopgate_beauty.csv"
LOG="logs/early60_overnight.log"
mkdir -p logs saved paper_recsys

echo "tag,model,lr,dropout,seed,use_cross,valid_mrr,test_mrr,valid_ndcg,test_ndcg" > "$CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

extract_metrics() {
  local LOGFILE="$1"
  # Prefer Phase-B Run Summary JSON (most reliable)
  python3 - "$LOGFILE" <<'PY' || true
import json, re, sys
path = sys.argv[1]
text = open(path, errors="ignore").read()
# find last Phase-B summary
cands = []
for m in re.finditer(r"\[Run Summary JSON\]\s*(\{.*\})", text):
    try:
        obj = json.loads(m.group(1))
    except Exception:
        continue
    cands.append(obj)
obj = None
for c in reversed(cands):
    if c.get("label") == "Phase-B" or "phase_b_result" in (c.get("results") or {}):
        obj = c
        break
if obj is None and cands:
    obj = cands[-1]
vm = tm = vn = tn = "0"
if obj:
    res = obj.get("results") or {}
    pb = res.get("phase_b_result") or res
    bv = pb.get("best_valid_result") or res.get("best_valid_result") or {}
    tr = pb.get("test_result") or {}
    vm = str(bv.get("mrr@10", bv.get("MRR@10", 0)))
    vn = str(bv.get("ndcg@10", bv.get("NDCG@10", 0)))
    tm = str(tr.get("mrr@10", tr.get("MRR@10", 0)))
    tn = str(tr.get("ndcg@10", tr.get("NDCG@10", 0)))
print(f"{vm},{tm},{vn},{tn}")
PY
}

run_one() {
  local LABEL="$1" MODEL="$2" YAML="$3" LR="$4" DO="$5" SEED="$6" TAG="$7"
  local USE_CROSS="${8:-false}"
  local EXTRA="${9:-}"
  local CKPT="./saved/${TAG}"
  local LOGFILE="logs/${TAG}.log"
  local CFG="{'learning_rate':${LR},'attn_dropout_prob':${DO},'hidden_dropout_prob':${DO},'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'use_cross':${USE_CROSS},'use_seq_text_cross':false,'use_multiview_text_cross':${USE_CROSS},'text_cross_layer_num':$([ "$USE_CROSS" = true ] && echo 1 || echo 0),'item_text_emb_path_base':'${TFIDF_E60}'${EXTRA}}"

  log ">>> $LABEL | seed=$SEED lr=$LR do=$DO cross=$USE_CROSS"
  local t0=$SECONDS

  python scripts/two_phase_train.py \
    --model "$MODEL" \
    --dataset Amazon_Beauty \
    --config_files "$ROOT/$YAML" \
    --config_dict "$CFG" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions $MIN5 \
    --phase_a_grid --align_grid "0.10" --tau_grid "0.05" \
    --backbone_burnin_epochs 0 --burnin_eval_step 2 \
    --phase_a_epochs $PA_EP --phase_a_eval_step 1 --phase_a_valid_metric "MRR@10" \
    --metric_baseline 0.0 --metric_gain_threshold 0.0 \
    --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b --phase_b_epochs $PB_EP --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT" --seed "$SEED" \
    --watchdog_disable --save \
    > "$LOGFILE" 2>&1 || true

  local elapsed=$(( SECONDS - t0 ))
  local METRICS
  METRICS=$(extract_metrics "$LOGFILE")
  IFS=',' read -r VALID_MRR TEST_MRR VALID_NDCG TEST_NDCG <<< "$METRICS"
  log "  done (${elapsed}s) valid_mrr=${VALID_MRR:-N/A} test_mrr=${TEST_MRR:-N/A}"
  echo "${TAG},${MODEL},${LR},${DO},${SEED},${USE_CROSS},${VALID_MRR:-0},${TEST_MRR:-0},${VALID_NDCG:-0},${TEST_NDCG:-0}" >> "$CSV"
}

# ── Step 0: feature rebuild ──────────────────────────────────────────────────
log "======== Early60 overnight started ========"
log "Seeds: ${SEEDS[*]} | RUN_EARLY_CROSS=$RUN_EARLY_CROSS"

if [[ "$SKIP_FEATURE_REBUILD" != "1" ]]; then
  log ">>> Rebuild TF-IDF @ early60 cutoff"
  t0=$SECONDS
  python tools/build_item_text_emb_base.py \
    --dataset Amazon_Beauty \
    --config "$BASE_CFG" \
    --output "$TFIDF_E60" \
    --title_field title \
    --svd_dim 256 \
    --dtype float16 \
    --center --whiten \
    >> logs/rebuild_early60_tfidf.log 2>&1
  log "  TF-IDF done ($((SECONDS - t0))s) → $TFIDF_E60"

  log ">>> Rebuild Qwen single/multi-view whitening @ early60"
  t0=$SECONDS
  python tools/rebuild_qwen_emb_ts.py \
    --dataset Amazon_Beauty \
    --config "$BASE_CFG" \
    --suffix .ts_early60 \
    --views_out_dir qwen2.5_7b_4views_ts_early60 \
    >> logs/rebuild_early60_qwen.log 2>&1
  log "  Qwen done ($((SECONDS - t0))s) → $QWEN_E60 + $VIEWS_E60"
else
  log "SKIP feature rebuild"
fi

# ── P1 offline Top-K head/tail share (can run before/while GPU train; uses GPU briefly) ──
if [[ "$SKIP_TOPK_SHARE" != "1" ]]; then
  log ">>> P1 Top-K head/tail share (Main Cross ckpts, offline)"
  t0=$SECONDS
  python scripts/analyze_rank_transition.py \
    --ckpt_nocross saved/ps_beauty_llm_nc_noboost_seed2025 \
    --ckpt_cross saved/ps_beauty_llm_cross_noboost_seed2025 \
    --dataset Amazon_Beauty \
    --max_users 3000 \
    --output paper_recsys/rank_transition_beauty_llm.md \
    >> logs/topk_share_llm.log 2>&1 || log "  WARN: LLM Top-K analysis failed (see logs/topk_share_llm.log)"
  python scripts/analyze_rank_transition.py \
    --ckpt_nocross saved/ps_beauty_tf_nc_noboost_seed2025 \
    --ckpt_cross saved/ps_beauty_tf_cross_noboost_seed2025 \
    --dataset Amazon_Beauty \
    --max_users 3000 \
    --output paper_recsys/rank_transition_beauty_tf.md \
    >> logs/topk_share_tf.log 2>&1 || log "  WARN: TF Top-K analysis failed"
  python scripts/analyze_rank_transition.py \
    --ckpt_nocross saved/ps_beauty_mv_nc_noboost_seed2025 \
    --ckpt_cross saved/ps_beauty_mv_cross_noboost_seed2025 \
    --dataset Amazon_Beauty \
    --max_users 3000 \
    --output paper_recsys/rank_transition_beauty_mv.md \
    >> logs/topk_share_mv.log 2>&1 || log "  WARN: MV Top-K analysis failed"
  log "  Top-K share done ($((SECONDS - t0))s)"
fi

# ── P0 Early window training ─────────────────────────────────────────────────
log "======== P0 Early60 TF/LLM/MV (frozen Main hparams) ========"
for SEED in "${SEEDS[@]}"; do
  log "-------- Seed=$SEED --------"

  run_one "LLM" "SASRecAlignV3" "sasrec_align_qwen3_stratified_v3_ts_early60.yaml" \
    0.0001 0.1 "$SEED" "ew_beauty_llm_lr1e4_do0.1_seed${SEED}" false \
    ",'item_text_emb_path_llm':'${QWEN_E60}'"

  run_one "TF" "SASRecAlignV3" "sasrec_align_base_stratified_v3_ts_early60.yaml" \
    0.0005 0.3 "$SEED" "ew_beauty_tf_lr5e4_do0.3_seed${SEED}" false

  run_one "MV" "SASRecAlignMultiViewV3" "sasrec_align_multi_view_v3_stratified_ts_early60.yaml" \
    0.0005 0.3 "$SEED" "ew_beauty_mv_lr5e4_do0.3_seed${SEED}" false \
    ",'item_text_emb_split_dir':'${VIEWS_E60}'"
done

# ── Optional: Early LLM ±Cross (1 seed) ──────────────────────────────────────
if [[ "$RUN_EARLY_CROSS" == "1" ]]; then
  log "======== P1 optional: Early LLM ±Cross seed=2025 ========"
  # no-Cross already done above for seed2025; only need +Cross
  run_one "LLM+Cross" "SASRecAlignV3" "sasrec_align_qwen3_stratified_v3_ts_early60.yaml" \
    0.0001 0.1 2025 "ew_beauty_llm_cross_lr1e4_do0.1_seed2025" true \
    ",'item_text_emb_path_llm':'${QWEN_E60}'"
fi

log "======== Early60 overnight completed ========"
log "CSV: $CSV"
python3 - "$CSV" <<'PY' | tee -a "$LOG"
import csv, sys
from collections import defaultdict
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
print("\n=== Early60 ranking by seed (test MRR) ===")
by_seed = defaultdict(list)
for r in rows:
    if r.get("use_cross") == "true":
        continue
    by_seed[r["seed"]].append(r)
for seed, items in sorted(by_seed.items()):
    items = sorted(items, key=lambda x: -float(x["test_mrr"] or 0))
    order = " > ".join(f"{i['tag'].split('_')[2]}={float(i['test_mrr']):.4f}" for i in items)
    print(f"  seed={seed}: {order}")
print("\n=== Means (no-Cross) ===")
by_model = defaultdict(list)
for r in rows:
    if r.get("use_cross") == "true":
        continue
    # tag: ew_beauty_{model}_...
    parts = r["tag"].split("_")
    model = parts[2] if len(parts) > 2 else r["model"]
    by_model[model].append(float(r["test_mrr"] or 0))
for m, vals in by_model.items():
    import statistics as st
    mean = sum(vals)/len(vals)
    std = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {m}: {mean:.4f} ± {std:.4f} (n={len(vals)})")
PY
