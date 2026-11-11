#!/usr/bin/env bash
# Retrain Beauty + Grocery UniSRec on cutoff-fit Qwen (.ts.npy) for Table 2.
#
# Why a new script:
#   Old runners only passed seed. unisrec_*_stratified_ts.yaml still pointed at
#   item_text_emb.qwen2.5_7b.base.npy (full-catalog whitening).
#   Toys was already retrained as unisrec_toys_gts_ts_* (0818/0819).
#
# Safety:
#   - New dirs only: unisrec_{beauty,grocery}_gts_ts_seed{42,2024,2026}
#   - Never writes into unisrec_beauty_gts_* or unisrec_grocery_gts_*
#   - yaml + config_dict both lock .ts.npy; train log must contain .base.ts.npy
#   - Train failure aborts (no || true)
#   - Last eval uses repo-root eval_gts_last.py (train-catalog ranking)
#   - Smoke-evals one known SASRec ckpt per dataset against locked Table-2 numbers
#
# Usage (on 5090):
#   bash run_5090_unisrec_beauty_grocery_ts_0822.sh --check
#   nohup bash run_5090_unisrec_beauty_grocery_ts_0822.sh \
#     > logs/unisrec_bg_ts_0822_nohup.log 2>&1 &
#
#   DATASETS=grocery bash run_5090_unisrec_beauty_grocery_ts_0822.sh
#   DATASETS=beauty  bash run_5090_unisrec_beauty_grocery_ts_0822.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
SEEDS=(42 2024 2026)
SAMPLE_SEEDS=(42 123 456)
DATASETS="${DATASETS:-grocery,beauty}"
CHECK_ONLY=0
SKIP_SMOKE=0
QLOG="$ROOT/logs/unisrec_bg_ts_0822_queue.log"
mkdir -p "$ROOT/logs" "$ROOT/saved" "$ROOT/paper_recsys"

for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    *)
      echo "Unknown arg: $arg" >&2
      echo "Usage: $0 [--check] [--skip-smoke]" >&2
      exit 2
      ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }

PY=python
command -v python >/dev/null 2>&1 || PY=python3

# Locked GTS-Last warm MRR from the paper Table-2 source CSVs (single seed).
# Smoke must match these before we trust the Last eval script.
SMOKE_GROCERY_TAG="wrap_grocery_mv_lr5e4_do0.3_seed42"
SMOKE_GROCERY_MRR="0.005191"
SMOKE_BEAUTY_TAG="wrap_beauty_mv_lr5e4_do0.3_seed42"
SMOKE_BEAUTY_MRR="0.006291"

# GTS-Last eligible / warm cohorts (min-5, same as main tables).
EXPECT_BEAUTY_N=4520
EXPECT_BEAUTY_WARM=3137
EXPECT_GROCERY_N=4152
EXPECT_GROCERY_WARM=2923

has_dataset() {
  local name="$1"
  [[ ",${DATASETS}," == *",${name},"* ]]
}

emb_path() {
  local ds_dir="$1"
  echo "${ROOT}/dataset/${ds_dir}/item_text_emb.qwen2.5_7b.base.ts.npy"
}

leaky_emb_path() {
  local ds_dir="$1"
  echo "${ROOT}/dataset/${ds_dir}/item_text_emb.qwen2.5_7b.base.npy"
}

forbid_old_tag() {
  local tag="$1"
  case "$tag" in
    unisrec_beauty_gts_seed*|unisrec_grocery_gts_seed*|unisrec_toys_gts_seed*)
      log "FATAL: refusing to write historical tag $tag"
      exit 1
      ;;
  esac
}

check_yaml_cutoff() {
  local yaml="$1"
  if ! grep -q "split: {'TS': \[0.8, 0.1, 0.1\]}" "$yaml"; then
    log "FATAL: $yaml is not TS[0.8,0.1,0.1]"
    exit 1
  fi
  if ! grep -q 'item_text_emb_path:.*qwen2.5_7b.base.ts.npy' "$yaml"; then
    log "FATAL: $yaml does not lock qwen2.5_7b.base.ts.npy"
    exit 1
  fi
  if grep -q 'qwen2.5_7b.base.npy"' "$yaml"; then
    log "FATAL: $yaml still contains non-ts .base.npy"
    exit 1
  fi
}

check_emb_pair() {
  local ds_dir="$1"
  local ts leaky
  ts="$(emb_path "$ds_dir")"
  leaky="$(leaky_emb_path "$ds_dir")"
  if [[ ! -f "$ts" ]]; then
    log "FATAL: missing cutoff-fit embedding $ts"
    exit 1
  fi
  if [[ ! -f "$leaky" ]]; then
    log "WARN: leaky .base.npy missing at $leaky (ok if unused)"
    return 0
  fi
  local ts_md leaky_md
  ts_md="$(md5sum "$ts" | awk '{print $1}')"
  leaky_md="$(md5sum "$leaky" | awk '{print $1}')"
  if [[ "$ts_md" == "$leaky_md" ]]; then
    log "FATAL: $ts is identical to $leaky — cutoff-fit file is not distinct"
    exit 1
  fi
  log "  emb OK $ds_dir ts=${ts_md:0:8} leaky=${leaky_md:0:8} (different)"
}

running_recbole_jobs() {
  # Match python workers only. A wrapper `bash -c '...two_phase_train.py...'`
  # must not count as a GPU job (that aborted the first launch).
  pgrep -af "python.*two_phase_train.py|python.*eval_gts_last.py|python.*eval_gts_random.py" \
    | grep -v "pgrep -af" || true
}

check_gpu_idle() {
  local others
  others="$(running_recbole_jobs)"
  if [[ -n "$others" ]]; then
    log "WARN: other train/eval processes:"
    echo "$others" | tee -a "$QLOG"
    if [[ "$CHECK_ONLY" -eq 0 ]]; then
      log "FATAL: refuse to start while another RecBole job is on GPU"
      exit 1
    fi
  fi
}

preflight() {
  log "======== PREFLIGHT ========"
  log "ROOT=$ROOT GPU_ID=$GPU_ID DATASETS=$DATASETS CHECK_ONLY=$CHECK_ONLY"
  [[ -f "$ROOT/scripts/two_phase_train.py" ]] || { log "FATAL: missing two_phase_train.py"; exit 1; }
  [[ -f "$ROOT/eval_gts_last.py" ]] || { log "FATAL: missing repo-root eval_gts_last.py"; exit 1; }
  [[ -f "$ROOT/scripts/eval_gts_random.py" ]] || { log "FATAL: missing eval_gts_random.py"; exit 1; }

  if ! grep -q "warm_candidates_only" "$ROOT/eval_gts_last.py"; then
    log "FATAL: eval_gts_last.py has no train-catalog ranking"
    exit 1
  fi
  if ! grep -q "warm_candidates_only" "$ROOT/scripts/eval_gts_random.py"; then
    log "FATAL: eval_gts_random.py has no train-catalog ranking"
    exit 1
  fi

  if has_dataset grocery; then
    check_yaml_cutoff "$ROOT/unisrec_grocery_stratified_ts.yaml"
    check_emb_pair "Amazon_Grocery_and_Gourmet_Food"
    [[ -f "$ROOT/dataset/Amazon_Grocery_and_Gourmet_Food/Amazon_Grocery_and_Gourmet_Food.inter" ]] \
      || { log "FATAL: missing Grocery .inter"; exit 1; }
    if ! head -1 "$ROOT/dataset/Amazon_Grocery_and_Gourmet_Food/Amazon_Grocery_and_Gourmet_Food.inter" | grep -q timestamp; then
      log "FATAL: Grocery .inter has no timestamp field"
      exit 1
    fi
    [[ -d "$ROOT/saved/wrap_grocery_mv_lr5e4_do0.3_seed42" ]] \
      || { log "FATAL: missing smoke ckpt saved/$SMOKE_GROCERY_TAG"; exit 1; }
  fi
  if has_dataset beauty; then
    check_yaml_cutoff "$ROOT/unisrec_beauty_stratified_ts_cutoff.yaml"
    check_emb_pair "Amazon_Beauty"
    [[ -f "$ROOT/dataset/Amazon_Beauty/Amazon_Beauty.inter" ]] \
      || { log "FATAL: missing Beauty .inter"; exit 1; }
    [[ -d "$ROOT/saved/wrap_beauty_mv_lr5e4_do0.3_seed42" ]] \
      || { log "FATAL: missing smoke ckpt saved/$SMOKE_BEAUTY_TAG"; exit 1; }
  fi

  for tag in unisrec_beauty_gts_seed42 unisrec_grocery_gts_seed42; do
    if [[ -d "$ROOT/saved/$tag" ]]; then
      log "  historical $tag present (will not touch)"
    fi
  done
  check_gpu_idle
  log "PREFLIGHT OK"
}

smoke_one() {
  local tag="$1" expect="$2"
  local out="$ROOT/logs/unisrec_bg_ts_0822_smoke_${tag}.csv"
  log "SMOKE Last eval $tag (expect warm MRR $expect)"
  "$PY" "$ROOT/eval_gts_last.py" \
    --ckpt_dir "$ROOT/saved/${tag}" \
    --out "$out" \
    --min_train_interactions 5 \
    --gpu_id "$GPU_ID"
  "$PY" - "$out" "$tag" "$expect" <<'PY'
import csv, sys
path, tag, expect = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = list(csv.DictReader(open(path)))
if not rows:
    raise SystemExit(f"SMOKE FAIL: empty {path}")
row = rows[0]
got = float(row["test_mrr@10_warm"])
scope = row.get("warm_candidate_scope", "")
if scope != "train_catalog":
    raise SystemExit(f"SMOKE FAIL: scope={scope!r} (want train_catalog)")
if abs(got - expect) > 1.5e-6:
    raise SystemExit(
        f"SMOKE FAIL: {tag} warm MRR {got:.6f} != locked {expect:.6f}. "
        "Last eval does not match Table 2 source CSVs — stop, do not train."
    )
print(f"SMOKE OK {tag} warm_mrr={got:.6f} n={row['n_users']}/{row['n_warm']}")
PY
}

run_train() {
  local tag="$1" dataset="$2" yaml="$3" seed="$4" emb="$5"
  forbid_old_tag "$tag"
  local ckpt="$ROOT/saved/${tag}"
  local logfile="$ROOT/logs/${tag}.log"
  if compgen -G "${ckpt}/*.pth" > /dev/null; then
    log "SKIP train (pth exists): $tag"
    return 0
  fi
  mkdir -p "$ckpt"
  local cfg
  cfg="{'seed':${seed},'item_text_emb_path':'${emb}'}"
  log ">>> TRAIN $tag"
  log "  yaml=$yaml"
  log "  config_dict=$cfg"
  local t0=$SECONDS
  "$PY" "$ROOT/scripts/two_phase_train.py" \
    --model UniSRec \
    --dataset "$dataset" \
    --config_files "$yaml" \
    --config_dict "$cfg" \
    --gpu_id "$GPU_ID" \
    --min_train_interactions 5 \
    --phase_a_epochs 50 --phase_a_eval_step 5 --phase_a_valid_metric "MRR@10" \
    --only_phase_a \
    --checkpoint_dir "$ckpt" --seed "$seed" \
    --watchdog_disable --save \
    > "$logfile" 2>&1
  log "  train done ($((SECONDS - t0))s) $tag"
  if ! compgen -G "${ckpt}/*.pth" > /dev/null; then
    log "FATAL: no .pth after train $tag (see $logfile)"
    exit 1
  fi
  if grep -q "Failed to parse --config_dict" "$logfile"; then
    log "FATAL: config_dict parse failed for $tag"
    exit 1
  fi
  if ! grep -q "qwen2.5_7b.base.ts.npy" "$logfile"; then
    log "FATAL: train log for $tag does not mention .base.ts.npy"
    exit 1
  fi
  if grep -F "item_text_emb.qwen2.5_7b.base.npy" "$logfile" \
      | grep -vF "item_text_emb.qwen2.5_7b.base.ts.npy" \
      | grep -q .; then
    log "FATAL: train log for $tag still references non-ts .base.npy"
    grep -F "item_text_emb.qwen2.5_7b" "$logfile" | tee -a "$QLOG"
    exit 1
  fi
  log "  log-lock OK (.base.ts.npy present, no leaky .base.npy)"
}

eval_last_list() {
  local list="$1" out="$2" expect_n="$3" expect_w="$4"
  log ">>> GTS-Last $list -> $out"
  "$PY" "$ROOT/eval_gts_last.py" \
    --ckpt_list "$list" \
    --out "$out" \
    --min_train_interactions 5 \
    --gpu_id "$GPU_ID"
  "$PY" - "$out" "$expect_n" "$expect_w" <<'PY'
import csv, sys
path, exp_n, exp_w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = list(csv.DictReader(open(path)))
if not rows:
    raise SystemExit(f"Last FAIL: empty {path}")
for r in rows:
    if "error" in r:
        raise SystemExit(f"Last FAIL: {r.get('tag')} error={r['error']}")
    if r.get("warm_candidate_scope") != "train_catalog":
        raise SystemExit(f"Last FAIL: {r['tag']} scope={r.get('warm_candidate_scope')}")
    n, w = int(r["n_users"]), int(r["n_warm"])
    if n != exp_n or w != exp_w:
        raise SystemExit(
            f"Last FAIL: {r['tag']} cohort {n}/{w} != {exp_n}/{exp_w}"
        )
    print(f"  {r['tag']} Last warm MRR={r['test_mrr@10_warm']} "
          f"NDCG={r['test_ndcg@10_warm']} Hit={r['test_hit@10_warm']} "
          f"n={n}/{w}")
print(f"Last OK {path} ({len(rows)} rows)")
PY
}

eval_random_list() {
  local list="$1" out_prefix="$2" expect_n="$3"
  for samp in "${SAMPLE_SEEDS[@]}"; do
    local out="${out_prefix}_seed${samp}.csv"
    log ">>> GTS-Random sample_seed=$samp -> $out"
    "$PY" "$ROOT/scripts/eval_gts_random.py" \
      --ckpt_list "$list" \
      --out "$out" \
      --min_train_interactions 5 \
      --sample_seed "$samp" \
      --gpu_id "$GPU_ID" \
      --force
    "$PY" - "$out" "$expect_n" "$samp" <<'PY'
import csv, sys
path, exp_n, samp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = list(csv.DictReader(open(path)))
if len(rows) < 1:
    raise SystemExit(f"Random FAIL: empty {path}")
for r in rows:
    if "error" in r:
        raise SystemExit(f"Random FAIL: {r.get('tag')} {r['error']}")
    if r.get("warm_candidate_scope") != "train_catalog":
        raise SystemExit(f"Random FAIL: scope={r.get('warm_candidate_scope')}")
    if int(r["n_users"]) != exp_n:
        raise SystemExit(f"Random FAIL: {r['tag']} n_users={r['n_users']} != {exp_n}")
    print(f"  {r['tag']} Rand{samp} warm MRR={r['test_mrr@10_warm']} n={r['n_users']}/{r['n_warm']}")
print(f"Random OK {path}")
PY
  done
}

run_dataset() {
  local name="$1" dataset="$2" yaml="$3" ds_dir="$4" expect_n="$5" expect_w="$6"
  local emb
  emb="$(emb_path "$ds_dir")"
  local list="$ROOT/logs/unisrec_${name}_gts_ts_ckpt_list.txt"
  : > "$list"
  for seed in "${SEEDS[@]}"; do
    local tag="unisrec_${name}_gts_ts_seed${seed}"
    forbid_old_tag "$tag"
    echo "${ROOT}/saved/${tag}" >> "$list"
    run_train "$tag" "$dataset" "$yaml" "$seed" "$emb"
  done
  eval_last_list "$list" \
    "$ROOT/paper_recsys/gts_last_eval_unisrec_${name}_ts.csv" \
    "$expect_n" "$expect_w"
  eval_random_list "$list" \
    "$ROOT/paper_recsys/gts_random_eval_unisrec_${name}_ts" \
    "$expect_n"
}

# ----- main -----
: > "$QLOG"
log "======== UniSRec Beauty/Grocery cutoff-fit START ========"
preflight

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  if [[ -n "$(running_recbole_jobs)" ]]; then
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
      log "SKIP smoke during --check: GPU job already running"
    else
      log "FATAL: GPU job appeared after preflight"
      exit 1
    fi
  else
    log "======== SMOKE (eval must match Table 2 source) ========"
    if has_dataset grocery; then
      smoke_one "$SMOKE_GROCERY_TAG" "$SMOKE_GROCERY_MRR"
    fi
    if has_dataset beauty; then
      smoke_one "$SMOKE_BEAUTY_TAG" "$SMOKE_BEAUTY_MRR"
    fi
  fi
else
  log "SKIP smoke (--skip-smoke)"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "======== CHECK-ONLY done (no training) ========"
  exit 0
fi

if has_dataset grocery; then
  run_dataset grocery Amazon_Grocery_and_Gourmet_Food \
    "$ROOT/unisrec_grocery_stratified_ts.yaml" \
    Amazon_Grocery_and_Gourmet_Food \
    "$EXPECT_GROCERY_N" "$EXPECT_GROCERY_WARM"
fi
if has_dataset beauty; then
  run_dataset beauty Amazon_Beauty \
    "$ROOT/unisrec_beauty_stratified_ts_cutoff.yaml" \
    Amazon_Beauty \
    "$EXPECT_BEAUTY_N" "$EXPECT_BEAUTY_WARM"
fi

log "======== ALL DONE ========"
log "New Last CSVs:"
log "  paper_recsys/gts_last_eval_unisrec_grocery_ts.csv"
log "  paper_recsys/gts_last_eval_unisrec_beauty_ts.csv"
log "Do not edit main_wsdm2027_v3_0823.tex until these are compared to Table 2."
touch "$ROOT/logs/unisrec_bg_ts_0822.DONE"
