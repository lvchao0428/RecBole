#!/usr/bin/env bash
# Final supplementary experiments for WSDM 2027 full paper submission.
# 1. Retrain Pure LLM (Beauty/Toys × 3 seeds) WITH --save
# 2. Retrain TF+LLM Hybrid (Beauty/Toys × 3 seeds) WITH --save
# 3. GTS-Last/Random post-hoc eval on all new checkpoints
# 4. Grocery raw 3584-d (Qwen2.5-7B hidden_size) CKA audit
#
# Estimated time: ~14-16 hours total
#   P1: Pure LLM 6 models × ~38min = ~4h
#   P2: TF+LLM hybrid 6 models × ~45min = ~4.5h
#   P3: GTS-Last + GTS-Random eval ~12 ckpts × 5min = ~1h
#   P4: Grocery raw embedding gen ~30min + CKA audit ~5min

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MAIN_YAML="$ROOT/sasrec_baseline_50ep_stratified_ts.yaml"
MV_YAML="$ROOT/sasrec_align_multi_view_v3_stratified_ts.yaml"
LOG="$ROOT/logs/final_supplement_0817.log"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "======== FINAL SUPPLEMENT START ========"

###############################################################################
# P1: Pure LLM-only (no TF-IDF base) — Beauty/Toys × 3 seeds
###############################################################################
log "======== P1: Pure LLM-only (with --save) ========"

PURELLM_CSV="$ROOT/logs/purellm_gts_all_0817.csv"
echo "tag,dataset,seed,test_mrr" > "$PURELLM_CSV"

run_purellm() {
  local DATASET=$1 SEED=$2 SHORT=$3
  local TAG="purellm_${SHORT}_lr1e4_do0.1_seed${SEED}"
  local CKPT_DIR="./saved/${TAG}"

  if [ "$DATASET" = "Amazon_Beauty" ]; then
    LLM_PATH="$ROOT/dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy"
  else
    LLM_PATH="$ROOT/dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.ts.npy"
  fi

  log "--- $TAG ---"
  python3 "$ROOT/two_phase_train.py" \
    --model SASRecAlignV3 --dataset "$DATASET" \
    --config_files "$MAIN_YAML" \
    --config_dict "{'learning_rate':1e-4,'attn_dropout_prob':0.1,'hidden_dropout_prob':0.1,'weight_decay':0.0,'cold_text_boost':0.0,'infer_boost':0.0,'use_cross':false,'use_seq_text_cross':false,'use_multiview_text_cross':false,'text_cross_layer_num':0,'item_text_emb_path_base':'','item_text_emb_path_llm':'${LLM_PATH}','use_llm':true,'disable_text_feature':false,'fuse_text_feature':true}" \
    --gpu_id 0 --min_train_interactions 5 \
    --phase_a_grid --align_grid 0.10 --tau_grid 0.05 \
    --backbone_burnin_epochs 0 --burnin_eval_step 2 \
    --phase_a_epochs 20 --phase_a_eval_step 1 \
    --phase_a_valid_metric MRR@10 \
    --metric_baseline 0.0 --metric_gain_threshold 0.0 \
    --lr_text_head 2e-3 --lr_dnn_cross 5e-4 \
    --phase_a_auto_to_b --phase_b_epochs 50 \
    --backbone_lr_scale 0.1 \
    --checkpoint_dir "$CKPT_DIR" \
    --seed "$SEED" --save \
    > "$ROOT/logs/${TAG}_0817.log" 2>&1

  local MRR=$(grep -o "mrr@10.*[0-9.]\+" "$ROOT/logs/${TAG}_0817.log" | tail -1 | grep -o "[0-9.]\+$")
  echo "${TAG},${DATASET},${SEED},${MRR}" >> "$PURELLM_CSV"
  log "  done: mrr=$MRR"
}

for SEED in 42 2024 2026; do
  run_purellm Amazon_Beauty "$SEED" beauty
  run_purellm Amazon_Toys_and_Games "$SEED" toys
done

log "======== P1 Pure LLM-only complete ========"

###############################################################################
# P2: TF+LLM Hybrid (MV arch, 1 view = Description) — Beauty/Toys × 3 seeds
###############################################################################
log "======== P2: TF+LLM Hybrid (with --save) ========"

HYBRID_CSV="$ROOT/logs/hybrid_gts_all_0817.csv"
echo "tag,dataset,seed,test_mrr" > "$HYBRID_CSV"

run_hybrid() {
  local DATASET=$1 SEED=$2 SHORT=$3
  local TAG="hybrid_${SHORT}_seed${SEED}"

  EXTRA=""
  if [ "$DATASET" = "Amazon_Toys_and_Games" ]; then
    EXTRA=",'dataset':'Amazon_Toys_and_Games','item_text_emb_path_base':'/home/charlie/project/RecBole/dataset/Amazon_Toys_and_Games/item_text_emb.base.npy','item_text_emb_split_dir':'/home/charlie/project/RecBole/dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views'"
  fi

  log "--- $TAG ---"
  python3 "$ROOT/two_phase_train.py" \
    --model SASRecAlignMultiViewV3 --dataset "$DATASET" \
    --config_files "$MV_YAML" \
    --config_dict "{'num_text_views':1,'text_view_indices':[0]${EXTRA}}" \
    --seed "$SEED" \
    --phase_a_epochs 8 --phase_b_epochs 40 \
    --variant_label "$TAG" --save \
    --gpu_id 0 \
    > "$ROOT/logs/${TAG}_0817.log" 2>&1

  local MRR=$(grep -o "mrr@10.*[0-9.]\+" "$ROOT/logs/${TAG}_0817.log" | tail -1 | grep -o "[0-9.]\+$")
  echo "${TAG},${DATASET},${SEED},${MRR}" >> "$HYBRID_CSV"
  log "  done: mrr=$MRR"
}

for SEED in 42 2024 2026; do
  run_hybrid Amazon_Beauty "$SEED" beauty
  run_hybrid Amazon_Toys_and_Games "$SEED" toys
done

log "======== P2 TF+LLM Hybrid complete ========"

###############################################################################
# P3: GTS-Last + GTS-Random post-hoc eval on new checkpoints
###############################################################################
log "======== P3: Post-hoc GTS-Last/Random eval ========"

GTS_LAST_CSV="$ROOT/logs/supplement_gts_last_0817.csv"
GTS_RAND_CSV="$ROOT/logs/supplement_gts_random_0817.csv"

# Collect all new checkpoint paths
collect_ckpts() {
  find "$ROOT/saved" -maxdepth 2 -name "*.pth" -newer "$LOG" 2>/dev/null | sort
}

# Wait for training to finish, then eval
CKPT_LIST="$ROOT/logs/supplement_ckpts_0817.txt"
find "$ROOT/saved" -maxdepth 2 -name "*.pth" -path "*purellm*" -newer "$PURELLM_CSV" > "$CKPT_LIST" 2>/dev/null || true
find "$ROOT/saved" -maxdepth 2 -name "*.pth" -path "*hybrid*" -newer "$PURELLM_CSV" >> "$CKPT_LIST" 2>/dev/null || true

# Also find newly saved Phase B checkpoints by timestamp
find "$ROOT/saved" -maxdepth 1 -name "*.pth" -newer "$PURELLM_CSV" >> "$CKPT_LIST" 2>/dev/null || true

log "Found $(wc -l < "$CKPT_LIST") checkpoints to evaluate"

# GTS-Last eval
log "--- GTS-Last eval ---"
python3 "$ROOT/scripts/eval_gts_last.py" \
  --ckpt_list "$CKPT_LIST" \
  --gpu_id 0 \
  --min_train_inter 5 \
  --output "$GTS_LAST_CSV" \
  >> "$ROOT/logs/supplement_gts_last_0817.log" 2>&1 || log "WARN: GTS-Last eval had errors"

log "GTS-Last results: $GTS_LAST_CSV"

# GTS-Random eval (3 target-sampling seeds)
log "--- GTS-Random eval ---"
for SAMPLE_SEED in 42 123 456; do
  python3 "$ROOT/eval_gts_random.py" \
    --ckpt_list "$CKPT_LIST" \
    --gpu_id 0 \
    --min_train_inter 5 \
    --sample_seed "$SAMPLE_SEED" \
    --output "$ROOT/logs/supplement_gts_random_seed${SAMPLE_SEED}_0817.csv" \
    >> "$ROOT/logs/supplement_gts_random_0817.log" 2>&1 || log "WARN: GTS-Random seed=$SAMPLE_SEED had errors"
done

log "======== P3 Post-hoc eval complete ========"

###############################################################################
# P4: Grocery raw 3584-d CKA audit
###############################################################################
log "======== P4: Grocery raw 3584-d embedding + CKA audit ========"

QWEN_MODEL="/home/charlie/project/qwen/Qwen2.5-7B-Instruct"
DATASET="Amazon_Grocery_and_Gourmet_Food"
MAPPING="$ROOT/dataset/${DATASET}/item_index_mapping.csv"
OUT_DIR="$ROOT/dataset/${DATASET}/qwen2.5_7b_4views_raw_4096d"
mkdir -p "$OUT_DIR"

log "Generating raw 3584-d 4-view embeddings for Grocery..."
python3 "$ROOT/tools/build_item_text_emb_qwen2.5_7b.py" \
  --mapping "$MAPPING" \
  --model_name_or_path "$QWEN_MODEL" \
  --output "$OUT_DIR/combined.npy" \
  --split_output_dir "$OUT_DIR" \
  --output_mode stack \
  --prompt_preset multiview \
  --batch_size 64 \
  --max_length 128 \
  --dtype float16 \
  --flash_attn \
  --dataset "$DATASET" \
  --config "$MAIN_YAML" \
  >> "$ROOT/logs/grocery_raw_3584d_gen.log" 2>&1

log "Grocery embedding done. Running CKA audit..."

# CKA audit on raw embeddings
python3 -c "
import numpy as np, json, sys
sys.path.insert(0, '$ROOT')

N_TRIALS = 3
SUBSAMPLE = 3000
np.random.seed(2024)

out_dir = '$OUT_DIR'
views = []
for i in range(4):
    v = np.load(f'{out_dir}/view_{i}.npy')
    views.append(v)
    print(f'  view_{i}: shape={v.shape}')

n_items = views[0].shape[0]
dim = views[0].shape[1]
print(f'  n_items={n_items}, dim={dim}')

def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro')**2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro')**2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro')**2
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

def effective_rank(X):
    X_c = X - X.mean(axis=0)
    s = np.linalg.svd(X_c, compute_uv=False)
    p = s / s.sum()
    p = p[p > 1e-10]
    return np.exp(-np.sum(p * np.log(p)))

view_names = ['Description', 'Function', 'Audience', 'Style']
pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
pair_names = [f'{view_names[i]} vs {view_names[j]}' for i,j in pairs]

results = {'dataset': 'Amazon_Grocery_and_Gourmet_Food', 'dim': dim,
           'n_items': n_items, 'stage': 'raw_qwen_hidden',
           'n_trials': N_TRIALS, 'subsample_size': SUBSAMPLE,
           'pairwise_cka': {}, 'effective_rank_single': {},
           'effective_rank_concat_4views': {}}

for pn, (i,j) in zip(pair_names, pairs):
    ckas = []
    for t in range(N_TRIALS):
        idx = np.random.choice(n_items, SUBSAMPLE, replace=False)
        ckas.append(linear_cka(views[i][idx], views[j][idx]))
    results['pairwise_cka'][pn] = {'mean': float(np.mean(ckas)), 'std': float(np.std(ckas))}

for i, name in enumerate(view_names):
    ers = []
    for t in range(N_TRIALS):
        idx = np.random.choice(n_items, SUBSAMPLE, replace=False)
        ers.append(effective_rank(views[i][idx]))
    results['effective_rank_single'][name] = {'mean': float(np.mean(ers)), 'ratio': float(np.mean(ers))/dim}

concat_ers = []
for t in range(N_TRIALS):
    idx = np.random.choice(n_items, SUBSAMPLE, replace=False)
    concat = np.concatenate([v[idx] for v in views], axis=1)
    concat_ers.append(effective_rank(concat))
results['effective_rank_concat_4views'] = {'mean': float(np.mean(concat_ers)), 'ratio': float(np.mean(concat_ers))/(4*dim)}

out_json = '$ROOT/paper_recsys/cka_raw_4096d_grocery.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved: {out_json}')
print(f'Mean pairwise CKA: {np.mean([v[\"mean\"] for v in results[\"pairwise_cka\"].values()]):.4f}')
print(f'Concat ER: {results[\"effective_rank_concat_4views\"][\"mean\"]:.1f} / {4*dim} ({results[\"effective_rank_concat_4views\"][\"ratio\"]*100:.1f}%)')
" >> "$ROOT/logs/grocery_raw_3584d_cka.log" 2>&1

log "Grocery CKA audit done."
log "======== P4 Grocery raw 3584-d complete ========"

log "======== ALL SUPPLEMENT EXPERIMENTS DONE ========"
log "Results:"
log "  Pure LLM GTS-All: $PURELLM_CSV"
log "  Hybrid GTS-All: $HYBRID_CSV"
log "  GTS-Last: $GTS_LAST_CSV"
log "  GTS-Random: logs/supplement_gts_random_seed{42,123,456}_0817.csv"
log "  Grocery CKA: paper_recsys/cka_raw_4096d_grocery.json"
