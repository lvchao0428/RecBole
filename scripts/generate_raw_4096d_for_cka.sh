#!/usr/bin/env bash
# Generate raw 4096-d Qwen2.5-7B embeddings (no SVD projection) for CKA audit.
# Run AFTER training completes (needs GPU).
# Expected time: ~20-30 min for Beauty + Toys, 4 views each.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

QWEN_MODEL="/home/charlie/project/qwen/Qwen2.5-7B-Instruct"

LOG="$ROOT/logs/raw_4096d_generation.log"
echo "[$(date '+%F %H:%M:%S')] Starting raw 4096-d embedding generation" | tee "$LOG"
echo "  Model: $QWEN_MODEL" | tee -a "$LOG"

for DATASET in Amazon_Beauty Amazon_Toys_and_Games Amazon_Grocery_and_Gourmet_Food; do
  MAPPING="$ROOT/dataset/${DATASET}/item_index_mapping.csv"
  OUT_DIR="$ROOT/dataset/${DATASET}/qwen2.5_7b_4views_raw_4096d"
  mkdir -p "$OUT_DIR"

  echo "[$(date '+%F %H:%M:%S')] Generating raw 4096-d 4-view for $DATASET" | tee -a "$LOG"

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
    --config "$ROOT/sasrec_baseline_50ep_stratified_ts.yaml" \
    >> "$LOG" 2>&1

  echo "[$(date '+%F %H:%M:%S')] Done $DATASET. Files in $OUT_DIR:" | tee -a "$LOG"
  ls -la "$OUT_DIR"/*.npy 2>/dev/null | tee -a "$LOG"
done

echo "[$(date '+%F %H:%M:%S')] Now running CKA on raw 4096-d views..." | tee -a "$LOG"

# Run CKA audit on the raw 4096-d views
python3 -c "
import json
import numpy as np
from pathlib import Path

ROOT = Path('$ROOT')

def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-12))

def effective_rank(X):
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

N_TRIALS = 3
SUBSAMPLE = 3000
view_names = ['Description', 'Function', 'Audience', 'Style']

for ds in ['Amazon_Beauty', 'Amazon_Toys_and_Games', 'Amazon_Grocery_and_Gourmet_Food']:
    raw_dir = ROOT / 'dataset' / ds / 'qwen2.5_7b_4views_raw_4096d'
    view_files = [raw_dir / f'view_{i}.npy' for i in range(4)]
    if not all(f.exists() for f in view_files):
        print(f'[{ds}] SKIP: raw 4096-d views not found')
        continue

    views = [np.load(str(f), mmap_mode='r') for f in view_files]
    n_items, dim = views[0].shape
    print(f'[{ds}] Loaded 4 views: {n_items} items × {dim}-d')

    cka_pairs = {}
    er_single = {v: [] for v in view_names}
    er_concat = []

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(42 + trial * 7)
        idx = rng.choice(n_items, size=min(SUBSAMPLE, n_items), replace=False)
        idx.sort()
        subs = [np.array(v[idx], dtype=np.float32) for v in views]

        for i in range(4):
            for j in range(i+1, 4):
                key = f'{view_names[i]} vs {view_names[j]}'
                cka_pairs.setdefault(key, []).append(linear_cka(subs[i], subs[j]))
            er_single[view_names[i]].append(effective_rank(subs[i]))

        er_concat.append(effective_rank(np.concatenate(subs, axis=1)))

    result = {
        'dataset': ds, 'dim': dim, 'n_items': n_items,
        'stage': 'raw_4096d_pre-SVD',
        'n_trials': N_TRIALS, 'subsample_size': SUBSAMPLE,
        'pairwise_cka': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in cka_pairs.items()},
        'effective_rank_single': {k: {'mean': float(np.mean(v)), 'ratio': float(np.mean(v))/dim} for k, v in er_single.items()},
        'effective_rank_concat_4views': {'mean': float(np.mean(er_concat)), 'ratio': float(np.mean(er_concat))/(dim*4)},
    }

    out_path = ROOT / 'paper_recsys' / f'cka_raw_4096d_{ds.lower().replace(\"amazon_\",\"\").replace(\"_and_\",\"_\")}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Wrote {out_path}')
    avg_cka = np.mean([v['mean'] for v in result['pairwise_cka'].values()])
    print(f'  Avg pairwise CKA = {avg_cka:.4f}')
    print(f'  ER(concat 4×{dim}) = {result[\"effective_rank_concat_4views\"][\"mean\"]:.1f} (ratio={result[\"effective_rank_concat_4views\"][\"ratio\"]:.3f})')
" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %H:%M:%S')] ALL DONE" | tee -a "$LOG"
