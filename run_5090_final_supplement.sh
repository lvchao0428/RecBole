#!/bin/bash
# Final supplementary experiments per 0806 guidance:
# 1. GTS-Random multi-seed (seeds 42/123/456) — GPU eval
# 2. Pre- vs post-whitening CKA redundancy audit — CPU only
set -euo pipefail
cd /home/charlie/project/RecBole
source scripts/recbole_env.sh

LOG=logs/final_supplement.log
echo "[$(date '+%F %T')] ======== Final supplement START ========" | tee -a $LOG

# ============================================================
# Part 1: GTS-Random with 3 target-sampling seeds
# ============================================================
echo "[$(date '+%F %T')] === Part 1: GTS-Random multi-seed ===" | tee -a $LOG

# Build checkpoint list (same as gts_last_eval.csv tags)
CKPT_LIST=(
  # Beauty
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
  # Toys
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
)

SAMPLE_SEEDS=(123 456)
OUT_DIR=logs

for SSEED in "${SAMPLE_SEEDS[@]}"; do
  OUT_CSV="${OUT_DIR}/gts_random_eval_seed${SSEED}.csv"
  echo "[$(date '+%F %T')] GTS-Random sample_seed=${SSEED} -> ${OUT_CSV}" | tee -a $LOG

  # Write checkpoint list file
  CKPT_FILE="/tmp/ckpt_list_random_${SSEED}.txt"
  printf "%s\n" "${CKPT_LIST[@]}" > "$CKPT_FILE"

  python3 eval_gts_random.py \
    --ckpt_list "$CKPT_FILE" \
    --out "$OUT_CSV" \
    --sample_seed "$SSEED" \
    --min_train_interactions 5 \
    --gpu_id 0 2>&1 | tee -a $LOG

  echo "[$(date '+%F %T')]   Done seed=${SSEED}, rows=$(wc -l < "$OUT_CSV")" | tee -a $LOG
done

# Aggregate multi-seed random into paper_recsys
echo "[$(date '+%F %T')] Aggregating multi-seed random results..." | tee -a $LOG
python3 -c "
import csv, statistics, json
from collections import defaultdict

seeds = [42, 123, 456]
data = defaultdict(lambda: defaultdict(list))  # tag -> metric -> [values across sample_seeds]

for s in seeds:
    with open(f'logs/gts_random_eval_seed{s}.csv') as f:
        for r in csv.DictReader(f):
            tag = r['tag']
            data[tag]['mrr_warm'].append(float(r['test_mrr@10_warm']))
            data[tag]['ndcg_warm'].append(float(r['test_ndcg@10_warm']))
            data[tag]['hit_warm'].append(float(r['test_hit@10_warm']))
            data[tag]['mrr_all'].append(float(r['test_mrr@10_all']))

# Write aggregated CSV
with open('paper_recsys/gts_random_multiseed_agg.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['tag','mrr_warm_mean','mrr_warm_std','ndcg_warm_mean','hit_warm_mean','n_sample_seeds'])
    for tag in sorted(data.keys()):
        d = data[tag]
        w.writerow([
            tag,
            f\"{statistics.mean(d['mrr_warm']):.6f}\",
            f\"{statistics.stdev(d['mrr_warm']):.6f}\" if len(d['mrr_warm'])>1 else '0',
            f\"{statistics.mean(d['ndcg_warm']):.6f}\",
            f\"{statistics.mean(d['hit_warm']):.6f}\",
            len(d['mrr_warm']),
        ])
print('Wrote paper_recsys/gts_random_multiseed_agg.csv')
" 2>&1 | tee -a $LOG

# ============================================================
# Part 2: Pre- vs post-whitening CKA audit
# ============================================================
echo "[$(date '+%F %T')] === Part 2: CKA pre-vs-post-whitening ===" | tee -a $LOG
python3 scripts/cka_pre_vs_post.py 2>&1 | tee -a $LOG

# ============================================================
# Done
# ============================================================
echo "[$(date '+%F %T')] ======== Final supplement COMPLETE ========" | tee -a $LOG
echo "Artifacts:" | tee -a $LOG
echo "  GTS-Random per-seed: logs/gts_random_eval_seed{42,123,456}.csv" | tee -a $LOG
echo "  GTS-Random aggregated: paper_recsys/gts_random_multiseed_agg.csv" | tee -a $LOG
echo "  CKA pre/post: paper_recsys/cka_pre_vs_post_*.json" | tee -a $LOG
