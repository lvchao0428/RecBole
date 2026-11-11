#!/bin/bash
# Evaluate Early60-EQ checkpoints under GTS-Last warm protocol
# This resolves the GTS-All / GTS-Last contradiction flagged by advisor (0816)
# Only eval, no training needed.

set -e
cd /home/a58/project/RecBole
source scripts/recbole_env.sh

OUT=paper_recsys/early60eq_gts_last_eval.csv
echo "tag,pth,dataset,model,min_train_interactions,eval_mode,n_users,n_warm,test_mrr@10_all,test_mrr@10_warm,test_ndcg@10_all,test_ndcg@10_warm,test_hit@10_all,test_hit@10_warm" > "$OUT"

echo "=== Early60-EQ GTS-Last warm evaluation ==="
echo "Looking for Early60-EQ checkpoints..."

# Find Early60-EQ checkpoints for Beauty (TF/LLM/MV × 3 seeds)
# Patterns: *early60*eq* or *early60_eq* or *early60eq*
for ckpt in $(find saved/ -maxdepth 1 -type d -name "*early60*" | sort); do
    echo "Found: $ckpt"
    python scripts/eval_gts_last.py --ckpt_dir "$ckpt" --min_train_inter 5 --out "$OUT" --gpu 0 2>&1
done

echo ""
echo "=== Also checking for checkpoints with 'eq' in name ==="
for ckpt in $(find saved/ -maxdepth 1 -type d -name "*eq*early*" -o -name "*early*eq*" | sort); do
    echo "Found: $ckpt"
    python scripts/eval_gts_last.py --ckpt_dir "$ckpt" --min_train_inter 5 --out "$OUT" --gpu 0 2>&1
done

echo ""
echo "=== Listing ALL saved checkpoints for reference ==="
ls -d saved/*beauty* saved/*Beauty* 2>/dev/null | head -40
echo "---"
ls -d saved/*early* saved/*Early* 2>/dev/null | head -40

echo ""
echo "=== Results ==="
cat "$OUT"
echo ""
echo "DONE. Results in $OUT"
