#!/usr/bin/env bash
# Fix for the eval section of early60eq_retrain_save_0816
# The training saves checkpoints as saved/SASRecAlignV3-Aug-16-2026_*.pth
# and saved/SASRecAlignMultiViewV3-Aug-16-2026_*.pth
# The original eval used  which won't match.
#
# This script finds ALL .pth files created today (Aug 16) and evaluates them.

set -e
cd /home/charlie/project/RecBole
export PATH=/home/charlie/anaconda3/bin:$PATH
export PYTHONPATH=/home/charlie/project/RecBole:${PYTHONPATH:-}

OUT=paper_recsys/early60eq_gts_last_eval.csv
echo "tag,pth,dataset,model,min_train_interactions,eval_mode,n_users,n_warm,test_mrr@10_all,test_mrr@10_warm,test_ndcg@10_all,test_ndcg@10_warm,test_hit@10_all,test_hit@10_warm" > "$OUT"

echo '=== Finding today'\''s Early60-EQ checkpoints ==='
# List all .pth from today with TS [0.6,0.1,0.1]
for pth in $(find saved/ -maxdepth 1 -name '*.pth' -newer logs/early60eq_retrain_save_0816.log | sort); do
  echo "Checking: $pth"
  # Peek inside to check it's Early60-EQ
  info=$(python -c "
import torch
d = torch.load('$pth', map_location='cpu', weights_only=False)
c = d['config']
split = c['eval_args']['split']
ts = split.get('TS', [])
seed = c['seed']
model = c['model']
# Only eval if it's our [0.6,0.1,0.1] split
if ts == [0.6, 0.1, 0.1]:
    print(f'MATCH seed={seed} model={model}')
else:
    print(f'SKIP split={ts}')
" 2>/dev/null | grep -v tensorflow | grep -v oneDNN | grep -v AVX)

  if echo "$info" | grep -q MATCH; then
    echo "  -> $info -> Evaluating GTS-Last warm..."
    # Create a temp directory to hold this pth for eval_gts_last.py
    tmpdir="saved/_eval_tmp_$$"
    mkdir -p "$tmpdir"
    cp "$pth" "$tmpdir/"
    python scripts/eval_gts_last.py --ckpt_dir "$tmpdir" --min_train_inter 5 --out "$OUT" --gpu 0 2>&1 | tail -3
    rm -rf "$tmpdir"
  else
    echo "  -> $info -> Skipping"
  fi
done

echo ''
echo '=== Results ==='
cat "$OUT"
echo ''
echo 'DONE'
