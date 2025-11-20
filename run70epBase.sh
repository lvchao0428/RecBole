#!/usr/bin/env bash
# Note: Even though this is ID-only baseline, we still include the stability config
# to ensure consistent model initialization (item_emb_norm will exist but not be used
# since disable_text_feature=true prevents fusion)
python run_recbole.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3.yaml overrides/id_only_70ep.yaml overrides/stability_enhance_with_id_ln.yaml"