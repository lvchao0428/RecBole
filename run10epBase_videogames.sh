#!/usr/bin/env bash
python run_recbole.py \
  --model SASRec_Align \
  --dataset Amazon_Video_Games \
  --config_files "sasrec_align_qwen3.yaml overrides/id_only_10ep.yaml"

