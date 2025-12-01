#!/usr/bin/env bash
python run_recbole.py \
  --model SASRecAlign \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_baseline_70ep.yaml"

