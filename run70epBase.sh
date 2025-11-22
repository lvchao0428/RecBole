#!/usr/bin/env bash
python run_recbole.py \
  --model SASRec \
  --dataset Amazon_Beauty \
  --config_files "sasrec_baseline_70ep.yaml"