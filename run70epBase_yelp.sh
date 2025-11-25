#!/usr/bin/env bash
python run_recbole.py \
  --model SASRecAlign \
  --dataset yelp \
  --config_files "sasrec_baseline_yelp_example.yaml"

