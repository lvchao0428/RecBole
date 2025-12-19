#!/usr/bin/env bash
cd /home/charlie/project/RecBole
python run_recbole.py \
  --model SASRecAlign \
  --dataset yelp \
  --config_files "yelp_config/sasrec_baseline_yelp_70ep.yaml"

