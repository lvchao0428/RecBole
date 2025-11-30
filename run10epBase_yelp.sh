#!/usr/bin/env bash

# 10-epoch baseline run for Yelp (reuses Yelp config from run70epBase_yelp.sh).
python run_recbole.py \
  --model SASRec_Align \
  --dataset yelp \
  --config_files "yelp_config/sasrec_baseline_yelp_70ep.yaml overrides/id_only_10ep.yaml"

