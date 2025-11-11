python scripts/run_plain_grid.py \
  --dataset Amazon_Beauty \
  --config_file sasrec_base_plain.yaml \
  --text_weights 0.4 0.8 \
  --text_gate_init_grid 0.3 0.5 \
  --text_gate_reg_entropy_grid 0.0 0.0005 \
  --text_gate_reg_l2_grid 0.0 0.00001 \
  --text_tail_threshold_grid 0 5 \
  --epochs 20 --eval_step 2 --stopping_step 5 \
  --eval_batch_size 32
