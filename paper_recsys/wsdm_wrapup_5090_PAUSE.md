# WSDM wrap-up PAUSED — 2026-07-24 12:29:18

## Why
User needs GPU for other work. Training stopped (kill).

## Queue
- Script: `run_5090_wsdm_wrapup_serial.sh`
- Queue log: `logs/wsdm_wrapup_5090_queue.log`
- CSV: `logs/wsdm_wrapup_toys_gts.csv`

## Completed Toys GTS (ID/TF/LLM/MV)

| tag | valid_mrr | test_mrr |
|-----|----------:|---------:|
| wrap_toys_id_lr1e4_do0.1_seed42 | 0* | 0.0070 |
| wrap_toys_tf_lr5e4_do0.3_seed42 | 0.0250 | 0.0116 |
| wrap_toys_llm_lr1e4_do0.1_seed42 | 0.0237 | 0.0111 |
| wrap_toys_mv_lr5e4_do0.3_seed42 | 0.0238 | 0.0102 |
| wrap_toys_id_lr1e4_do0.1_seed2024 | 0* | 0.0070 |
| wrap_toys_tf_lr5e4_do0.3_seed2024 | 0.0254 | 0.0112 |
| wrap_toys_llm_lr1e4_do0.1_seed2024 | 0.0237 | 0.0105 |
| wrap_toys_mv_lr5e4_do0.3_seed2024 | 0.0240 | 0.0112 |
| wrap_toys_id_lr1e4_do0.1_seed2026 | 0* | 0.0069 |
| wrap_toys_tf_lr5e4_do0.3_seed2026 | 0.0245 | 0.0110 |

\* ID-only valid CSV extraction broken (0); test OK.

## Interrupted (incomplete)
- **tag**: `wrap_toys_llm_lr1e4_do0.1_seed2026`
- Phase-A: DONE (valid MRR@10≈0.0025)
- Phase-B: killed mid-run — last log line stage: epoch 40 training (~epoch 39/50)
- Ckpts under `saved/wrap_toys_llm_lr1e4_do0.1_seed2026/`:
  - Phase-A: `SASRecAlignV3-Jul-24-2026_11-26-35.pth`
  - Phase-B mid: `SASRecAlignV3-Jul-24-2026_11-42-32.pth` (not final; do not report)
- Log: `logs/wrap_toys_llm_lr1e4_do0.1_seed2026.log`

## Resume (recommended)
1. Re-run LLM Toys seed=2026 from scratch (same wrapup args).
2. Then MV Toys seed=2026: `wrap_toys_mv_lr5e4_do0.3_seed2026`.
3. Partial Phase-B metrics are not authoritative.

## Not started
- wrap_toys_mv_lr5e4_do0.3_seed2026

