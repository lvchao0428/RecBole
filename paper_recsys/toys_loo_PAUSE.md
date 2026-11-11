# Toys Clean LOO PAUSED — 2026-07-27 13:49:36

## Why
User needs GPU temporarily. Training stopped (kill). Resume later on request.

## Queue
- Script: run_5090_toys_loo_serial.sh
- Queue log: logs/toys_loo_queue.log
- CSV: logs/toys_loo_results.csv

## Completed (authoritative — in CSV)

| tag | valid_mrr | test_mrr |
|-----|----------:|---------:|
| loo_toys_id_seed42 | 0* | 0.0150 |
| loo_toys_tfidf_seed42 | 0.0314 | 0.0248 |
| loo_toys_llm_seed42 | 0.0312 | 0.0251 |
| loo_toys_mv_seed42 | 0.0309 | 0.0243 |
| loo_toys_id_seed2024 | 0* | 0.0158 |
| loo_toys_tfidf_seed2024 | 0.0310 | 0.0252 |
| loo_toys_llm_seed2024 | 0.0313 | 0.0250 |
| loo_toys_mv_seed2024 | 0.0300 | 0.0240 |
| loo_toys_id_seed2026 | 0* | 0.0148 |
| loo_toys_tfidf_seed2026 | 0.0309 | 0.0240 |

* ID-only valid CSV extraction often shows 0; test OK.

## Interrupted (INCOMPLETE — do not report)

- tag: loo_toys_llm_lr1e4_do0.1_seed2026
- Stage when killed: epoch 18 evaluating (Phase-A, ~18/20)
- Ckpt dir: saved/loo_toys_llm_lr1e4_do0.1_seed2026/
- Files: SASRecAlignV3-Jul-27-2026_13-26-52.pth
- Log: logs/loo_toys_llm_lr1e4_do0.1_seed2026.log
- Note: Phase-A mid-run. Partial ckpt kept for reference only; re-run this tag from scratch on resume.

## Resume (recommended)

1. Re-run LLM Toys LOO seed=2026 from scratch (same serial-script args).
2. Then MV Toys LOO seed=2026: loo_toys_mv_lr5e4_do0.3_seed2026.

## Not started
- loo_toys_mv_lr5e4_do0.3_seed2026
