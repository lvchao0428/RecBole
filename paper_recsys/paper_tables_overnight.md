# Paper Tables Pack — overnight

> Generated: 2026-08-02 17:00  
> **Diagnostic pack only.** Warm-only and UniSRec checkpoint reloads must reproduce `min_train_interactions=5` before they are merged into paper tables.

## Beauty Clean LOO (test MRR@10)

| Model | seed42 | seed2024 | seed2026 | mean ± std |
|-------|-------:|---------:|---------:|-----------:|
| ID | 0.0179 | 0.0171 | 0.0174 | 0.0175 ± 0.0004 |
| TF | 0.025 | 0.0253 | 0.0249 | 0.0251 ± 0.0002 |
| LLM | 0.0251 | 0.0252 | 0.0258 | 0.0254 ± 0.0004 |
| MV | 0.0258 | 0.0251 | 0.025 | 0.0253 ± 0.0004 |

## Toys Clean LOO (test MRR@10)

| Model | seed42 | seed2024 | seed2026 | mean ± std |
|-------|-------:|---------:|---------:|-----------:|
| ID | 0.015 | 0.0158 | 0.0148 | 0.0152 ± 0.0005 |
| TF | 0.0248 | 0.0252 | 0.024 | 0.0247 ± 0.0006 |
| LLM | 0.0251 | 0.025 | 0.0272 | 0.0258 ± 0.0012 |
| MV | 0.0243 | 0.024 | 0.029 | 0.0258 ± 0.0028 |

## Toys GTS (test MRR@10)

| Model | seed42 | seed2024 | seed2026 | mean ± std |
|-------|-------:|---------:|---------:|-----------:|
| ID | 0.007 | 0.007 | 0.0069 | 0.0070 ± 0.0001 |
| TF | 0.0116 | 0.0112 | 0.011 | 0.0113 ± 0.0003 |
| LLM | 0.0111 | 0.0105 | 0.0104 | 0.0107 ± 0.0004 |
| MV | 0.0102 | 0.0112 | 0.0113 | 0.0109 ± 0.0006 |

## UniSRec GTS (min5 test MRR@10)

| Dataset | seed42 | seed2024 | seed2026 | mean |
|---------|-------:|---------:|---------:|-----:|
| Beauty | — | — | — | — |
| Toys | — | — | — | — |

## Warm-only vs All-target (test MRR@10)

| tag | unseen_share | MRR all | MRR warm |
|-----|-------------:|--------:|---------:|
| wrap_beauty_id_lr1e4_do0.1_seed42 | 0.3192 | 0.011160 | 0.016393 |
| wrap_beauty_id_lr1e4_do0.1_seed2024 | 0.3192 | 0.010393 | 0.015267 |
| wrap_beauty_id_lr1e4_do0.1_seed2026 | 0.3192 | 0.010331 | 0.015176 |
| sg_beauty_tf_lr5e4_do0.3_seed42 | 0.3192 | 0.015976 | 0.023468 |
| sg_beauty_tf_lr5e4_do0.3_seed2024 | 0.3192 | 0.016148 | 0.023720 |
| sg_beauty_tf_lr5e4_do0.3_seed2026 | 0.3192 | 0.017369 | 0.025513 |
| sg_beauty_llm_lr1e4_do0.1_seed42 | 0.3192 | 0.015628 | 0.022955 |
| sg_beauty_llm_lr1e4_do0.1_seed2024 | 0.3192 | 0.016068 | 0.023602 |
| sg_beauty_llm_lr1e4_do0.1_seed2026 | 0.3192 | 0.016492 | 0.024225 |
| sg_beauty_mv_lr5e4_do0.3_seed42 | 0.3192 | 0.015906 | 0.023364 |
| sg_beauty_mv_lr5e4_do0.3_seed2024 | 0.3192 | 0.015289 | 0.022459 |
| sg_beauty_mv_lr5e4_do0.3_seed2026 | 0.3192 | 0.016935 | 0.024876 |
| wrap_toys_id_lr1e4_do0.1_seed42 | 0.4078 | 0.007033 | 0.011880 |
| wrap_toys_id_lr1e4_do0.1_seed2024 | 0.4078 | 0.007047 | 0.011900 |
| wrap_toys_id_lr1e4_do0.1_seed2026 | 0.4078 | 0.006883 | 0.011623 |
| wrap_toys_tf_lr5e4_do0.3_seed42 | 0.4078 | 0.011640 | 0.019654 |
| wrap_toys_tf_lr5e4_do0.3_seed2024 | 0.4078 | 0.011237 | 0.018975 |
| wrap_toys_tf_lr5e4_do0.3_seed2026 | 0.4078 | 0.011017 | 0.018603 |
| wrap_toys_llm_lr1e4_do0.1_seed42 | 0.4078 | 0.011105 | 0.018751 |
| wrap_toys_llm_lr1e4_do0.1_seed2024 | 0.4078 | 0.010540 | 0.017798 |
| wrap_toys_llm_lr1e4_do0.1_seed2026 | 0.4078 | 0.010401 | 0.017563 |
| wrap_toys_mv_lr5e4_do0.3_seed42 | 0.4078 | 0.010175 | 0.017181 |
| wrap_toys_mv_lr5e4_do0.3_seed2024 | 0.4078 | 0.011167 | 0.018856 |
| wrap_toys_mv_lr5e4_do0.3_seed2026 | 0.4078 | 0.011340 | 0.019149 |
