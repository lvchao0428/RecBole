# GRU4Rec Phase 4 结果矩阵

> 生成: 2026-06-29 19:19:59 · 主机: charlie-ultra

## 5090 — LLM + MV

| Seed | B-LLM | B-MV | T-LLM | T-MV |
|------|-------|------|-------|------|
| 42 | ✅ | ✅ | ✅ | ✅ |
| 2024 | ✅ | ✅ | ✅ | ✅ |
| 2025 | ✅ | ✅ | ✅ | ✅ |
| 2026 | ✅ | ✅ | ✅ | ✅ |

## log10 / 5090 — ID + TF-IDF

| Seed | B-ID | B-TF | T-ID | T-TF | 机器(推断) |
|------|------|------|------|------|------------|
| 42 | ✅ | ✅ | ✅ | ✅ | log10 |
| 2024 | ✅ | ✅ | ✅ | ✅ | 5090 |
| 2025 | ✅ | ✅ | ✅ | ✅ | 5090 |
| 2026 | ✅ | ✅ | ✅ | ✅ | 5090 |

## 计数
- 5090 text (LLM+MV): **16/16**
- baseline (ID+TF-IDF): **16/16**
- 合计 checkpoint 块: **32/32**

## 日志路径
- 5090 text: `logs/gru4rec_*_text_seed*.log`
- log10 baseline: `logs/log10_gru4rec_*.log`
- 5090 baseline: `logs/5090_gru4rec_*.log`
- 汇总: `logs/post_main_pipeline_20260625.log`
