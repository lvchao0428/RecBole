# Pipeline Checkpoint — 手动停止（2026-06-27）

> **状态**: 5090 + log10 训练已 **手动停止** · GPU 空闲 · 可安全断电  
> **策略**: 中断中的 run **丢弃**，恢复时从下方 checkpoint **整 run 重跑**

---

## 5090 — 已完成（保留）

| Seed | Beauty LLM | Beauty MV | Toys LLM | Toys MV |
|------|------------|-----------|----------|---------|
| 42 | ✅ | ✅ | ✅ | ✅ |
| 2024 | ✅ | ✅ | ✅ | ✅ |
| 2025 | ✅ | ✅ | ❌ 已丢弃 | ❌ |
| 2026 | — | — | — | — |

**进度**: 10/16 有效完成（Toys LLM 2025 中途停止，结果不用）

---

## log10 — 已完成（保留）

| Seed | Beauty ID | Beauty TF-IDF | Toys ID | Toys TF-IDF |
|------|-----------|---------------|---------|-------------|
| 42 | ✅ | ✅ | ✅ | ✅ |
| 2024+ | — | ❌ 丢弃 | — | — |

**进度**: 4/16 有效完成（Beauty TF-IDF 2024 中途停止，结果不用）

---

## 恢复命令（重启后各跑一条）

### 5090

```bash
cd ~/project/RecBole
source scripts/recbole_env.sh
nohup bash run_5090_resume_after_poweroff.sh \
  >> logs/resume_after_poweroff_nohup.log 2>&1 &
tail -f logs/resume_after_poweroff_nohup.log
```

**等价于** `run_5090_phase4_text_resume_from_toys2025.sh`：
Toys 2025 LLM+MV → Beauty/Toys 2026 → wait log10 → **Phase 5 UniSRec balanced ×2**

### log10

```bash
cd ~/project/RecBole
source scripts/recbole_env.sh
nohup bash run_log10_resume_after_poweroff.sh \
  >> logs/log10_resume_after_poweroff_nohup.log 2>&1 &
tail -f logs/log10_resume_after_poweroff_nohup.log
```

**等价于** `run_log10_gru4rec_baselines_from_2024.sh`：
seed 2024/2025/2026 · Beauty+Toys · ID+TF-IDF

---

## 可选：清理半成品 checkpoint（非必须）

```bash
# 5090 — 丢弃中断的 Toys LLM 2025
rm -rf saved/gru4rec_tfidf_llm_v3_toys_stratified_seed2025

# log10 — 丢弃中断的 Beauty TF-IDF 2024
rm -rf saved/gru4rec_tfidf_v3_beauty_stratified_seed2024
```

重跑脚本会覆盖 log 追加写入，无需删 log。

---

## 文档

- 总览: [`experiment_status_20260627.md`](experiment_status_20260627.md)
