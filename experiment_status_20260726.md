# Experiment Status — 2026-07-26

> 更新时间: 16:15（本地）  
> **5090 正在跑 Beauty LOO resume + Toys LOO watcher 已就位**

---

## 〇、今日操作时间线

| 时间 | 操作 |
|------|------|
| 00:14 | 恢复 Toys seed=2026 LLM + MV（resume 脚本） |
| 01:27 | Toys LLM seed=2026 完成 (valid=0.0242, test=0.0104) |
| 03:16 | Toys MV seed=2026 完成 (valid=0.0238, test=0.0113)，watcher 自动启动 LOO |
| 03:16–13:07 | Beauty LOO 连续跑完 seed42 全部 + seed2024 前三组（ID/TF/LLM） |
| 13:07 | **暂停**：用户需 GPU（其他项目 train_phase2.py） |
| 15:58 | **恢复** Beauty LOO resume（MV seed2024 + seed2026 全部 = 5 runs） |
| 16:12 | 部署 Toys LOO watcher，Beauty 完成后自动接续 Toys LOO 12 runs |
| 16:15 | 三端代码/文档同步完成 |

---

## 一、5090 完整执行队列（→ 明天中午）

| 序号 | 脚本 | tag | 预估耗时 | 预计完成 |
|:----:|------|-----|----------|----------|
| 1 | `beauty_loo_resume` | MV seed2024 | ~100 min | ~17:40 |
| 2 | ↑ | ID seed2026 | ~30 min | ~18:10 |
| 3 | ↑ | TF seed2026 | ~70 min | ~19:20 |
| 4 | ↑ | LLM seed2026 | ~70 min | ~20:30 |
| 5 | ↑ | MV seed2026 | ~100 min | **~22:10** |
| | | **Beauty LOO 完成** | | |
| 6 | `toys_loo_serial` | ID seed42 | ~30 min | ~22:40 |
| 7 | ↑ | TF seed42 | ~70 min | ~23:50 |
| 8 | ↑ | LLM seed42 | ~70 min | ~01:00 (7/27) |
| 9 | ↑ | MV seed42 | ~110 min | ~02:50 |
| 10 | ↑ | ID seed2024 | ~30 min | ~03:20 |
| 11 | ↑ | TF seed2024 | ~70 min | ~04:30 |
| 12 | ↑ | LLM seed2024 | ~70 min | ~05:40 |
| 13 | ↑ | MV seed2024 | ~110 min | ~07:30 |
| 14 | ↑ | ID seed2026 | ~30 min | ~08:00 |
| 15 | ↑ | TF seed2026 | ~70 min | ~09:10 |
| 16 | ↑ | LLM seed2026 | ~70 min | ~10:20 |
| 17 | ↑ | MV seed2026 | ~110 min | **~12:10 (7/27)** |
| | | **总计 17 runs** | **~20h** | **7/27 ~12:10** |

> Watcher 脚本 `run_5090_toys_loo_watcher.sh` 监听 Beauty LOO resume 完成后自动启动 Toys LOO 队列。

---

## 二、Toys GTS Main 3-seed（已定稿 ✅）

协议：Global TS · min5 · no-boost · no-Cross · train-only `.ts` 特征

### Test MRR@10

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** |
|------|-------:|---------:|---------:|:--------------:|
| ID-only | 0.0070 | 0.0070 | 0.0069 | **0.0070 ± 0.0001** |
| **TF** | **0.0116** | **0.0112** | **0.0110** | **0.0113 ± 0.0003** |
| LLM | 0.0111 | 0.0105 | 0.0104 | 0.0107 ± 0.0004 |
| MV | 0.0102 | 0.0112 | 0.0113 | 0.0109 ± 0.0006 |

---

## 三、Beauty Clean LOO 进展（7/12 完成 → 今晚定稿）

### 已完成结果

| tag | valid_mrr | test_mrr | 状态 |
|-----|----------:|---------:|:----:|
| loo_beauty_id_seed42 | — | 0.0179 | ✅ |
| loo_beauty_tfidf_seed42 | 0.0311 | 0.0250 | ✅ |
| loo_beauty_llm_seed42 | 0.0306 | 0.0251 | ✅ |
| loo_beauty_mv_seed42 | 0.0309 | 0.0258 | ✅ |
| loo_beauty_id_seed2024 | — | 0.0171 | ✅ |
| loo_beauty_tfidf_seed2024 | 0.0307 | 0.0253 | ✅ |
| loo_beauty_llm_seed2024 | 0.0309 | 0.0252 | ✅ |
| loo_beauty_mv_seed2024 | — | — | 🔄 训练中 |
| loo_beauty_*_seed2026 × 4 | — | — | 待跑 |

### LOO vs GTS 初步对照

| 模型 | LOO test MRR (2 seed) | GTS test MRR (3 seed) | LOO/GTS |
|------|----------------------:|----------------------:|:-------:|
| ID-only | 0.0175 | 0.0106 | 1.65× |
| TF | 0.0252 | 0.0165 | 1.53× |
| LLM | 0.0252 | 0.0161 | 1.57× |
| MV | 0.0258 (1 seed) | 0.0160 | 1.61× |

**关键发现**：LOO 下 MV 不再最差 → **LOO 改变模型排序**

---

## 四、Beauty GTS 3-seed（对照用）

| 模型 | test MRR mean ± std |
|------|:-------------------:|
| ID-only | 0.0106 ± 0.0005 |
| TF | 0.0165 ± 0.0008 |
| LLM | 0.0161 ± 0.0005 |
| MV | 0.0160 ± 0.0008 |

---

## 五、机器状态

| 机器 | 状态 | 当前任务 |
|------|------|----------|
| **5090** | GPU 77%, ~5.6 GiB | `loo_beauty_mv_seed2024` Phase-A → 后续自动接续 |
| **log10** | 空闲 | 已完成 Beauty ID 3-seed |
| **本机** | 空闲 | — |

---

## 六、文件清单

| 路径 | 说明 |
|------|------|
| `run_5090_beauty_loo_resume.sh` | Beauty LOO resume（当前运行中） |
| `run_5090_toys_loo_serial.sh` | Toys LOO 串行队列 |
| `run_5090_toys_loo_watcher.sh` | Watcher：Beauty 完成后启动 Toys |
| `logs/beauty_loo_results.csv` | Beauty LOO 结果 CSV |
| `logs/toys_loo_results.csv` | Toys LOO 结果 CSV（待生成） |
| `logs/wsdm_wrapup_toys_gts.csv` | Toys GTS 3-seed 完整结果 |
| `paper_recsys/experiment_status_20260726.md` | 本文件 |
| `paper_recsys/experiment_plan_20260727.md` | 明日计划 |
