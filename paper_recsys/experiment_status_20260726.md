# Experiment Status — 2026-07-26

> 更新时间: 16:05（本地）  
> **5090 已恢复 Beauty LOO 训练**

---

## 〇、今日操作时间线

| 时间 | 操作 |
|------|------|
| 00:14 | 恢复 Toys seed=2026 LLM + MV（resume 脚本） |
| 01:27 | Toys LLM seed=2026 完成 (valid=0.0242, test=0.0104) |
| 03:16 | Toys MV seed=2026 完成 (valid=0.0238, test=0.0113)，watcher 自动启动 LOO |
| 03:16–13:07 | Beauty LOO 连续跑完 seed42 全部 + seed2024 前三组（ID/TF/LLM） |
| 13:07 | **暂停**：用户需 GPU（其他项目 train_phase2.py） |
| 15:58 | **恢复** Beauty LOO（MV seed2024 + seed2026 全部 = 5 runs） |
| 16:00 | 三端代码同步完成 |

---

## 一、Toys GTS Main 3-seed（已定稿 ✅）

协议：Global TS · min5 · no-boost · no-Cross · train-only `.ts` 特征  
超参：ID/LLM `lr=1e-4, do=0.1`；TF/MV `lr=5e-4, do=0.3`

### Test MRR@10

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** |
|------|-------:|---------:|---------:|:--------------:|
| ID-only | 0.0070 | 0.0070 | 0.0069 | **0.0070 ± 0.0001** |
| **TF** | **0.0116** | **0.0112** | **0.0110** | **0.0113 ± 0.0003** |
| LLM | 0.0111 | 0.0105 | 0.0104 | 0.0107 ± 0.0004 |
| MV | 0.0102 | 0.0112 | 0.0113 | 0.0109 ± 0.0006 |

### Valid MRR@10

| 模型 | seed42 | seed2024 | seed2026 | mean |
|------|-------:|---------:|---------:|-----:|
| TF | 0.0250 | 0.0254 | 0.0245 | 0.0250 |
| LLM | 0.0237 | 0.0237 | 0.0242 | 0.0239 |
| MV | 0.0238 | 0.0240 | 0.0238 | 0.0239 |

### Toys 结论

- 文本 vs ID-only：**+50~60%** MRR 增益（TF 0.0113 vs ID 0.0070）
- **TF 最优**；LLM ≈ MV，与 Beauty 一致
- Valid→test drop ~55%（系统性 temporal shift）
- MV stop-gate 跨域验证通过

---

## 二、Beauty Clean LOO 进展

协议：Leave-one-out (LS: valid_and_test) · min5 · no-boost · no-Cross · per-source align  
特征：全量 embedding（非 train-only `.ts`）

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
| loo_beauty_mv_seed2024 | — | — | 🔄 训练中（15:58 启动） |
| loo_beauty_id_seed2026 | — | — | 待跑 |
| loo_beauty_tfidf_seed2026 | — | — | 待跑 |
| loo_beauty_llm_seed2026 | — | — | 待跑 |
| loo_beauty_mv_seed2026 | — | — | 待跑 |

### 预估时间

| 任务 | 预估耗时 | 预计完成 |
|------|----------|----------|
| MV seed2024 | ~100 min | ~17:40 |
| ID seed2026 | ~30 min | ~18:10 |
| TF seed2026 | ~70 min | ~19:20 |
| LLM seed2026 | ~70 min | ~20:30 |
| MV seed2026 | ~100 min | ~22:10 |
| **总计** | **~6h** | **~22:10** |

### LOO vs GTS 初步对照（seed42 + seed2024 均值，2/3 seed）

| 模型 | LOO test MRR | GTS test MRR | LOO/GTS |
|------|-------------:|-------------:|:-------:|
| ID-only | 0.0175 | 0.0106 | 1.65× |
| TF | 0.0252 | 0.0165 | 1.53× |
| LLM | 0.0252 | 0.0161 | 1.57× |
| MV | 0.0258 (1 seed) | 0.0160 | 1.61× |

**关键发现**：
1. LOO 下所有模型 MRR 比 GTS 高 ~1.5-1.7×
2. LOO 下 MV 不再最差（seed42 下 MV 最高 0.0258）→ **LOO 改变模型排序**
3. LOO valid→test drop 更小（~20%），因无 temporal shift

---

## 三、Beauty GTS 3-seed（已有，对照用）

| 模型 | test MRR mean ± std | 来源 |
|------|:-------------------:|------|
| ID-only | 0.0106 ± 0.0005 | log10 wrap-up |
| TF | 0.0165 ± 0.0008 | stop-gate 3-seed |
| LLM | 0.0161 ± 0.0005 | stop-gate 3-seed |
| MV | 0.0160 ± 0.0008 | stop-gate 3-seed |

---

## 四、当前队列状态

| 机器 | 状态 | 当前任务 |
|------|------|----------|
| **5090** | GPU 78%, ~5.6 GiB | `loo_beauty_mv_seed2024`（Phase-A） |
| **log10** | 空闲 | 已完成 Beauty ID 3-seed + UniSRec |
| **本机** | 空闲 | 文档整理 |

---

## 五、文件清单

| 路径 | 说明 |
|------|------|
| `run_5090_beauty_loo_resume.sh` | LOO resume 脚本（当前运行中） |
| `logs/beauty_loo_results.csv` | Beauty LOO 结果 CSV |
| `logs/wsdm_wrapup_toys_gts.csv` | Toys GTS 3-seed 完整结果 |
| `logs/beauty_loo_queue.log` | LOO 队列时间线 |
| `paper_recsys/experiment_status_20260726.md` | 本文件 |
