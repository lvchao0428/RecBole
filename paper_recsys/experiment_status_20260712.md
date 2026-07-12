# 实验进展 — 2026-07-12

> 最后更新: 2026-07-12 22:52
> 指导依据: 老师 0712 建议 (WSDM 改为 leakage-aware controlled study)

---

## 一、今日完成事项

### 1. 特征重新生成 (V3, 13:28–13:35, 5090)

所有三数据集的文本特征严格按 TS train cutoff 重新生成，与老师「vocabulary/IDF、SVD、center/whiten 全部只在 train cutoff 前拟合」要求完全对齐：

| 数据集 | TF-IDF (vocab+IDF+SVD+center+whiten) | Qwen single-view (center+whiten) | Qwen multi-view × 4 |
|--------|:------------------------------------:|:--------------------------------:|:-------------------:|
| Beauty | ✅ 73s | ✅ | ✅ |
| Toys | ✅ 83s | ✅ | ✅ |
| Grocery | ✅ 47s | ✅ | ✅ |

关键改进：TF-IDF 新增 ZCA-whiten（之前只做 L2），统计量只用 train items，每份附 `_whiten_stats.npz` 可审计。

### 2. 实验队列启动

- **5090** (`run_5090_queue_20260712.sh`): 13:35 启动，Beauty Phase-2/3 + Toys/Grocery Phase-4
- **log10** (`run_log10_id_only_all_min5.sh`): 13:42 启动，全数据集 ID-only 3-seed min5

### 3. V1 "打平异常"根因排查

老师关注的 TF-IDF+LLM=0.0042 已确认：`--config_dict` 中 `false` 未被 `ast.literal_eval` 识别为 `False`，导致 Phase-A freeze_backbone=True，模型严重欠训练。V2 已修复（`re.sub` 预处理）。

之前报告 0.0157 也是读日志错误（取了早期 epoch 的 valid 值，非 Phase-B 最终 test 值）。

### 4. Beauty 实验完整结果确认 (V3 特征, TS, min5, no-boost)

层级关系清晰，MV > LLM > TF >> ID，seed 方差极小：

| 模型 | mean HR@10 | mean R@10 | mean MRR@10 | mean NDCG@10 |
|------|:----------:|:---------:|:-----------:|:------------:|
| ID-only (3 seed) | 0.0452 | 0.0452 | 0.0190 | 0.0252 |
| TF-IDF (3 seed) | 0.0533 | 0.0533 | 0.0276 | 0.0336 |
| TF-IDF+LLM (3 seed) | 0.0543 | 0.0543 | 0.0278 | 0.0340 |
| MV-Align (2 seed, 42 pending) | ~0.0549 | ~0.0549 | ~0.0282 | ~0.0344 |

完整分层数据见下方详表。

---

## 二、Beauty 完整结果详表 (test @10)

### 全局指标

| 模型 | seed | HR@10 | R@10 | MRR@10 | NDCG@10 |
|------|:----:|:-----:|:----:|:------:|:-------:|
| ID-only | 2024 | 0.0456 | 0.0456 | 0.0190 | 0.0253 |
| ID-only | 2025 | 0.0451 | 0.0451 | 0.0190 | 0.0252 |
| ID-only | 42 | 0.0450 | 0.0450 | 0.0190 | 0.0251 |
| **ID-only** | **mean** | **0.0452** | **0.0452** | **0.0190** | **0.0252** |
| TF-IDF | 2024 | 0.0524 | 0.0524 | 0.0273 | 0.0332 |
| TF-IDF | 2025 | 0.0540 | 0.0540 | 0.0279 | 0.0340 |
| TF-IDF | 42 | 0.0534 | 0.0534 | 0.0276 | 0.0337 |
| **TF-IDF** | **mean** | **0.0533** | **0.0533** | **0.0276** | **0.0336** |
| TF-IDF+LLM | 2024 | 0.0545 | 0.0545 | 0.0277 | 0.0339 |
| TF-IDF+LLM | 2025 | 0.0545 | 0.0545 | 0.0280 | 0.0342 |
| TF-IDF+LLM | 42 | 0.0538 | 0.0538 | 0.0278 | 0.0339 |
| **TF-IDF+LLM** | **mean** | **0.0543** | **0.0543** | **0.0278** | **0.0340** |
| MV-Align | 2024 | 0.0546 | 0.0546 | 0.0281 | 0.0343 |
| MV-Align | 2025 | 0.0552 | 0.0552 | 0.0282 | 0.0345 |
| MV-Align | 42 | *(~01:00 完成)* | | | |
| **MV-Align** | **mean** | **~0.0549** | | **~0.0282** | **~0.0344** |

### 分层 Recall@10 (item frequency: new/few/frequent)

| 模型 | seed | R_new@10 | R_few@10 | R_freq@10 |
|------|:----:|:--------:|:--------:|:---------:|
| ID-only | mean | 0.0135 | 0.0273 | 0.0839 |
| TF-IDF | mean | 0.0152 | 0.0257 | 0.1022 |
| TF-IDF+LLM | mean | 0.0155 | 0.0276 | 0.1034 |
| MV-Align | 2024+2025 | 0.0159 | 0.0259 | 0.1055 |

---

## 三、Toys ID-only 结果 (log10, seed=2024)

| HR@10 | R@10 | MRR@10 | NDCG@10 | R_new | R_few | R_freq |
|:-----:|:----:|:------:|:-------:|:-----:|:-----:|:------:|
| 0.0343 | 0.0343 | 0.0143 | 0.0190 | 0.0155 | 0.0372 | 0.0640 |

---

## 四、当前运行状态 (22:52)

### 5090 队列

| 任务 | 状态 | 完成时间 |
|------|:----:|:-------:|
| Beauty LLM seed=2024 | ✅ | 15:15 |
| Beauty LLM seed=42 | ✅ | 16:56 |
| Beauty MV seed=2024 | ✅ | 20:20 |
| **Beauty MV seed=42** | 🔄 | ~01:00 (7/13) |
| MV no-Cross ablation (seed=2025) | ⏳ | ~04:20 |
| Toys ID-only min5 | ⏳ | ~04:44 |
| Toys TF-IDF | ⏳ | ~06:02 |
| Toys TF-IDF+LLM | ⏳ | ~07:43 |
| Toys MV-Align | ⏳ | ~11:07 |
| Grocery ID-only min5 | ⏳ | ~11:31 |
| Grocery TF-IDF | ⏳ | ~12:49 |
| Grocery TF-IDF+LLM | ⏳ | ~14:30 |
| Grocery MV-Align | ⏳ | ~17:54 |

### log10 队列

| 任务 | 状态 | 完成时间 |
|------|:----:|:-------:|
| Beauty ID-only × 3 seeds | ✅ | ~19:09 |
| Toys ID-only seed=2024 | ✅ | 21:31 |
| **Toys ID-only seed=2025** | 🔄 | ~00:10 (7/13) |
| Toys ID-only seed=42 | ⏳ | ~02:30 |
| Grocery ID-only × 3 seeds | ⏳ | ~10:00 |

> log10 完成后自动 rsync → 5090 `logs/log10/`

---

## 五、明天 (7/13) 待完成实验

| 时间 | 事项 |
|------|------|
| 早上 | MV-Align seed=42 结果确认，补全 Beauty 3-seed 均值 |
| 早上 | Cross ablation 结果出炉，判断是否保留 Cross 模块 |
| 上午 | Toys 4-config (TF/LLM/MV) 陆续完成 |
| 下午 | Grocery 4-config 完成 |
| 下午 | log10 Grocery ID-only 3-seed 完成，rsync 到 5090 |
| 晚上 | 汇总 Toys/Grocery 结果，更新进展文档 |

---

## 六、待安排实验 (下一阶段)

| 优先级 | 任务 | 前置条件 | 机器 |
|--------|------|---------|------|
| P5 | Beauty TF-IDF 3-seed 完整均值±std 报告 | MV seed=42 ✅ → 明早 | — |
| P5 | Toys/Grocery 3-seed text 扩展 | 7/13 Toys/Grocery seed=2025 完成后 | 5090 |
| P5 | UniSRec portability: Base→+Align→+MV | Toys/Grocery 结束后 | 5090 |
| P5 | ID-only Toys/Grocery 3-seed 完整均值 | log10 7/13 完成后 | — |
| P6 | 机制分析: rank-transition, score entropy, Top-K head/tail | 所有 checkpoint | 本机 |
| P7 | WSDM 叙事重构 + 正式表格 | P5 全部完成 | — |

---

## 七、老师指导要点 (0712)

1. **叙事转变**: TF-IDF 是 strong baseline，复杂文本模型边际收益收缩 = WSDM finding，不追求漂亮层级
2. **item frequency 分桶改名**: low/mid/head，不再叫 cold-start
3. **模型简化**: SE 已删，cold_boost/infer_boost 从主方法移除，two-phase 只作 optimization protocol
4. **Cross 待验**: 若 no-Cross ablation 无显著差异 → 从最终模型移除
5. **机制分析**: 先定义假设再验证；Cross 做 per-user rank-transition matrix + score entropy；Align 做 view redundancy + leave-one-view-out
6. **UniSRec portability + 第三数据集**: 必须在新 eval protocol 下重跑
7. **评测标准**: global TS split + min_train_interactions=5（非严格 5-core）

---

## 八、结果收集路径

```
5090: /home/charlie/project/RecBole/logs/
  ts_beauty_{tfidf,llm,mv}_v2_seed{2024,2025,42}.log    ← Beauty text 模型
  ts_beauty_id_only_min5_seed2025.log                   ← Beauty ID-only (5090)
  ts_beauty_mv_nocross_v2_seed2025.log                  ← Cross ablation
  ts_{toys,grocery}_{id_only,tfidf,llm,mv}_*seed2025*   ← Toys/Grocery
  log10/
    ts_{beauty,toys,grocery}_id_only_min5_seed*.log     ← ID-only 3-seed (log10)
```
