# TS Split 实验结果汇总 (2026-07-12)

> 收集时间: 2026-07-12 08:56
> 日志已同步至本地 `logs/` 和 `logs/log10/`

## 一、夜间流水线完成情况 (5090, 21:11 → 06:39)

| # | 任务 | 状态 | 耗时 |
|---|------|------|------|
| 1 | ID-only min5 (Beauty, seed=2025) | ✅ | 24 min |
| 2 | TF-IDF + cold_boost=3.0 | ✅ | 79 min |
| 3 | TF-IDF+LLM + cold_boost=3.0 | ✅ | 101 min |
| 4 | MV-Align + cold_boost=3.0 | ✅ | 204 min |
| 5 | Toys/Grocery Qwen TS-aware 特征 | ✅ | 2 min |
| 6 | Beauty TF-IDF no-boost (seed=2024,42) | ✅ | 157 min |

log10: 8 个 ID-only 实验全部完成 (Beauty/Toys/Grocery × 3 seeds, 无 min5)

---

## 二、Beauty 主实验对比 (seed=2025, TS split, min5 filter)

### Phase-B 最终 test 结果

| 配置 | cold_boost | test MRR@10 | test NDCG@10 | test R@10 | R_new@10 | R_few@10 | R_freq@10 |
|------|-----------|------------|-------------|----------|---------|---------|----------|
| **ID-only min5** | — | 0.0112 | 0.0151 | 0.0277 | 0.0137 | 0.0386 | 0.0563 |
| TF-IDF no-boost (V2) | 0.0 | 0.0157 | 0.0197 | 0.0326 | 0.0183 | 0.0345 | 0.0719 |
| TF-IDF+LLM no-boost (V2) | 0.0 | 0.0157 | 0.0195 | 0.0320 | 0.0198 | 0.0308 | 0.0717 |
| MV-Align no-boost (V2) | 0.0 | 0.0157 | 0.0198 | 0.0333 | 0.0224 | 0.0324 | 0.0735 |
| **TF-IDF + boost** | 3.0 | **0.0162** | **0.0201** | **0.0328** | 0.0224 | 0.0320 | 0.0722 |
| **TF-IDF+LLM + boost** | 3.0 | 0.0158 | 0.0196 | 0.0319 | **0.0244** | 0.0320 | 0.0682 |
| **MV-Align + boost** | 3.0 | **0.0159** | **0.0201** | **0.0338** | 0.0234 | 0.0332 | **0.0740** |

### 关键观察

1. **ID-only min5 baseline**: test MRR@10=0.0112，明显低于所有 text 模型 (~0.0157-0.0162)
2. **no-boost 下 text 模型几乎无层级**: TF-IDF ≈ LLM ≈ MV (MRR@10 均为 0.0157)
3. **加 boost 后微弱改善**: MV-Align R_freq@10 最高 (0.0740)，TF-IDF MRR@10 微升 (0.0162)
4. **LLM 在 new item 上略优**: TF-IDF+LLM boost R_new@10=0.0244 vs TF-IDF 0.0224

---

## 三、Beauty TF-IDF 3-seed (no-boost, V2, min5)

| seed | test MRR@10 | test NDCG@10 | test R@10 | valid MRR@10 |
|------|------------|-------------|----------|-------------|
| 2024 | 0.0149 | 0.0187 | 0.0310 | 0.0276 |
| 2025 | 0.0157 | 0.0197 | 0.0326 | 0.0281 |
| 42 | 0.0163 | 0.0202 | 0.0332 | 0.0276 |
| **mean** | **0.0156** | **0.0195** | **0.0323** | **0.0278** |

---

## 四、log10 ID-only 结果 (无 min5 过滤, ⚠️ 评测用户群不同)

### Beauty

| seed | test MRR@10 | test NDCG@10 | test R@10 |
|------|------------|-------------|----------|
| 2024 | 0.0155 | 0.0198 | 0.0338 |
| 42 | 0.0156 | 0.0198 | 0.0335 |

### Toys

| seed | test MRR@10 | test NDCG@10 | test R@10 |
|------|------------|-------------|----------|
| 2024 | 0.0187 | 0.0227 | 0.0355 |
| 2025 | 0.0182 | 0.0221 | 0.0349 |
| 42 | 0.0185 | 0.0225 | 0.0354 |

### Grocery

| seed | test MRR@10 | test NDCG@10 | test R@10 |
|------|------------|-------------|----------|
| 2024 | 0.0106 | 0.0139 | 0.0245 |
| 2025 | 0.0107 | 0.0137 | 0.0237 |
| 42 | 0.0108 | 0.0142 | 0.0252 |

---

## 五、特征生成完成

| 数据集 | TF-IDF (.ts.npy) | Single-view Qwen (.ts.npy) | Multi-view (4views_ts/) |
|--------|-----------------|---------------------------|------------------------|
| Beauty | ✅ | ✅ | ✅ |
| Toys | ✅ | ✅ (04:01) | ✅ (04:01) |
| Grocery | ✅ | ✅ (04:01) | ✅ (04:01) |

---

## 六、耗时记录 (5090 Beauty, 实测)

| 配置 | 耗时 |
|------|------|
| ID-only min5 (50ep) | 24 min |
| TF-IDF no-boost A+B | 78 min |
| TF-IDF+LLM no-boost A+B | 101 min |
| MV-Align no-boost A+B | 204 min |
| TF-IDF boost A+B | 79 min |
| TF-IDF+LLM boost A+B | 101 min |
| MV-Align boost A+B | 204 min |
| TF-IDF 3-seed (each) | ~78 min |

---

## 七、本地文件索引

```
logs/
  ts_beauty_id_only_min5_seed2025.log      # ID-only min5
  ts_beauty_tfidf_v2_seed{2024,2025,42}.log  # TF-IDF no-boost 3-seed
  ts_beauty_tfidf_llm_v2_seed2025.log      # TF-IDF+LLM no-boost
  ts_beauty_mv_v2_seed2025.log             # MV-Align no-boost
  ts_beauty_tfidf_boost_seed2025.log       # TF-IDF boost
  ts_beauty_llm_boost_seed2025.log         # TF-IDF+LLM boost
  ts_beauty_mv_boost_seed2025.log          # MV-Align boost
  overnight_20260711.log                   # 流水线总日志
  log10/
    ts_amazon_*_id_only_seed*.log          # log10 ID-only (8 runs)
    ts_id_only_nohup.log                   # log10 调度日志
```
