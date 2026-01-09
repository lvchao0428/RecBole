# 实验数据分析报告 (0107.csv & 0108.csv)

**日期**: 2025-01-09  
**实验配置**: phaseA/B temperature 调整为 0.05，无 cold start weighted，无 IPW

---

## 1. 数据总览

### 1.1 Beauty 数据集 (0107.csv, phaseA τ=0.05)

| Model | HR@10 | NDCG@10 | MRR@10 | 备注 |
|-------|-------|---------|--------|------|
| base | 4.69 | 2.69 | 2.07 | baseline |
| tfidf | 5.98 | 3.79 | 3.12 | +27.5% HR |
| tfidf + llm | 6.00 | 3.80 | **3.13** | 略优于 tfidf |
| **multi-view 7B** | 6.05 | 3.82 | **3.14** ✓ | MRR 最高 |
| **multi-view 14B** | **6.22** ✓ | **3.86** ✓ | 3.13 | HR/NDCG 最高 |
| multi-view 32B | 6.05 | 3.81 | 3.12 | 退化 |

### 1.2 Toys 数据集 (0107.csv, phaseA τ=0.05)

| Model | HR@10 | NDCG@10 | MRR@10 | 备注 |
|-------|-------|---------|--------|------|
| base | 5.97 | 3.32 | 2.49 | baseline |
| tfidf | 6.86 | 4.42 | **3.66** | +14.9% HR |
| tfidf + llm | 6.83 ❌ | 4.39 ❌ | 3.64 ❌ | **劣于 tfidf** |
| multi-view 7B | 6.90 | 4.43 | 3.67 | — |
| multi-view 14B | 6.95 | 4.43 | 3.66 ❌ | MRR = tfidf |
| **multi-view 32B** | **7.02** ✓ | **4.47** ✓ | **3.68** ✓ | 全指标最佳 |

### 1.3 Beauty 数据集 (0108.csv, phaseB τ=0.05)

| Model | HR@10 | NDCG@10 | MRR@10 | vs 0107 |
|-------|-------|---------|--------|---------|
| tfidf | 5.93 | 3.75 | 3.09 | HR↓, MRR↓ |
| tfidf + llm | 5.94 | 3.78 | 3.12 | HR↓, MRR≈ |
| multi-view 7B | — | — | — | 进行中 |
| multi-view 14B | 6.02 | 3.79 | 3.11 | HR↓, MRR↓ |
| multi-view 32B | 6.01 | 3.80 | 3.12 | HR↓, MRR= |

**结论**: phaseB τ=0.05 整体比 phaseA τ=0.05 略差

---

## 2. 问题诊断

### 2.1 ❌ 问题1: tfidf + llm 在 Toys 上劣于纯 tfidf

```
Toys MRR@10:  tfidf (3.66) > tfidf+llm (3.64)  → -0.55%
Toys HR@10:   tfidf (6.86) > tfidf+llm (6.83)  → -0.44%
Toys NDCG@10: tfidf (4.42) > tfidf+llm (4.39)  → -0.68%
```

**这是最反常的现象** —— 加 LLM 特征反而全面变差！

**可能原因**:
- LLM embedding 融合权重/方式不对
- Toys 数据集的 LLM embedding 质量问题
- `tfidf + llm` 的具体配置需要检查

### 2.2 ❌ 问题2: MRR scale law 不成立

**Beauty MRR@10**:
```
7B (3.14) > 14B (3.13) > 32B (3.12)  → 反向 scale law
```

**Toys MRR@10**:
```
32B (3.68) > 7B (3.67) > 14B (3.66)  → 不单调，14B 是谷底
```

**可能原因**:
- temperature τ=0.05 可能对大模型不适合
- 固定嵌入预算下，大模型被过度压缩
- 不同 scale 可能需要不同的超参数

### 2.3 ⚠️ 问题3: Beauty HR/NDCG 与 MRR 不同步

| 指标 | 最佳模型 |
|------|---------|
| HR@10 | **14B** (6.22) |
| NDCG@10 | **14B** (3.86) |
| MRR@10 | **7B** (3.14) |

这体现了 **HR--ranking trade-off**，但预期 MRR 和 NDCG 应该同向变化。

### 2.4 ⚠️ 问题4: phaseB τ=0.05 整体退化

对比 0107 (phaseA) 和 0108 (phaseB)，在相同 τ=0.05 下：
- phaseB 的 HR 普遍下降 0.2-0.3%
- phaseB 的 MRR 普遍下降 0.01-0.03

**说明**: phaseA 和 phaseB 可能需要不同的 temperature

---

## 3. 预期 vs 实际对比

### 3.1 预期结果
1. **顺序**: multi-view > single-view (tfidf+llm) > tfidf
2. **MRR scale law**: 7B < 14B < 32B
3. **MRR 与 NDCG 成正比**: 同向变化
4. **某些情形存在 HR--ranking trade-off**

### 3.2 实际达成情况

| 预期 | Beauty | Toys |
|------|--------|------|
| multi-view > single-view | ✓ (微弱) | ✓ |
| single-view > tfidf | ✓ (微弱) | ❌ **反向** |
| MRR scale law | ❌ **反向** | ⚠️ 不单调 |
| MRR ∝ NDCG | ⚠️ 14B 不一致 | ✓ |

---

## 4. 消融实验结果 (0107.csv)

### 4.1 SENet + Cross 消融 (Beauty)

| Configuration | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|---------------|-------|---------|--------|-----------|
| Full (SENet + Cross) | 6.05 | 3.82 | 3.14 | 2.09 |
| − SENet | 6.11 (+1.0%) | 3.82 (=) | 3.11 (−1.0%) | 2.16 |
| − Cross | 6.56 (+8.4%) | 3.64 (−4.7%) | 2.74 (−12.7%) | 2.40 |
| − Both | 6.57 (+8.6%) | 3.63 (−5.0%) | 2.73 (−13.1%) | 2.42 |

**结论**: Cross Network 对排序质量至关重要，移除后 HR↑ 但 MRR/NDCG↓↓

### 4.2 SENet + Cross 消融 (Toys)

| Configuration | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|---------------|-------|---------|--------|-----------|
| Full (SENet + Cross) | 6.90 | 4.43 | 3.67 | 2.30 |
| − SENet | 6.95 (+0.7%) | 4.46 (+0.7%) | 3.69 (+0.5%) | 2.30 |
| − Cross | 7.43 (+7.7%) | 4.13 (−6.8%) | 3.10 (−15.5%) | 2.44 |
| − Both | 7.41 (+7.4%) | 4.13 (−6.8%) | 3.11 (−15.3%) | 2.40 |

**结论**: 与 Beauty 一致，Cross 是排序质量的关键

### 4.3 Whitening 消融 (Beauty)

| Configuration | HR@10 | NDCG@10 | MRR@10 |
|---------------|-------|---------|--------|
| multi-view 7B w/ whiten | 6.05 | 3.82 | 3.14 |
| multi-view 7B w/o whiten | 6.76 (+11.7%) | 3.58 (−6.3%) | 2.60 (−17.2%) |
| tfidf+llm w/ whiten | 6.00 | 3.80 | 3.13 |
| tfidf+llm w/o whiten | 6.01 (+0.2%) | 3.80 (=) | 3.12 (−0.3%) |

**结论**: Whitening 对 multi-view 至关重要，对 single-view 影响很小

### 4.4 Whitening 消融 (Toys)

| Configuration | HR@10 | NDCG@10 | MRR@10 |
|---------------|-------|---------|--------|
| multi-view 7B w/ whiten | 6.90 | 4.43 | 3.67 |
| multi-view 7B w/o whiten | 6.84 (−0.9%) | 4.37 (−1.4%) | 3.61 (−1.6%) |
| tfidf+llm w/ whiten | 6.83 | 4.39 | 3.64 |
| tfidf+llm w/o whiten | 7.00 (+2.5%) | 4.47 (+1.8%) | 3.68 (+1.1%) |

**注意**: Toys 上 tfidf+llm 无 whiten 反而更好！这与 Beauty 不同。

---

## 5. Center-only 归一化消融 (0107.csv)

### 5.1 Beauty

| Configuration | HR@10 | NDCG@10 | MRR@10 |
|---------------|-------|---------|--------|
| tfidf (full whiten) | 5.98 | 3.79 | 3.12 |
| tfidf (center only) | 5.97 | 3.78 | 3.11 |
| tfidf+llm (full whiten) | 6.00 | 3.80 | 3.13 |
| tfidf+llm (center only) | 6.01 | 3.81 | 3.14 |
| multi-view 7B (full whiten) | 6.05 | 3.82 | 3.14 |
| multi-view 7B (center only) | 6.06 | 3.79 | 3.10 |

**结论**: Center-only 与 full whiten 差异很小

### 5.2 Toys

| Configuration | HR@10 | NDCG@10 | MRR@10 |
|---------------|-------|---------|--------|
| tfidf (full whiten) | 6.86 | 4.42 | 3.66 |
| tfidf (center only) | 6.94 | 4.45 | 3.68 |
| tfidf+llm (full whiten) | 6.83 | 4.39 | 3.64 |
| tfidf+llm (center only) | 6.90 | 4.45 | 3.69 |

**注意**: Toys 上 center-only 比 full whiten 更好！

---

## 6. 待确认问题

1. **`tfidf + llm` 和 `multi-view` 的配置差异是什么？**
   - 是否都用了相同的 LLM embedding？
   - 融合方式是否一致（add text weight 等）？

2. **IPW 组件的具体实现是什么？** 计划后续加入？

3. **Temperature 的最优设置**:
   - phaseA 和 phaseB 是否需要不同的 τ？
   - 不同 LLM scale 是否需要不同的 τ？

4. **Toys 上的异常现象**:
   - tfidf+llm < tfidf 的根因
   - center-only > full whiten 的原因

---

## 7. 下一步建议

1. **排查 tfidf+llm 在 Toys 上的退化原因**
   - 检查 LLM embedding 的来源和质量
   - 比较融合权重配置

2. **尝试不同的 temperature 设置**
   - 对 32B 尝试更小的 τ（如 0.03）
   - 对 phaseB 尝试更大的 τ（如 0.07）

3. **加入 cold start weighted 和 IPW**
   - 观察对 scale law 的影响

4. **深入分析 Toys 的 whiten 效果**
   - 为何 center-only 和 no-whiten 在 Toys 上更好

