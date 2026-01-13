# Beauty Scale Law 专项实验方案

> 更新日期：2026-01-13
> 目标：使 Beauty 在 HR_new 指标上展现 Scale Law (7B < 14B < 32B)

---

## 一、当前问题分析

### 1.1 现有实验结果

| 配置 | 7B | 14B | 32B | Scale Law |
|------|-----|-----|-----|-----------|
| **Standard** | 0.0167 | 0.0167 | 0.0166 | ❌ 持平 |
| **Aggressive** | 0.0181 | 0.0182 | 0.0181 | ❌ 持平 |
| **14B infer=2.0** | - | 0.0180 | - | ❌ 反向 |

### 1.2 与 Toys 对比

| 数据集 | 7B | 14B | 32B | Scale Law |
|--------|-----|-----|-----|-----------|
| **Toys (Standard)** | 0.0201 | 0.0202 | 0.0208 | ✅ 成立 |
| **Beauty (Aggressive)** | 0.0181 | 0.0182 | 0.0181 | ❌ 失效 |

### 1.3 问题假设

1. **SVD 压缩假设**：14B/32B 压缩到 64D 信息损失更大
2. **语义饱和假设**：Beauty 语义同质性高，7B 已饱和
3. **参数不匹配假设**：大模型需要不同的 boost 策略

---

## 二、新实验设计

### 2.1 核心思路

> **假设**：大模型语义更丰富，不需要过度放大冷启动权重，而是需要：
> 1. 更强的对齐（alignment_weight ↑）
> 2. 更低的人工放大（cold_boost ↓, infer_boost ↓）

### 2.2 实验矩阵

| 实验 | Model | cold_boost | infer_boost | align_weight | 目的 |
|------|-------|------------|-------------|--------------|------|
| E1 | 14B | 1.5 | 0.5 | 0.15 | 降低放大 + 增强对齐 |
| E2 | 32B | 1.5 | 0.5 | 0.15 | 降低放大 + 增强对齐 |
| E3 | 14B | 2.0 | 0.0 | 0.15 | 纯训练对齐，无推理放大 |
| E4 | 32B | 2.0 | 0.0 | 0.15 | 纯训练对齐，无推理放大 |

### 2.3 预期结果

如果 **E1/E2 的 Hit_new@10** 展现出 32B > 14B > 7B：
- 证明大模型需要"轻推理放大 + 强对齐"策略
- 可以形成参数调优指导

如果仍然持平：
- 证明 Beauty 语义饱和假设成立
- 论文中说明数据集特性导致 Scale Law 差异

---

## 三、实验配置

### 3.1 新参数配置：Low Boost + High Align

```yaml
# 针对大模型 (14B/32B) 的优化配置
cold_start_align_boost: 1.5       # 降低 (原 2.5)
cold_start_align_threshold: 10
inference_cold_text_boost: 0.5    # 大幅降低 (原 1.5)

alignment_weight: 0.15            # 提高 (原 0.10)
temperature: 0.05
```

### 3.2 对照组（保持 7B aggressive 作为基准）

```yaml
# 7B aggressive (已验证最优)
cold_start_align_boost: 2.5
inference_cold_text_boost: 1.5
alignment_weight: 0.10
```

---

## 四、GPU 分配计划

假设 4090 有空闲卡：

| GPU | 实验 | 预计时间 |
|-----|------|----------|
| GPU 4 | E1: 14B low_boost | ~3h |
| GPU 5 | E2: 32B low_boost | ~4h |
| GPU 6 | E3: 14B no_infer | ~3h |
| GPU 7 | E4: 32B no_infer | ~4h |

---

## 五、成功标准

### 5.1 最低标准
- Hit_new@10: 32B > 14B（差距 ≥ 0.0003）

### 5.2 理想标准
- Hit_new@10: 32B > 14B > 7B aggressive
- 即 32B > 0.0182 且 14B > 0.0181

---

## 六、后续计划

如果实验成功：
1. 更新 CHANGELOG_0112.md
2. 整理 Beauty 最优参数文档
3. 完善论文 Scale Law 章节

如果实验失败：
1. 考虑 SVD 自适应压缩实验
2. 或在论文中说明 Beauty 作为反例

---

**状态：待执行**
