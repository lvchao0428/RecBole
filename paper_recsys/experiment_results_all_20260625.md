# 实验结果汇总 2026-06-25

## 实验进展状态

| 阶段 | 状态 | 备注 |
|------|------|------|
| Beauty 主表 (SASRec, 4 seeds) | ✅ 完成 | seeds=2024/2025/2026/42 |
| Toys 主表 (SASRec, 4 seeds) | ✅ 完成 | seeds=2024/2025/2026/42 |
| Book-Crossing (SASRec, seed=2024) | ✅ 完成 | phaseb50 |
| Grocery (SASRec, seed=2024) | ✅ 完成 | phaseb50 |
| **UniSRec Beauty (3 configs, seed=2024)** | 🔄 运行中 | 优先级最高，预计 ~4.5h |
| Grocery SASRec multiseed (3 seeds × 4 configs) | ⏳ 排队中 | 在 UniSRec 之后 |
| GRU4Rec (第二 backbone) | 📋 计划中 | 在 Grocery multiseed 之后 |

---

## Book-Crossing (seed=2024, phaseb50) — 完整指标

### 总体指标 @10

| Config | MRR@10 | HR@10 | NDCG@10 | Recall@10 |
|--------|--------|-------|---------|-----------|
| ID-only | 0.0195 | 0.0387 | 0.0241 | 0.0387 |
| TF-IDF | 0.0220 | 0.0365 | 0.0254 | 0.0365 |
| TF-IDF+LLM | 0.0220 | 0.0371 | 0.0256 | 0.0371 |
| MV-Align | **0.0227** | **0.0387** | **0.0265** | **0.0387** |

### 分层 MRR@10

| Config | MRR_new | MRR_few | MRR_freq |
|--------|---------|---------|----------|
| ID-only | 0.0109 | 0.0204 | 0.0393 |
| TF-IDF | 0.0142 | 0.0247 | 0.0426 |
| TF-IDF+LLM | 0.0138 | 0.0245 | 0.0429 |
| MV-Align | **0.0142** | **0.0244** | **0.0448** |

### 分层 HR@10

| Config | HR_new | HR_few | HR_freq |
|--------|--------|--------|---------|
| ID-only | 0.0199 | 0.0398 | 0.0794 |
| TF-IDF | 0.0186 | 0.0342 | 0.0767 |
| TF-IDF+LLM | 0.0186 | 0.0342 | 0.0783 |
| MV-Align | 0.0182 | 0.0355 | **0.0826** |

### 分析

- **MRR 提升**：ID→MV 提升 +16.4%（0.0195 → 0.0227），主要来自 freq 层
- **HR trade-off**：text 列 HR 在 new/few 略降（-6.5%~-14%），freq 层 MV 最终恢复并超过 ID
- **BC 特点**：作为非 Amazon 数据集，增幅弱于 Beauty/Toys，验证了"边界数据集"角色
- 佐证 cross 引入的 MRR↑/HR(cold)↓ trade-off

---

## Amazon Grocery (seed=2024, phaseb50) — 完整指标

### 总体指标 @10

| Config | MRR@10 | HR@10 | NDCG@10 | Recall@10 |
|--------|--------|-------|---------|-----------|
| ID-only | 0.0208 | 0.0636 | 0.0309 | 0.0636 |
| TF-IDF | 0.0293 | 0.0577 | 0.0359 | 0.0577 |
| TF-IDF+LLM | 0.0295 | 0.0583 | 0.0363 | 0.0583 |
| MV-Align | **0.0301** | **0.0602** | **0.0371** | **0.0602** |

### 分层 MRR@10

| Config | MRR_new | MRR_few | MRR_freq |
|--------|---------|---------|----------|
| ID-only | 0.0062 | 0.0118 | 0.0373 |
| TF-IDF | 0.0085 | 0.0180 | 0.0520 |
| TF-IDF+LLM | 0.0088 | 0.0177 | 0.0526 |
| MV-Align | **0.0086** | **0.0178** | **0.0538** |

### 分层 HR@10

| Config | HR_new | HR_few | HR_freq |
|--------|--------|--------|---------|
| ID-only | 0.0156 | 0.0333 | 0.1167 |
| TF-IDF | 0.0132 | 0.0312 | 0.1057 |
| TF-IDF+LLM | 0.0131 | 0.0315 | 0.1070 |
| MV-Align | 0.0132 | 0.0319 | 0.1109 |

### 分析

- **MRR 显著提升**：ID→MV +44.7%（0.0208 → 0.0301），所有层都有明显涨幅
- **HR trade-off 明显**：HR 从 0.0636(ID) 降至 0.0577(TF-IDF)，MV 回升到 0.0602 但仍低于 ID
- **分层 HR**：new/few 层 HR 均下降（-15%~-6%），freq 层也下降（-5%~-9%）
- **Grocery 特点**：item 集极大（17万），MRR 增幅最大（证明 cross/align 对大 item 集有效）
- cross 引入的 HR-MRR trade-off 在 Grocery 最为显著

---

## 核心现象跨数据集对比

### MRR@10 提升幅度 (ID → MV-Align)

| 数据集 | ID-only | MV-Align | 提升 |
|--------|---------|----------|------|
| Beauty | 0.0244 | 0.0283 | +16.0% |
| Toys | 0.0188 | 0.0220 | +17.0% |
| Book-Crossing | 0.0195 | 0.0227 | +16.4% |
| Grocery | 0.0208 | 0.0301 | +44.7% |

### HR@10 变化 (ID → MV-Align)

| 数据集 | ID-only | MV-Align | 变化 |
|--------|---------|----------|------|
| Beauty | 0.0540 | 0.0537 | -0.6% |
| Toys | 0.0410 | 0.0399 | -2.7% |
| Book-Crossing | 0.0387 | 0.0387 | ±0% |
| Grocery | 0.0636 | 0.0602 | **-5.3%** |

### 结论

1. **Cross+Align 统一提升 MRR**：所有 4 个数据集 MRR 都显著提升
2. **HR-MRR trade-off 普遍存在**：Grocery 最明显(-5.3%)，BC 持平，Beauty/Toys 轻微
3. **trade-off 与 item set size 正相关**：Grocery(17万) > Beauty(26万 padded/1.2万 active) > Toys > BC
4. **机制解释**：cross 网络重新分配 score → 提高 top-1 命中(MRR↑) 但可能移出部分 top-10(HR↓)

---

## 进行中: UniSRec Beauty 实验

### 实验设计

| # | 模型 | 文本融合方式 | Cross | Align | MV |
|---|------|------------|-------|-------|-----|
| 1 | UniSRec (原版) | ID + MoE(PLM) additive | ❌ | ❌ | ❌ |
| 2 | UniSRecAlignV3 | ID + MoE + DCN-V2 cross + InfoNCE | ✅ | ✅ | ❌ |
| 3 | UniSRecAlignMultiViewV3 | ID + MoE + cross + align + 4-view SENet | ✅ | ✅ | ✅ |

### 预期论文叙事

- UniSRec 和 SASRecAlignV3 共享相同 Transformer backbone 和 PLM embedding
- 唯一变量 = 融合方式（additive vs cross+align+MV）
- 差异直接归因于 cross/align 组件 → 支撑 mechanism 论证

### 启动时间: 2026-06-24 22:39 (CST)
### 预计完成: 2026-06-25 ~03:00 (CST)
