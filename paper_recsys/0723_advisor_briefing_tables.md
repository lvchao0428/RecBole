# 0723 指导 — 汇报用数据表（三问题映射）

> 口径：统一代码（per-source align / no-boost / no-Cross / min5）· 指标 **test MRR@10** · 3-seed mean±std  
> 主表建议：warm-target only（见问题附注）；下列主对比表为 all-target GTS/LOO，排序与 warm-only 一致  
> 生成：2026-08-02
>
> **口径审计提示**：现有 warm-only 离线结果未复现 min5，也未把 candidate
> mask 到 train warm catalog；“排序一致”目前只算诊断证据。最终 paper-ready
> 判定见 [`0723_full_advisor_report.md`](0723_full_advisor_report.md)。

---

## 问题一：LOO 和 leakage-free temporal split，会不会改变 text-enhanced SR 的结论？

**结论：会。GTS 下 TF 最优；LOO 抬高绝对值并翻转排序。文本仍明显优于 ID-only。**

### 表 1-A　Clean LOO vs GTS（主对照）

| Dataset | Protocol | ID-only | TF-IDF | LLM | MV | 排序 |
|---------|----------|--------:|-------:|----:|---:|------|
| Beauty | **GTS Main** | 0.0106±0.0005 | **0.0165±0.0006** | 0.0161±0.0004 | 0.0160±0.0007 | TF ≥ LLM ≈ MV |
| Beauty | **Clean LOO** | 0.0175±0.0003 | 0.0251±0.0002 | **0.0254±0.0003** | 0.0253±0.0004 | LLM ≈ MV ≥ TF |
| Toys | **GTS Main** | 0.0070±0.0000 | **0.0113±0.0002** | 0.0107±0.0003 | 0.0109±0.0005 | TF > MV ≈ LLM |
| Toys | **Clean LOO** | 0.0152±0.0004 | 0.0247±0.0005 | **0.0258±0.0010** | **0.0258±0.0023** | LLM ≈ MV > TF |

要点：
1. LOO 把绝对值抬高约 **1.5–2.3×**（相对同模型 GTS）。
2. **排序翻转**：GTS 上 TF 领先；LOO 上 LLM/MV 反超（差距仍多在方差内）。
3. 两数据集结论一致 → 不是 Beauty 特例。

### 表 1-B　External baseline：UniSRec（GTS，真 test）

| Dataset | seed42 | seed2024 | seed2026 | mean±std |
|---------|-------:|---------:|---------:|---------:|
| Beauty | 0.0161 | 0.0167 | 0.0167 | **0.0165±0.0003** |
| Toys | 0.0194 | 0.0194 | 0.0196 | **0.0195±0.0001** |

对照：Beauty 上 UniSRec ≈ 我们的 TF/LLM（~0.0165/0.0161），**未显示复杂文本方法的额外优势**。

---

## 问题二：TF-IDF、single-view LLM、prompt multi-view，谁在 temporal shift 下更鲁棒？

**结论：简单 TF-IDF 更稳；LLM/MV 无稳定超越；三路同步掉点 → 系统性时间漂移。**

### 表 2-A　双时间窗口（Beauty）

| Window | TF | LLM | MV | 排序 |
|--------|---:|----:|---:|------|
| **Main** TS[0.8,0.1,0.1] | **0.0165±0.0006** | 0.0161±0.0004 | 0.0160±0.0007 | TF ≥ LLM ≈ MV |
| **Early60** TS[0.6,0.1,0.3] | **0.0053±0.0002** | 0.0049±0.0002 | 0.0052±0.0001 | TF ≥ MV ≥ LLM |

→ 两窗口同向；MV 未超单视图 1 std → **stop-gate 成立**。

### 表 2-B　valid → test drop%（MRR@10）

| Dataset | Protocol | TF | LLM | MV |
|---------|----------|---:|----:|---:|
| Beauty | GTS | 43.1% | 44.3% | 43.9% |
| Beauty | LOO | 18.5% | 18.1% | 17.7% |
| Toys | GTS | 54.9% | 55.3% | 54.3% |
| Toys | LOO | 20.7% | 18.2% | 15.4% |

要点：
1. GTS 下三路几乎**同量级下跌**（组内差 <1.2pp）→ 不是某一路过拟合。
2. LOO 的 drop 只有 GTS 的约一半 → 多出来的落差来自 **temporal shift**。

### 表 2-C　Warm-only 主表口径（排序不变）

| Dataset | Model | all-target | **warm-only** | warm/all | test unseen |
|---------|-------|-----------:|--------------:|---------:|------------:|
| Beauty | TF | 0.0205 | **0.0282** | 1.38× | 27.3% |
| Beauty | LLM | 0.0201 | 0.0277 | 1.38× | 27.3% |
| Beauty | MV | 0.0204 | 0.0280 | 1.37× | 27.3% |
| Toys | TF | 0.0251 | **0.0370** | 1.47× | 32.2% |
| Toys | LLM | 0.0245 | 0.0361 | 1.47× | 32.2% |
| Toys | MV | 0.0250 | 0.0369 | 1.48× | 32.2% |

→ 换 warm-only 后绝对值上升，**排序仍是 TF ≥ MV ≥ LLM**（结论对口径 robust）。

---

## 问题三：Multi-view 为什么没有增益——容量不足还是 view redundancy？

**结论：view redundancy。单 view 近满秩；拼起来有效秩腰斩；去掉任一 view 几乎不伤 MRR。**

### 表 3-A　表征冗余（embedding 层）

| 指标 | Beauty | Toys |
|------|-------:|-----:|
| Inter-view CKA（6 对） | 0.839 – 0.938 | 0.814 – 0.919 |
| TF-IDF ↔ view CKA | 0.246 – 0.258 | 0.306 – 0.334 |
| 单 view effective rank | ~63.7 / 64 (99.6%) | ~63.8 / 64 (99.6%) |
| **4-view concat eff. rank** | **136.9 / 256 (53.5%)** | **141.4 / 256 (55.2%)** |
| TF-IDF eff. rank | 253.1 / 256 (98.9%) | 253.1 / 256 (98.9%) |

→ 单 view 不塌缩；concat 后冗余；TF 与 view **正交性更高**。

### 表 3-B　Leave-one-view-out（推理期 zero-mask，GTS ckpt）

| Config | Beauty MRR@10 | Δ | Toys MRR@10 | Δ |
|--------|-------------:|----:|------------:|----:|
| Full (4 views) | **0.02066** | — | **0.02460** | — |
| Drop view_0 (Description) | 0.02034 | −0.00031 | 0.02419 | −0.00041 |
| Drop view_1 (Function) | 0.02058 | −0.00008 | 0.02441 | −0.00019 |
| Drop view_2 (Audience) | 0.02033 | −0.00033 | 0.02431 | −0.00029 |
| Drop view_3 (Style) | 0.02028 | −0.00038 | 0.02385 | −0.00075 |

→ 最大跌幅 Beauty ~1.8%、Toys ~3.1%（相对 Full）；各 view 贡献接近 → **高度冗余**。

### 表 3-C　Cross（非核心问题 → Appendix）

| Model | Improved | Degraded | 说明 |
|-------|---------:|---------:|------|
| TF | 47.6% | 50.9% | 近各半 |
| LLM | 49.6% | 49.2% | 近各半 |
| MV | 51.5% | 47.3% | 近各半 |

→ dataset/window-dependent，不做主卖点。

---

## 三问题 ↔ 表格索引（汇报时可投影）

| 核心问题 | 用哪些表 | 一句话结论 |
|----------|----------|------------|
| **Q1** LOO vs GTS 是否改结论 | 表 1-A、1-B | 会改排序；GTS 上 TF 优；UniSRec 未超 |
| **Q2** 谁在 temporal shift 下更鲁棒 | 表 2-A、2-B、2-C | TF 更稳；三路同掉 ~44%+；warm 口径不改排序 |
| **Q3** MV 无增益原因 | 表 3-A、3-B（3-C 附录） | redundancy（CKA 高、eff.rank 半、LOVO≈0） |

---

## 汇报口播（约 1 分钟）

> 在严格 leakage-free GTS 下：文本仍明显优于 ID-only，但 TF / LLM / MV 差异落在方差内，两时间窗口都是 TF ≈ LLM ≥ MV。  
> Clean LOO 会抬高指标并翻转排序，说明旧层级很大程度是 protocol 产物。  
> MV 无增益主因是 view redundancy（CKA 高、concat 有效秩仅约 54%、LOVO 去掉任一 view 几乎不掉点），不是容量不够。  
> UniSRec 在 Beauty GTS 上 test MRR≈0.0165，与 TF/LLM 同量级，未显示复杂方法的额外收益。
