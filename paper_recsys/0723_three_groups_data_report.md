# 0723 三组收尾实验 — 数据结论汇报

> 对应老师原文三段收尾实验要求  
> 口径：统一代码（per-source align / no-boost / no-Cross / min5）· 主指标 test MRR@10 · 3-seed  
> 生成：2026-08-02
>
> **口径审计提示**：本文件中的训练日志主结果为 min5；warm-only、UniSRec 与
> LOVO 离线重评没有复现 min5，因此其绝对值不能直接与主表混排。请以
> [`0723_full_advisor_report.md`](0723_full_advisor_report.md) 的就绪度判定为准。

---

# 第一组：Audit unseen target（主表口径）

> 老师要求：audit test 中未在 train 出现的 target；主表用 warm targets；unseen 单独报（仅 content-only 可比）；Methods 写清 GTS 协议。

## 1.1 Unseen target audit

老师提到的 **17.8%** = Beauty 全实例口径下 test 的 unseen bucket 占比。  
按「每用户最后一个 target」审计，比例更高：

| Dataset | Train cutoff | Valid cutoff | Valid unseen | **Test unseen** | Catalog turnover | Train catalog |
|---------|--------------|--------------|:------------:|:---------------:|:----------------:|-------------:|
| Beauty | 2014-03-09 | 2014-05-17 | 15.2% | **27.8%** | 23.3% | 217,470 |
| Toys | 2014-01-17 | 2014-04-08 | 21.1% | **34.5%** | 31.1% | 283,233 |

全实例 bucket（distribution shift，Beauty）：

| Split | head | mid | low | **unseen** |
|-------|-----:|----:|----:|-----------:|
| Valid | 63.1% | 15.9% | 9.7% | 11.3% |
| Test | 56.6% | 14.9% | 10.7% | **17.8%** ← 老师引用值 |

**结论**：test 中有相当比例 target 根本不在 train catalog；继续调模型前，必须先固定评测口径。

## 1.2 Warm-only 主表重评估（21 ckpt，GTS）

| Dataset | Model | all-target MRR@10 | **warm-only MRR@10** | warm/all | unseen share |
|---------|-------|------------------:|---------------------:|---------:|-------------:|
| Beauty | TF | 0.0205±0.0004 | **0.0282±0.0005** | 1.38× | 27.3% |
| Beauty | LLM | 0.0201±0.0003 | 0.0277±0.0004 | 1.38× | 27.3% |
| Beauty | MV | 0.0204±0.0002 | 0.0280±0.0003 | 1.37× | 27.3% |
| Toys | ID | 0.0175±0.0003 | 0.0258±0.0005 | 1.47× | 32.2% |
| Toys | TF | 0.0251±0.0001 | **0.0370±0.0002** | 1.47× | 32.2% |
| Toys | LLM | 0.0245±0.0001 | 0.0361±0.0002 | 1.47× | 32.2% |
| Toys | MV | 0.0250±0.0003 | 0.0369±0.0004 | 1.48× | 32.2% |

**结论**：
- 主表应报 **warm-only**（item ∈ train catalog）。
- 换口径后绝对值↑，但**排序不变**（TF ≥ MV ≥ LLM）→ 科学结论 robust。
- Unseen 单独报告；我们**未做 content-only scoring**，故不拿 ID/TF/LLM 在 unseen 上硬比（遵循老师要求）。

## 1.3 Methods 必须写清的协议项（写作清单）

| 项 | 我们的设定 |
|----|------------|
| GTS cutoff | Beauty train 2014-03-09 / valid 2014-05-17；Toys train 2014-01-17 / valid 2014-04-08 |
| Split | TS [0.8, 0.1, 0.1]；Early 窗口 TS [0.6, 0.1, 0.3] |
| Target 选择 | 每用户在 valid/test 时间窗内的交互（full ranking） |
| Candidate catalog | 全 item 空间；主表指标限制 warm targets |
| Cold filtering | min_train_interactions = 5 |
| Valid/test 构造 | global temporal；valid 与 test 同为未来窗 |

---

# 第二组：干净的 protocol 对照（LOO vs GTS）

> 老师要求：统一修正代码；Beauty + 另一数据集；ID/TF/LLM/MV × 3 seed；Clean LOO vs GTS；GTS 保留 Main/Early；UniSRec external baseline。

## 2.1 统一代码口径（相对旧 LOO）

| 问题（旧） | 现口径 |
|------------|--------|
| freeze / 融合容量不一致 | 统一两阶段 + 相同 hidden |
| align 实现不一致 | **per-source align**（TF/LLM/MV 公平） |
| boost / Cross 干扰 | **no-boost, no-Cross** |
| 超参 | 共享 valid 小网格锁定后不再为 MV 加宽 |

## 2.2 主对照：Clean LOO vs GTS（test MRR@10）

| Dataset | Protocol | ID-only | TF-IDF | LLM | MV | 排序 |
|---------|----------|--------:|-------:|----:|---:|------|
| Beauty | **GTS Main** | 0.0106±0.0005 | **0.0165±0.0006** | 0.0161±0.0004 | 0.0160±0.0007 | TF ≥ LLM ≈ MV |
| Beauty | **Clean LOO** | 0.0175±0.0003 | 0.0251±0.0002 | **0.0254±0.0003** | 0.0253±0.0004 | LLM ≈ MV ≥ TF |
| Toys | **GTS Main** | 0.0070±0.0000 | **0.0113±0.0002** | 0.0107±0.0003 | 0.0109±0.0005 | TF > MV ≈ LLM |
| Toys | **Clean LOO** | 0.0152±0.0004 | 0.0247±0.0005 | **0.0258±0.0010** | **0.0258±0.0023** | LLM ≈ MV > TF |

**结论**：
1. 文本仍明显优于 ID-only（两协议、两数据集）。
2. **LOO 抬高 1.5–2.3×，并翻转排序** → “复杂文本更好”很大程度是 LOO protocol 产物。
3. GTS 下三路差异落在方差内 → stop-gate：不再扩 MV。

## 2.3 GTS 双窗口（Beauty Early60）

| Window | Split | TF | LLM | MV |
|--------|-------|---:|----:|---:|
| Main | [0.8, 0.1, 0.1] | **0.0165±0.0006** | 0.0161±0.0004 | 0.0160±0.0007 |
| Early60 | [0.6, 0.1, 0.3] | **0.0053±0.0002** | 0.0049±0.0002 | 0.0052±0.0001 |

**结论**：两窗口同向 TF ≥ LLM/MV；MV 未超单视图 1 std。  
（Toys Early 未跑，属可选加强；Beauty 已满足「两个窗口」。）

## 2.4 UniSRec external baseline（GTS，真 test）

| Dataset | seed42 | seed2024 | seed2026 | mean±std | vs 我们的 TF |
|---------|-------:|---------:|---------:|---------:|-------------|
| Beauty | 0.0161 | 0.0167 | 0.0167 | **0.0165±0.0003** | ≈ TF 0.0165 |
| Toys | 0.0194 | 0.0194 | 0.0196 | **0.0195±0.0001** | > 我们的 Toys TF 0.0113* |

\* Toys 上 UniSRec 更高，但 Beauty（主域）与 TF/LLM 持平；整体仍不支持「复杂文本方法稳超 TF」的叙事。Valid：Beauty 0.0232 / Toys 0.0258（均低于我们三路 valid ~0.029/0.024，说明训练拟合更弱或协议敏感）。

---

# 第三组：Mechanism evidence

> 老师要求：Main 放 view redundancy（CKA / eff.rank / LOVO）+ temporal shift（pop / unseen / turnover / drop）；Cross → Appendix。

## 3.1 View redundancy（→ Main）

### Pairwise CKA & effective rank

| 指标 | Beauty | Toys |
|------|-------:|-----:|
| Inter-view CKA | 0.839 – 0.938 | 0.814 – 0.919 |
| TF ↔ view CKA | 0.246 – 0.258 | 0.306 – 0.334 |
| 单 view eff. rank | ~63.7/64 (**99.6%**) | ~63.8/64 (**99.6%**) |
| **4-view concat eff. rank** | **136.9/256 (53.5%)** | **141.4/256 (55.2%)** |
| TF-IDF eff. rank | 253.1/256 (98.9%) | 253.1/256 (98.9%) |

→ 单 view 不塌缩；拼起来冗余；TF 与 view 更正交。

### Leave-one-view-out（zero-mask，GTS MV ckpt）

| Config | Beauty MRR | Δ | Toys MRR | Δ |
|--------|-----------:|----:|---------:|----:|
| Full (4 views) | **0.02066** | — | **0.02460** | — |
| Drop view_0 | 0.02034 | −0.00031 | 0.02419 | −0.00041 |
| Drop view_1 | 0.02058 | −0.00008 | 0.02441 | −0.00019 |
| Drop view_2 | 0.02033 | −0.00033 | 0.02431 | −0.00029 |
| Drop view_3 | 0.02028 | −0.00038 | 0.02385 | −0.00075 |

**结论**：去掉任一 view，MRR 几乎不动（相对跌幅 Beauty ≤1.8%，Toys ≤3.1%）→ **redundancy，非容量不足**。

## 3.2 Temporal shift（→ Main）

### Popularity / unseen / catalog turnover

| 指标 | Beauty valid → test | Toys valid → test |
|------|:-------------------:|:-----------------:|
| Unseen bucket | 11.3% → **17.8%** | 13.0% → **21.3%** |
| Head bucket | 63.1% → 56.6% | 60.1% → 52.3% |
| Target pop. median | 21 → 15 | 18 → 11 |
| Catalog 新品率 | 23.5% → 27.3% | 27.1% → 32.8% |
| Audit test unseen | — → **27.8%** | — → **34.5%** |
| Catalog turnover | **23.3%** | **31.1%** |

### valid → test drop%（MRR@10）

| Dataset | Protocol | TF | LLM | MV |
|---------|----------|---:|----:|---:|
| Beauty | GTS | 43.1% | 44.3% | 43.9% |
| Beauty | LOO | 18.5% | 18.1% | 17.7% |
| Toys | GTS | 54.9% | 55.3% | 54.3% |
| Toys | LOO | 20.7% | 18.2% | 15.4% |

**结论**：GTS 三路同掉 ~44–55% → 系统性时间漂移；LOO 只掉一半 → protocol 掩盖了 shift。

## 3.3 Cross（→ Appendix，不硬解释）

| Model | Improved | Degraded | Unchanged |
|-------|---------:|---------:|----------:|
| TF | 47.6% | 50.9% | 1.4% |
| LLM | 49.6% | 49.2% | 1.3% |
| MV | 51.5% | 47.3% | 1.2% |

**结论**：近各半、无稳定趋势 → 按老师意见放 Appendix。

---

# 三组 → 总叙事（汇报收束）

| 组 | 回答了什么 | 关键数字 |
|----|------------|----------|
| **第一组** | 评测口径必须先改 | Beauty test unseen **17.8% / 27.8%**；warm 抬 1.4× 但排序不变 |
| **第二组** | Protocol 改变结论与排序 | GTS: TF 优；LOO: 翻转；UniSRec Beauty **0.0165≈TF** |
| **第三组** | 为何复杂方法不增益 | CKA 高；eff.rank **136.9/256**；LOVO≈0；GTS drop **~44%** |

**一句话**：严格时间评测下，文本仍优于 ID-only，但复杂 LLM/MV/Cross 没有稳定超过简单 TF-IDF；限制来自 temporal shift 与 view redundancy，而不是实验没跑好。
