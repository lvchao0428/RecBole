# 0803 指导落实汇报：核心论点 · 关键数据 · 章节规划

> 供老师把关。依据：`0803zhidao.txt`  
> 日期：2026-08-06  
> 投稿意向：WSDM 2027 Main（Findings 兜底）  
> 工作标题（待定其一）：  
> **When More Text Is Not Better: A Leakage-Aware Study of Text Representations for Sequential Recommendation**

---

## 一、核心论点（请老师确认方向）

### 1.1 一句话

文本增强序列推荐里，“更强的 LLM / 更多 prompt view 一定更好”在严格时间评测下**不成立**；评测协议会改变 TF-IDF、单 view LLM 与 multi-view 的相对结论，而 prompt multi-view 无增益可用表示冗余解释。

### 1.2 不是什么 / 是什么

| 不再主张 | 改为主张 |
|----------|----------|
| 提出稳定更强的新模型 MV-Align | 做 leakage-aware 的**受控比较研究** |
| Cross / boost 是核心贡献 | Cross 仅 negative appendix |
| 主结论建立在 LOO 上 | 主结论建立在 **GTS-Last（+Random）+ warm + train-cutoff 特征** |
| “LLM/MV 全面优于 TF” | **文本 ≫ ID** 稳定；TF/LLM/MV **相对排序不 robust** |

### 1.3 三个研究问题（RQ）

1. **RQ1（Protocol）**：从 Clean LOO 切到 GTS，是否改变 TF / LLM / MV 的排序与绝对量级？  
2. **RQ2（Temporal robustness）**：Main vs Early 时间窗下，谁更稳？分布漂移与 valid→test 下跌如何同时出现？  
3. **RQ3（Why no MV gain）**：prompt multi-view 无稳定增益，是容量不足还是 view 冗余？

### 1.4 切入点（相对已有 LOO vs GTS 文献）

已有工作研究过 split 本身；我们的贡献点是：  
**协议变化具体改变了哪一类文本表征的结论（TF vs LLM vs prompt-MV）**，并用 CKA + effective rank + LOOV 解释为何“更多语义 view ≠ 更多有效信息”。

---

## 二、老师关心的数据（按 0803 三问）

> 主协议锁定：**GTS-Last**，warm（target∈train catalog），`min_train=5`，特征仅 train-cutoff 拟合。  
> Successive-All / Clean LOO 仅作对照或附录。  
> 标注：✅ 已齐 · ⏳ 队列中（今晚）· † 附录参考

### 2.1 RQ1 — Protocol effect

**现象（老师原述）**：GTS 与 LOO 排序不同；LOO 绝对值明显抬高。

#### 表 A. Clean LOO vs GTS-All† vs GTS-Last（MRR@10，3-seed mean，Beauty/Toys）

| Dataset | Protocol | ID | TF | LLM | MV |
|---------|----------|---:|---:|----:|---:|
| Beauty | Clean LOO | 0.0175 | 0.0251 | **0.0254** | 0.0253 |
| Beauty | GTS-All† | 0.0106 | **0.0165** | 0.0161 | 0.0160 |
| Beauty | **GTS-Last** | 0.0022 | 0.0036 | 0.0041 | **0.0042** |
| Toys | Clean LOO | 0.0152 | 0.0247 | **0.0258** | **0.0258** |
| Toys | GTS-All† | 0.0070 | **0.0113** | 0.0107 | 0.0109 |
| Toys | **GTS-Last** | 0.0019 | 0.0025 | 0.0022 | **0.0026** |

**读法（拟写进论文）**  
- LOO 相对 GTS-Last 绝对值抬升约一个数量级，且偏向 LLM≈MV。  
- GTS-All† 上 TF 略强；GTS-Last 上 Beauty 变为 MV≈LLM≳TF——**细排序随 target 定义而变**，但“文本变体差距近 seed 方差、无稳定层级”一致。  
- 协议身份已确认：原训练日志 GTS = Successive-All；主表改 Last。

#### 表 B. 主表候选：GTS-Last + Warm（推荐主报告）

| Dataset | ID | TF | LLM | MV | UniSRec |
|---------|---:|---:|----:|---:|--------:|
| Beauty warm | 0.0032 | 0.0051 | 0.0059 | **0.0061** | 0.0052 |
| Toys warm | 0.0031 | 0.0040 | 0.0035 | **0.0042** | 0.0041 |

排序：Beauty MV≈LLM≳TF≈UniSRec≫ID；Toys MV≈UniSRec≈TF>LLM≫ID。  
**GTS-Random**：⏳ 同 ckpt 重评中，完成后与 Last 并列。

---

### 2.2 RQ2 — Temporal robustness

**现象（老师原述）**：Main/Early 中 TF 最稳；valid→test 跌 43–55%；与 unseen/head/pop/turnover 同现（associated with，非因果）。

#### 表 C. Beauty Main vs Early60（GTS-All† test MRR@10；超参冻结）

| Window | Split | TF | LLM | MV |
|--------|-------|---:|----:|---:|
| Main | [0.8,0.1,0.1] | **0.0165±0.0006** | 0.0161±0.0004 | 0.0160±0.0007 |
| Early60 | [0.6,0.1,0.3] | **0.0053±0.0002** | 0.0049±0.0002 | 0.0052±0.0001 |

→ 两窗均 TF≥LLM/MV；差距不超过约 1 个 seed std。  
⏳ Early 的 Last/Random 重评已排入队列（现有 `ew_beauty_*` ckpt，不重训）。

#### 表 D. valid→test MRR drop（%）

| Dataset | Protocol | TF | LLM | MV |
|---------|----------|---:|----:|---:|
| Beauty | GTS | **43.1** | 44.3 | 43.9 |
| Beauty | LOO（对照） | 18.5 | 18.1 | 17.7 |
| Toys | GTS | **54.9** | 55.3 | 54.3 |
| Toys | LOO（对照） | 20.7 | 18.2 | 15.4 |

→ GTS 三路同步大跌；LOO 跌幅约一半 → 更像系统性时间漂移，而非单路过拟合。

#### 表 E. 分布漂移（valid → test，formal JSON）

| 指标 | Beauty | Toys |
|------|--------|------|
| Unseen target 占比 | 11.3% → **17.8%** | 13.0% → **21.3%** |
| Head bucket 占比 | 63.1% → 56.6% | 60.1% → 52.3% |
| Target pop. 中位数 | 21 → 15 | 18 → 11 |
| Catalog 新品率 | 23.5% → **27.3%** | 27.1% → **32.8%** |

写作口径：与性能下降 **associated with**，不作因果断言。

---

### 2.3 RQ3 — Why multi-view does not help

**现象（老师原述）**：CKA / ER / LOOV 三角支持 prompt 冗余；TF 与 LLM 更互补；勿单靠 CKA。

#### 表 F. 表示冗余（3 次 item 子样本 mean；Beauty / Toys）

| 证据 | Beauty | Toys |
|------|-------:|-----:|
| Inter-view CKA（4 prompt） | **0.84–0.94** | **0.82–0.92** |
| TF–LLM view CKA | **≈0.20** | **≈0.19** |
| Concat-4 effective rank | **136/256 (53%)** | **141/256 (55%)** |
| TF-IDF effective rank | 250/256 (98%) | 251/256 (98%) |

（旧全量 TF–view CKA 约 0.25–0.33；子样本略低，定性一致。）

#### 表 G. Leave-one-view-out（zero-mask，min5，seed42）

| Dataset | Full MRR | 最大 \|Δ\| | 相对下降 |
|---------|---------:|----------:|---------:|
| Beauty | 0.01591 | 0.00034 | **≈2.1%** |
| Toys | 0.01018 | 0.00019 | **≈1.9%** |

四 view 全量展示；相对下降落在老师所说 0.4%–3.1% 范围。

#### 表 H. 容量/成本（排除“没给够参数”）

Beauty：ID/TF/LLM/MV ≈ **67.16 / 67.43 / 67.56 / 67.71** M，推理延迟同量级。

**综合表述（拟采用）**：CKA 单独不证明无用；CKA + ER + LOOV 共同支持 prompt views 高结构冗余；lexical TF 与 semantic LLM 更互补，但当前 MV 并未利用该互补——故“更多同质语义 view ≠ 更多有效信息”。

---

### 2.4 协议与主文材料清单（0803“只做三件事”）

| 要求 | 状态 |
|------|------|
| 主表 GTS-Last / Random + warm + cutoff-fit | Last+warm ✅；Random ⏳ |
| All / unseen → Appendix | ✅ 规划如此 |
| LOO 仅 protocol contrast | ✅ |
| 主文：LOO vs GTS、Main/Early、UniSRec、shift、CKA+ER+LOOV 图 | 数据✅；版面组装中 |
| Warm/per-seed/Cross/unseen → 附录 | ✅ |
| CKA 3-subsample；LOOV 四 view；成本表 | ✅ |
| 不扩 4×256 / boost / 救 Cross | ✅ |
| Grocery 第三域（Amazon，有 timestamp） | ⏳ 对齐协议重训后 Last/Random |

---

## 三、文章章节规划（约 8 页，ACM sigconf）

> 结构对齐常规长文；**Method → Study Design**（不做新模型卖点）。  
> 页数预算：Intro 1.1 · Related 0.9 · Study Design 1.7 · Setup 1.0 · Results 2.6 · Conclusion 0.35 · 薄附录 0.5–1。

| 章节 | 内容要点 | 主浮件 |
|------|----------|--------|
| **Abstract** | 协议敏感 + 三发现 + 冗余解释 | — |
| **§1 Introduction** | 直觉（LLM/多view更好）脆弱；切入点=文本表征排序；RQ1–3；贡献三条 | 可选 teaser：LOO vs GTS flip |
| **§2 Related Work** | 文本增强序列推荐；时间评测/泄漏；多视角与表示相似性；**定位：受控研究非新 SOTA** | — |
| **§3 Study Design** | LOO / GTS-All / **Last / Random**；leakage controls；ID/TF/LLM/MV/UniSRec；共享 SASRec + two-phase；明确无 Cross 主方法 | **Fig: protocol overview**（待画） |
| **§4 Experimental Setup** | Beauty/Toys（+Grocery 若完成）；指标；超参冻结；复现 | Tab: dataset / hyper |
| **§5 Results** | | |
|  5.1 RQ1 | 表 B 主表 + 表 A 对照；UniSRec | Fig: protocol contrast（已有） |
|  5.2 RQ2 | 表 C–E；associated with | 可选 drop 条形图 |
|  5.3 RQ3 | 表 F–H | Fig: CKA+ER（已有） |
|  5.4 Takeaways | 实践三条：用 Last/Random；先试 TF；单 view 常够 | — |
| **§6 Conclusion** | 重述；Limitations（域、编码器、Successive 全量） | — |
| **Ethics + Refs** | 不计页限 | — |
| **Appendix** | GTS-All；Random 细节；unseen；Cross negative；per-seed | — |

### 主文 vs 附录（按老师要求）

**主文**：LOO vs GTS 聚合、GTS Main/Early、UniSRec、temporal-shift、CKA heatmap+ER+LOOV、成本。  
**附录**：warm 细节扩展、per-seed、Cross rank transition、unseen audit、All-target 全表。

---

## 四、请老师拍板的问题

1. **主表协议**：以 **GTS-Last + warm** 为主报告，Random 并列（跑完后），All 仅附录——是否同意？  
2. **核心叙事**：强调“排序不 robust + prompt 冗余”，而非“TF 永远第一”——Last 上 Beauty 实为 MV≈LLM≳TF，这一读法是否合适？  
3. **第三域**：Grocery（Amazon，有 timestamp）按同一协议补一轮是否需要进主文，还是附录即可？  
4. **篇幅/投递**：按约 8 页写满 Main；Findings 兜底——是否按此推进？  
5. **题目**：现标题是否可定，或更倾向 *Do LLM Text Views Survive Temporal Evaluation?*

---

## 五、当前进度（供老师知情）

| 事项 | 状态 |
|------|------|
| 三 RQ 方向结论与机制三角 | ✅ 可写 |
| GTS-Last 全表（Beauty/Toys + UniSRec） | ✅ |
| 初稿 `main_wsdm2027.tex`（主表已切 Last） | ✅ ~4 页，扩至 ~8 页中 |
| GTS-Random；Early Last/Random；Grocery | ⏳ 5090 夜间队列 |
| 研究设计总览图 | 待画 |

---

*以上数据和规划基于 0803 指导整理，请老师审阅核心论点与主表口径；确认后按此定稿扩写。*
