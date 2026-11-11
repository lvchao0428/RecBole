# 0723 老师指导 — 核心数据验证清单

> 原文：[`0723zhidao.txt`](0723zhidao.txt)  
> 整理：2026-08-02  
> 目的：把老师说的「该验证什么」从叙述拆成可勾选的数据项，并标现状。
>
> **2026-08-02 口径审计更新**：warm-only、UniSRec test 与 LOVO 的 checkpoint
> 重载没有复现训练时的 `min_train_interactions=5`，只能作诊断结果，不能与 min5
> 主表直接混排。最新、完整判定见
> [`0723_full_advisor_report.md`](0723_full_advisor_report.md)。

---

## A. 止损已成立的证据（老师认定可停扩 MV）

这些是**已经够用来止损**的证据，不是还要新开大实验：

| # | 老师原话对应 | 需要的数据 | 现状 | 关键数字 |
|:-:|--------------|------------|:----:|----------|
| A1 | TF / LLM / MV 差异落在方差附近 | Beauty GTS Main 3-seed | ✅ | TF 0.0165±0.0006 ≈ LLM 0.0161 ≈ MV 0.0160 |
| A2 | 两个时间窗口都是 TF ≈ LLM ≥ MV | Main + Early60 | ✅ | Early：TF 0.0053 ≥ MV 0.0052 ≥ LLM 0.0049 |
| A3 | 四 view CKA 很高 | pairwise CKA | ✅ | Beauty 0.839–0.938 |
| A4 | effective rank 只有 136.9/256 | concat eff. rank | ✅ | Beauty 136.9/256 (53.5%)；Toys 141.4/256 |
| A5 | leave-one-view-out ≈ 0 | 有效 LOVO | 🔄 | zero-mask 已有非零 Δ，但旧离线结果未复现 min5；同口径复评待跑 |
| A6 | Cross improved/degraded ≈ 各半 | rank transition | ✅ | TF 47.6/50.9；LLM 49.6/49.2；MV 51.5/47.3 → Appendix |

**止损结论（老师原话落点）**：不再扩 MV / boost / Cross / 追求层级；故事改为  
*严格时间评测下，文本仍明显优于 ID-only，但复杂 LLM/MV/Cross 额外收益不 robust。*

---

## B. 论文三个核心问题 ↔ 必须验证的数据

| 核心问题 | 必须拿到的验证数据 | 现状 |
|----------|-------------------|:----:|
| **Q1** LOO vs leakage-free GTS 会不会改变 text-enhanced SR 的结论 | Clean LOO vs GTS：Beauty+Toys × ID/TF/LLM/MV × 3 seed | ✅ |
| | UniSRec 作 external baseline（至少一代表性数据集的**真 test**） | 🔄 旧 6-ckpt test 未复现 min5；修复脚本已就绪，待复评 |
| **Q2** TF / single-view LLM / MV 谁在 temporal shift 下更鲁棒 | GTS Main 排序 + Early 窗口同向 | ✅ Beauty 双窗口；Toys Main ✅ |
| | valid→test drop（三路是否同量级） | ✅ GTS ~43–55%；LOO ~15–21% |
| **Q3** MV 无增益：容量不足还是 view redundancy | CKA + effective rank | ✅ |
| | leave-one-view-out（有效） | 🔄 zero-mask 完成诊断版；min5 同口径复评待跑 |
| | （可选）drop_view_* 的 eff.rank 作旁证 | ✅ 已有，删一 view 几乎不掉信息 |

Cross：**不是核心问题**，只用 A6 做 negative analysis。

---

## C. 三组收尾实验 — 逐项勾选

### 第一组：Audit unseen target（主表口径）

| 验证项 | 老师要求 | 现状 | 备注 |
|--------|----------|:----:|------|
| C1.1 unseen target 占比 | Test 里有多少 target 不在 train catalog | ✅ | Beauty **27.8%** / Toys **34.5%**（老师说的 17.8% = Beauty 全实例 unseen bucket） |
| C1.2 主表 = warm targets only | 限制 train catalog 内 | 🔄 | 旧 21 ckpt 结果未复现 min5 且缺 Beauty ID；修复 evaluator 已就绪，24 ckpt 待复评 |
| C1.3 unseen 单独报告 | 仅 content-only 模型才和 ID 比 | ⏳ | 我们未做 content-only；可不比，Methods 写清即可 |
| C1.4 Methods 写清协议 | GTS cutoff / target 选择 / candidate catalog / cold filtering / valid·test 构造 | ⏳ | **写作项**，非 GPU 实验 |

### 第二组：干净 protocol 对照（LOO vs GTS）

| 验证项 | 老师要求 | 现状 | 备注 |
|--------|----------|:----:|------|
| C2.1 统一代码 | 修正 freeze / 融合容量 / align | ✅ | per-source align，no-boost，no-Cross |
| C2.2 Beauty Clean LOO × 3 seed | ID/TF/LLM/MV | ✅ | LOO 抬高 ~1.5×，排序翻转为 LLM≈MV≥TF |
| C2.3 第二数据集 Clean LOO | Toys | ✅ | 同向：LLM≈MV > TF |
| C2.4 GTS Main × 3 seed | Beauty + Toys | ✅ | 两数据集都是 **TF ≥ LLM≈MV** |
| C2.5 GTS Early 窗口 | Main/Early 双窗口 | ✅ Beauty / ⏳ Toys Early 可选 | Beauty Early 已够证「两窗口」 |
| C2.6 UniSRec portability | 代表性数据集真 test | 🔄 | 6 ckpt 已有诊断 test，但未复现 min5；新输出将写入 `unisrec_gts_results_min5.csv` |

### 第三组：Mechanism evidence

| 验证项 | 位置 | 现状 | 关键数字 |
|--------|:----:|:----:|----------|
| C3.1 pairwise CKA | Main | ✅ | inter-view 高；TF↔view 仅 0.25–0.33 |
| C3.2 effective rank | Main | ✅ | 4-view concat 53–55%；单 view 近满秩 → 非编码器塌缩 |
| C3.3 leave-one-view-out | Main | 🔄 | zero-mask 修复完成；旧结果为未过滤 cohort，min5 复评待跑 |
| C3.4 popularity / unseen / turnover | Main | ✅ | 见 warm audit + distribution_shift JSON |
| C3.5 valid→test drop | Main | ✅ | 三路 GTS 同掉 ~44%+ → 系统性漂移 |
| C3.6 Cross entropy/margin/transition | Appendix | ✅ | 无稳定趋势，不硬解释 |

### Hyper（附带口径，非第三组实验）

| 验证项 | 老师要求 | 现状 |
|--------|----------|:----:|
| C0 shared valid grid | 小网格排除 underfitting；不为 MV 加宽 | ✅ | 同等预算锁参；三路 drop 一致 |

---

## D. 一页结论：哪些「核心验证」还没落地

### 硬缺口（挡论文闭环）

1. **C1.2 warm-only 最终主表** — 需 min5 + train-catalog candidates，并补 Beauty ID 3 ckpt  
2. **C2.6 UniSRec 真 test** — 修复脚本已同步 5090，待 GPU 空闲后复评 6 ckpt  
3. **C3.3 有效 LOVO** — zero-mask 逻辑有效，待按 min5 cohort 复评 Beauty/Toys  

### 非硬缺口（可不挡投稿主线）

- C1.3 content-only unseen 对照（老师说只有支持才比）  
- C1.4 Methods 文字（写作）  
- C2.5 Toys Early60（Beauty 双窗口已满足「两个时间窗口」）  

### 已经够用的主叙事数据

- Q1：Clean LOO vs GTS 排序翻转 ✅  
- Q2：双窗口 + 跨域 Toys + drop 同量级 ✅  
- Q3：CKA + eff.rank 已独立支撑 redundancy；LOVO 是补强 ✅/🔄  

---

## E. 和老师叙事句的对应（写论文时直接用）

| 叙事句 | 用哪组数据撑 |
|--------|--------------|
| 文本仍明显优于 ID-only | GTS/LOO 主表 ID vs TF/LLM/MV |
| 复杂 LLM/MV 额外收益不 robust | A1+A2；差异落在方差内 |
| 受 temporal shift 限制 | C3.4 + C3.5（unseen↑、drop~44%） |
| 受 view redundancy 限制 | C3.1 + C3.2 + C3.3 |
| LOO 会改变结论/排序 | C2.2–C2.4 |
| Cross 不作为卖点 | A6 → Appendix |
