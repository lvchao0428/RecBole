# 0717 指导落实汇报 — 2026-07-22 晚

> 依据：`paper_recsys/0717zhidao.txt` 最后两段（stop-gate + WSDM 主线/机制）  
> 协议（固定）：Beauty · Global TS · full ranking · min train history≥5 · train-only TF-IDF/SVD/whiten · 无 SE / 无 cold&infer boost · per-source align · two-phase 仅作优化协议  
> 超参（valid 共享网格锁定）：TF `lr=5e-4, do=0.3` · LLM `lr=1e-4, do=0.1` · MV `lr=5e-4, do=0.3`

---

## 0. 今晚想跟老师对齐的三句话

1. **Stop-gate 已触发**：Beauty 上 Main + Early 两窗口、各 3 seeds，MV 相对 TF/LLM **不能稳定超过 1 个 std** → **MV 不作主方法**。  
2. **Cross 已可删默认**：Main 上无稳定 MRR/NDCG 增益；Early 上 LLM±Cross 持平；机制上无清晰「换覆盖换排序」叙事。  
3. **主线可按 0717 L11 定稿**；机制清单只差 **UniSRec portability**（另有 2 项定稿向补齐）。

---

## 1. Stop-gate（0717 倒数第二段）— 结论：条件齐，可正式关

### 1.1 条件对照

| 老师要求 | 状态 | 证据 |
|----------|:----:|------|
| 修正 train-only TF-IDF/SVD/whitening | ✅ | Main `.ts`；Early 另建 `.ts_early60`（train items≈118k） |
| Beauty 3 seeds | ✅ | Main: 42/2024/2026；Early: 2025/42/2024 |
| ≥2 temporal windows | ✅ | Main `TS[0.8,0.1,0.1]` + Early `TS[0.6,0.1,0.3]` |
| MV 不能超单视图 1 std → 停主方法 | ✅ | 见下表 |
| Cross 不能稳提 NDCG/MRR → 删默认 | ✅ | §2 |
| 删 SE / cold boost；two-phase 只作协议 | ✅ | 当前训练脚本统一关闭 |

### 1.2 Main 窗口 — test MRR@10（3-seed）

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** |
|------|-------:|---------:|---------:|---------------:|
| **TF** | 0.0160 | 0.0161 | 0.0174 | **0.0165 ± 0.0008** |
| **LLM** | 0.0156 | 0.0161 | 0.0165 | **0.0161 ± 0.0005** |
| **MV** | 0.0159 | 0.0153 | 0.0169 | **0.0160 ± 0.0008** |
| MV* do=0.5（附带） | 0.0154 | 0.0156 | 0.0154 | 0.0155 ± 0.0001 |

- MV − LLM = **−0.0001** ≪ 1×LLM_std(0.0005)  
- MV − TF = **−0.0005** ≪ 1×TF_std(0.0008)  
→ **Main：TF ≈ LLM ≥ MV，MV 未过 gate**

### 1.3 Early 窗口 — test MRR@10（3-seed，冻结 Main 超参）

| 模型 | seed2025 | seed42 | seed2024 | **mean ± std** |
|------|---------:|-------:|---------:|---------------:|
| **TF** | 0.0052 | 0.0056 | 0.0052 | **0.0053 ± 0.0002** |
| **MV** | 0.0051 | 0.0051 | 0.0053 | **0.0052 ± 0.0001** |
| **LLM** | 0.0051 | 0.0050 | 0.0047 | **0.0049 ± 0.0002** |

Valid（同窗口，供对照）：三路均在 **0.032–0.034**，TF 略高。

- Early 绝对值远低于 Main（test 段更长 / cutoff 更早），**比的是排序与方差，不是跨窗口绝对数值**  
- MV − TF = **−0.0001**；相对 TF/LLM 均 **未拉开 1 std**  
- 单 seed 排序会抖（如 2024: MV>TF>LLM），但 **3-seed mean 仍无 MV 显著增益**

### 1.4 两窗口合读（对老师可说）

| 窗口 | 排序（test MRR mean） | MV 过 1-std gate？ |
|------|----------------------|:-----------------:|
| Main | TF ≈ LLM ≥ MV | ❌ |
| Early | TF ≥ MV ≈ LLM | ❌ |

→ **Stop-gate 正式触发：停止把 MV-Align 当主方法，转为 leakage-aware controlled study。**

---

## 2. Cross（同段要求）— 结论：删默认，保留负结果+机制

### 2.1 Main 公平对比（seed=2025，no-boost，per-source）

| 模型 | no-Cross MRR | +Cross MRR | Δ MRR | Δ R@10 | Δ R_new |
|------|:------------:|:----------:|:-----:|:------:|:-------:|
| TF | 0.0162 | 0.0164 | +0.0002 | −0.0021 | −0.0041 |
| LLM | 0.0164 | 0.0162 | −0.0002 | −0.0023 | +0.0026 |
| MV | 0.0158 | 0.0156 | −0.0002 | −0.0004 | +0.0015 |

→ **无稳定 MRR/NDCG 净收益**；偶发覆盖/新 item 波动不足以支撑默认打开 Cross。

### 2.2 Early 窗口：LLM ±Cross（seed=2025）

| | valid MRR | test MRR | valid NDCG | test NDCG |
|--|----------:|---------:|-----------:|----------:|
| no-Cross | 0.0324 | **0.0051** | 0.0389 | 0.0061 |
| +Cross | 0.0320 | **0.0051** | 0.0384 | 0.0061 |

→ Early 上 **完全持平**，未出现「依赖窗口才涨」的正向证据。

### 2.3 机制补全（Main ckpt，offline）

**LLM**（≈3000 users）：rank 改善 49.6% / 变差 49.2% ≈ 对半；entropy +0.001；Top1–Top10 margin −0.04。

| Top-10 head share | no-Cross | +Cross | Δ |
|-------------------|:--------:|:------:|------:|
| LLM | 0.665 | 0.667 | +0.002 |
| TF | 0.697 | 0.683 | −0.014 |
| MV | 0.742 | 0.672 | −0.070 |

→ Cross **没有**稳定抬高 head 占比；MV 上反而略降。  
**决定**：Cross **从默认模型删除**，论文中作负结果 + score/rank 机制分析。

---

## 3. WSDM 主线叙事（0717 最后一段）— 证据映射

老师建议主线：

> 严格 GTS 下静态文本仍显著改善 ID-only；TF-IDF 与 single-view LLM 已解释大部分收益；prompt multi-view 边际价值受视角冗余、时间漂移、item frequency 限制；Cross 对 score distribution 的作用依赖数据与窗口，并非稳定增益。

| 论点 | 数据支撑 | 定稿状态 |
|------|----------|:--------:|
| 文本 ≫ ID-only | Beauty TS+min5：ID-only MRR **0.0112** vs TF **0.0162** / LLM **0.0164**（约 **+45%**；早期同协议 seed2025） | ⚠️ 需与当前 stop-gate 超参对齐补一版主表 ID-only |
| TF ≈ LLM，解释大部分收益 | Main 3-seed：TF 0.0165±0.0008 ≈ LLM 0.0161±0.0005 | ✅ |
| MV 视角冗余 | view 对 **CKA > 0.83**；4-view concat eff-rank **136.9/256 (0.535)**；leave-one-view Δ≈0 | ✅ |
| 时间漂移 | valid→test unseen target **11.3%→17.8%**；head **63.1%→56.6%**；三模型同步掉约 44% | ✅ |
| item frequency 限制 | MV 在 new 桶最弱（MRR_new 0.0104 vs LLM 0.0121，seed2025 fair run） | ⚠️ 主表需统一 low/mid/head 命名定稿 |
| Cross 非稳定增益 | §2 全部数字 | ✅ |

---

## 4. 机制分析清单（L11）— 逐项状态

| 机制项 | 状态 | 关键数字 / 产物 |
|--------|:----:|----------------|
| Rank transition | ✅ | LLM：Improve≈Degrade≈49.5%；报告 `rank_transition_beauty_*.md` |
| Score entropy | ✅ | LLM Top10：0.9195 → 0.9209（+0.001） |
| Top1–Top10 margin | ✅ | LLM：1.338 → 1.298（−0.040） |
| Top-K head share | ✅ | §2.3；LLM Δ≈0，TF/MV 略降 |
| View redundancy | ✅ | CKA>0.83；SVD 谱图 `fig_svd_spectrum_beauty.png` |
| **UniSRec portability** | ❌ | 仅有旧 Base/Align（MRR 0.0124，无增益）；**MV-V3 OOM，未在当前 no-boost 协议下重跑** |

---

## 5. 尚未关死的点（今晚重点征求意见）

按「硬缺口 / 定稿向」分开：

### A. 硬缺口（叙事完整性）

| # | 缺口 | 现状 | 建议问老师 |
|---|------|------|------------|
| **A1** | **UniSRec portability** | Base/Align 旧结果；MV 曾 OOM | 是否必须进主文？若要：Beauty 上 UniSRec + TF/LLM（no-boost）各 1–3 seed 是否够？是否可砍 MV-UniSRec？ |

### B. 定稿向（证据有，表未锁）

| # | 缺口 | 现状 | 建议问老师 |
|---|------|------|------------|
| **B1** | **ID-only 主表对齐** | 有 0.0112（同协议早期）；未与 stop-gate 同超参 3-seed 同表 | 是否接受「单 seed / 已有数」进主表，还是补 3-seed？ |
| **B2** | **item-freq 分桶表定稿** | 已有 freq/few/new 与 head share；命名待统一为 low/mid/head | 主文是否必须一张 low/mid/head 表？item-age 是否做（老师写「也可以考虑」）？ |

### C. 可选扩展（老师未写死）

| # | 项 | 说明 |
|---|-----|------|
| C1 | Toys / Grocery 同结论复现 | 跨域；非 stop-gate 必要条件 |
| C2 | MV 4×256 vs LLM 1024 | 老师：仅当确认互补而非冗余时再做；**当前 CKA 已判冗余 → 建议不做** |

---

## 6. 建议的论文定位（请老师拍板）

**标题级贡献**：Leakage-aware controlled study under global temporal evaluation  

**主方法**：SASRec + TF-IDF 或 SASRec + single-view LLM（no-Cross, no-boost）  
**分析对象 / 负结果**：prompt multi-view；Cross  
**优化细节（非贡献）**：two-phase  

不需要追回「漂亮层级」；用 stop-gate + 机制数字支撑洞察即可。

---

## 7. 若老师认可 stop-gate：下一步优先级（草案）

| 优先级 | 任务 | 预算 |
|:------:|------|------|
| P0 | UniSRec + TF/LLM Beauty（按老师是否要求） | 1–2 天 |
| P1 | ID-only 与主表同配置对齐（1 或 3 seed） | 半天–1 天 |
| P1 | low/mid/head 主表一行定稿（离线即可） | 半天 |
| P2 | Toys 快速复现（TF/LLM/MV 1–3 seed） | 视截稿 |
| — | 开写 Method / Findings 草稿 | 并行 |

---

## 附录：关键文件路径（本机）

| 内容 | 路径 |
|------|------|
| 本汇报 | `paper_recsys/advisor_briefing_20260722.md` |
| Early CSV | `paper_recsys/early60_stopgate_beauty.csv` |
| Main 3-seed | `paper_recsys/stopgate_3seed_beauty_metrics.csv` |
| Cross 机制 | `paper_recsys/rank_transition_beauty_{llm,tf,mv}.md` |
| 日状态 | `paper_recsys/experiment_status_20260722.md` |
| 老师原文 | `paper_recsys/0717zhidao.txt` |
