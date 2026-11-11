# WSDM 2027 投稿收敛追踪

> 创建: 2026-07-17  
> 最后更新: 2026-07-26  
> 目标会议: WSDM 2027（Abstract deadline TBD，full paper TBD）  
> 本文档持续追踪至截稿，与每日进展文档(`experiment_status_*.md`)互补  
> **0723 止损**：不再救 MV-Align；论文改为 leakage-aware controlled study

---

## 一、论文定位

### 当前主线叙事（0723 锁定）

> When More Text Is Not Better：严格 GTS 下文本仍明显优于 ID-only，但复杂 LLM/MV/Cross 额外收益不 robust，受 temporal shift 与 view redundancy 限制。

**核心贡献方向**：Leakage-aware controlled study + negative results + 机制分析  
**停止**：扩 MV / 调 boost / 加 Cross / 追求漂亮层级。

### Stop Gate（0717 设定 → 0723 已触发）

- MV 在 Main + Early 两窗口均未超过 single-view 一个 std → **MV = 分析对象**
- Cross 无稳定增益 → **Appendix negative analysis only**

---

## 二、版本迭代历史

### V1 — RecSys 2026 投稿版（已拒）

| 项目 | 内容 |
|------|------|
| 评测协议 | Per-user leave-one-out + sampled candidates |
| 模型结构 | SASRec + SENet + multi-view align + Cross + cold_text_boost + infer_boost + two-phase |
| 主要贡献 | MV-Align 框架：多视角对齐 + squeeze-excite + 双重 cold boost |
| 数据集 | Beauty / Toys / Book-Crossing (LOO split) |
| 拒稿主因 | (1) global temporal leakage (2) 5-core 声明与数据不符 (3) 复杂度高/边际收益小 (4) cold-start 定义不准确 |

### V2 — 修正评测协议（7/10–7/13）

| 项目 | 内容 |
|------|------|
| 评测协议 | **Global time split** (stratified, train cutoff) + full ranking |
| 用户筛选 | train cutoff 前 ≥5 条历史 |
| TF-IDF 处理 | train-only fit (vocabulary, IDF, SVD, center-whiten) |
| LLM 处理 | 静态 Qwen2.5-7B encode（title 不含未来信息）|
| 模型结构 | 保留 SE、boost、Cross |
| 关键发现 | 层级大幅缩小；MV 不再稳定优于 LLM；但文本整体有效（+40%~50% vs ID-only）|

### V3 — 简化模型 + 公平对比（7/13–7/16）

| 项目 | 内容 |
|------|------|
| 评测协议 | 同 V2 |
| 模型结构 | **删除 SE、cold_text_boost、infer_boost**；two-phase 作 optimization protocol |
| Fusion | no-Cross: concat+predictor；+Cross: DCN-V2 |
| Align | per-source align（TF: 1× InfoNCE, LLM: 2×, MV: 5×）|
| MV 修复 | concat-fix（加法→concat+predictor）、per-view align + base align |
| 关键发现 | TF 0.0162 ≈ LLM 0.0164 > MV 0.0158；Cross 无净收益；MV 塌缩诊断确认 view 冗余 |
| 状态 | **当前版本** |

### V4 — WSDM 投稿版（规划中，0717 起）

| 项目 | 计划内容 |
|------|---------|
| 评测协议 | 同 V3 + rolling validation window + item-frequency 分桶改名 |
| 超参调优 | 共享小网格（lr/dropout/wd 各 2-3 值，同预算）；不再根据 test 选配置 |
| Stop Gate | 3-seed + 2 temporal windows 判定 MV/Cross |
| 分桶 | low / mid / head（按 train cutoff 前交互次数）+ item age 分桶 |
| 机制分析 | rank-transition matrix, score entropy, view redundancy, leave-one-view-out, gate-by-frequency |
| 数据集 | Beauty + Toys (+ Grocery 可选 / 非 Amazon 待选) |
| 模型 | SASRecAlignV3 (TF+LLM, no-Cross, no-boost) 为主方法 |
| MV/Cross | 作为分析对象，不作为主方法（除非通过 stop gate）|

---

## 三、关键实验结果存档

### Beauty TS, seed=2025, min5, per-source align, no-boost（V3 固定配置）

| 配置 | MRR@10 | NDCG@10 | R@10 | R_new@10 | MRR_freq | MRR_few | MRR_new |
|------|:------:|:-------:|:----:|:--------:|:--------:|:-------:|:-------:|
| ID-only | 0.0112 | 0.0151 | 0.0277 | 0.0137 | — | — | — |
| TF no-Cross | **0.0162** | 0.0205 | 0.0348 | 0.0234 | 0.0340 | 0.0177 | 0.0116 |
| LLM no-Cross | **0.0164** | **0.0209** | **0.0358** | 0.0198 | 0.0339 | 0.0186 | **0.0121** |
| MV no-Cross | 0.0158 | 0.0198 | 0.0330 | 0.0168 | 0.0329 | 0.0186 | 0.0104 |
| TF +Cross | 0.0164 | 0.0202 | 0.0327 | 0.0193 | 0.0339 | 0.0186 | 0.0119 |
| LLM +Cross | 0.0162 | 0.0202 | 0.0335 | 0.0224 | 0.0338 | 0.0180 | 0.0118 |
| MV +Cross | 0.0156 | 0.0196 | 0.0326 | 0.0183 | 0.0320 | 0.0187 | 0.0106 |

### Beauty 共享小网格最优（7/18–7/20 Grid + Fix，按 valid 选参）

| 模型 | 最优配置 | valid MRR | test MRR | test NDCG | test HR | MRR_new |
|------|----------|:---------:|:--------:|:---------:|:-------:|:-------:|
| **MV (正确 Phase-A)** | lr=5e-4, do=0.3 | **0.0284** | 0.0158 | 0.0199 | **0.0333** | 0.0119 |
| **TF-IDF** | lr=5e-4, do=0.3 | 0.0280 | 0.0155 | 0.0191 | 0.0310 | 0.0098 |
| **LLM** | lr=1e-4, do=0.1 | 0.0279 | **0.0162** | **0.0199** | 0.0319 | **0.0133** |

> Grid: lr∈{1e-4,5e-4,1e-3} × dropout∈{0.1,0.3,0.5}；MV 使用 Phase-A 20ep + lr groups。  
> 结论草案：三模型 test 差距 <0.001，**待 3-seed stop-gate**；LLM 在 MRR_new 仍占优。

### Beauty 网格：按 test 选参诊断（7/20，非规范）

| 模型 | 配置 | valid | test MRR | 翻转？ | 平均 gap |
|------|------|:-----:|:--------:|:------:|:--------:|
| LLM | lr=1e-4, do=0.1 | 0.0279 | **0.0162** | 否 | 45.7% |
| MV | lr=5e-4, do=0.5 | 0.0269 | **0.0162** | **是**（valid 选 do=0.3） | **43.6%** |
| TF-IDF | lr=5e-4, do=0.3 | 0.0280 | 0.0155 | 否 | 45.7% |

> **诊断结论**：平均 gap 上 MV 并不更大；问题是选参不稳定（唯一翻转）。偷看 test 时 MV 追平 LLM，但 MRR_new 仍落后（0.0120 vs 0.0133）。规范仍用 valid；3-seed 建议主报 MV do=0.3，附带 do=0.5。

### Beauty Stop-Gate 3-seed（7/20–7/21，权威：`logs/stopgate_3seed_beauty_metrics.csv`）

| 配置 | seed42 | seed2024 | seed2026 | **test MRR mean±std** |
|------|:------:|:--------:|:--------:|:---------------------:|
| **TF** lr=5e-4/do=0.3 | 0.0160 | 0.0161 | 0.0174 | **0.0165±0.0008** |
| **LLM** lr=1e-4/do=0.1 | 0.0156 | 0.0161 | 0.0165 | **0.0161±0.0005** |
| **MV** lr=5e-4/do=0.3 | 0.0159 | 0.0153 | 0.0169 | **0.0160±0.0008** |
| MV* lr=5e-4/do=0.5 | 0.0154 | 0.0156 | 0.0154 | 0.0155±0.0001 |

**Stop-gate 结论**：MV 未超过 LLM/TF 一个 std → **MV 不进主方法，作 view-redundancy / negative-result 分析**；主方法在 TF-IDF 与 LLM 之间（均值 TF 略高，方差 LLM 更小）。

### 嵌入塌缩诊断结论

| 诊断项 | 结果 | 解释 |
|--------|------|------|
| MV views concat effective rank | 136.4/256 (0.533) | 4 view concat 后严重冗余 |
| LLM+TF concat effective rank | 485/512 (0.947) | 两源互补 |
| Mode B: Raw text MV/LLM ratio | 0.405 | 训练后塌缩加剧 |
| Mode B: Projected MV/LLM ratio | 0.721 | 投影层部分缓解 |
| Mode B: Fused MV/LLM ratio | 1.113 | ID embedding 完全补偿 |
| view_1 vs view_2 cosine | 0.4177 | Function vs Audience 高冗余 |
| MV gate α | 0.685 (< LLM 0.717) | 模型学会降低文本权重 |

### Toys GTS 3-seed（7/24–7/26，定稿：`logs/wsdm_wrapup_toys_gts.csv`）

| 模型 | seed42 | seed2024 | seed2026 | **test MRR mean±std** |
|------|:------:|:--------:|:--------:|:---------------------:|
| ID-only | 0.0070 | 0.0070 | 0.0069 | **0.0070±0.0001** |
| **TF** lr=5e-4/do=0.3 | **0.0116** | **0.0112** | **0.0110** | **0.0113±0.0003** |
| LLM lr=1e-4/do=0.1 | 0.0111 | 0.0105 | 0.0104 | 0.0107±0.0004 |
| MV lr=5e-4/do=0.3 | 0.0102 | 0.0112 | 0.0113 | 0.0109±0.0006 |

**Toys 结论**：TF 最优；LLM ≈ MV，与 Beauty 方向完全一致。Valid→test drop ~55%。

### Beauty Clean LOO 3-seed（✅ 定稿：`logs/beauty_loo_results.csv`）

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** | vs GTS |
|------|:------:|:--------:|:--------:|:--------------:|:------:|
| ID-only | 0.0179 | 0.0171 | 0.0174 | **0.0175±0.0004** | 1.65× |
| TF | 0.0250 | 0.0253 | 0.0249 | **0.0251±0.0002** | 1.52× |
| LLM | 0.0251 | 0.0252 | 0.0258 | **0.0254±0.0004** | 1.58× |
| MV | 0.0258 | 0.0251 | 0.0250 | **0.0253±0.0004** | 1.58× |

**关键发现**：LOO 下 LLM≈MV≥TF（GTS 下 TF 最优）→ **协议改变模型排序**。

### Cross 效应摘要

| 模型 | Δ MRR@10 | Δ R@10 | Δ R_new@10 | 判定 |
|------|:--------:|:------:|:----------:|------|
| TF | +0.0002 | -0.0021 | -0.0041 | 无排序增益，压覆盖 |
| LLM | -0.0002 | -0.0023 | **+0.0026** | 覆盖 vs 排序 trade-off |
| MV | -0.0002 | -0.0004 | +0.0015 | 无净正向 |

---

## 四、待验证的 stop gate 实验

| # | 实验 | 目标 | 机器 | 状态 |
|---|------|------|:----:|:----:|
| 0 | 共享小网格 TF/LLM/MV（含 TF log 修复 + MV 正确 PA） | 同预算选参 | 5090 | ✅ 7/20 05:27 完成 |
| 1 | LLM no-Cross 3-seed (42, 2024, 2026) Beauty，lr=1e-4/do=0.1 | 主方法方差 | 5090 | ✅ test **0.0161±0.0005** |
| 2 | TF 3-seed Beauty，lr=5e-4/do=0.3 | baseline 方差 | 5090 | ✅ test **0.0165±0.0008** |
| 3 | MV 3-seed Beauty，lr=5e-4/do=0.3 + 正确 PA（主） | MV stop-gate | 5090 | ✅ test **0.0160±0.0008**；**未超 LLM 1std → 降为分析对象** |
| 3b | MV 3-seed Beauty，lr=5e-4/do=0.5 + 正确 PA（附带） | 选参稳定性 | 5090 | ✅ test 0.0155±0.0001（更稳但更低） |
| 4 | 2 temporal windows (Early60) Beauty | 时间稳定性 | 5090 | ✅ TF≥MV≥LLM；两窗口 stop-gate 正式触发 |
| 5 | Toys GTS Main 3-seed（wrap-up 队列） | 跨域验证 | 5090 | ✅ TF **0.0113±0.0003** > LLM 0.0107 ≈ MV 0.0109；与 Beauty 一致 |
| 6 | Beauty ID-only 3-seed + UniSRec（log10） | 主表 ID baseline / portability | log10 | ✅ ID GTS **0.0106±0.0005** |
| 7 | Warm-target audit（G1） | 主表口径 | CPU | ✅ Beauty unseen 27.8%；Toys 34.5% |
| 8 | **Beauty Clean LOO 3-seed** | LOO vs GTS protocol 对照 | 5090 | ✅ 12/12；ID 0.0175 / TF 0.0251 / LLM 0.0254 / MV 0.0253；LOO≈1.5×GTS，排序翻转 |
| 9 | **Toys Clean LOO 3-seed** | LOO vs GTS 跨域 | 5090 | ✅ 12/12；见 `logs/toys_loo_results.csv` |
| 10 | **UniSRec GTS** Beauty+Toys 3-seed | portability | 5090 | 🔄 **P0 队列中**：`run_5090_remaining_p0.sh` 对 6 ckpt 做真 test(+valid) → `logs/unisrec_gts_test_eval.csv` |
| 11 | **Warm-only 重评估** | 主表口径 | 5090 | ✅ 21 ckpt；warm≈1.37×(Beauty)/1.48×(Toys)，**排序不变** |
| 12 | **LOVO 修复重跑** | view redundancy | 5090 | 🔄 接在 UniSRec 后：zero-mask → `leave_one_view_out_{beauty,toys}.md` |

### 数据有效性问题（2026-08-02 核查）

| # | 项目 | 问题 | 处理 |
|:-:|------|------|------|
| B1 | Leave-one-view-out | 脚本改 `model.text_view_indices`，前向实际读 `active_buffer_positions`（`__init__` 固定）→ 属性赋值是 no-op，故 4 view 的 Δ 全为 `0.000000` | 需改为对 concat 向量的对应切片做零掩码，或用 config 重建模型。**effective rank 的 `drop_view_*` 证据不受影响，仍可用** |
| B2 | UniSRec | `--only_phase_a` 路径下 UniSRec 日志只有 `valid result`，无 `test_result`；提取正则抓到末次 valid | 用已存 6 个 ckpt 补 test-only eval（~20 min），勿重训 |

> 详见 [`0723_results_summary.md`](0723_results_summary.md)

---

## 五、论文组件清单

| 组件 | V4 定位 | 备注 |
|------|---------|------|
| Global time split 评测 | **核心贡献** | 展示 protocol 改变结论 |
| TF-IDF 作强 baseline | **核心论据** | 简单方法已解释大部分收益 |
| LLM single-view | **主方法** | 最优配置 |
| Multi-view | **分析对象** (negative/boundary result) | view 冗余导致不优于 single-view |
| Cross-attention | **机制分析** | 覆盖 vs 排序 trade-off |
| 两阶段训练 | Optimization protocol | 非贡献 |
| SE / cold boost / infer boost | **已删除** | 不进入 WSDM 版 |
| UniSRec portability | 补充验证 | 跨 backbone |
| SVD 谱图 + effective rank | **论文 Figure** | 可视化塌缩证据 |
| Rank-transition matrix | **论文 Figure** | Cross 机制证据 |
| Item-frequency 分桶 | **论文 Table** | low/mid/head 分析 |

---

## 六、时间线

| 日期 | 里程碑 | 状态 |
|------|--------|:----:|
| 7/10 | RecSys 拒稿，启动 WSDM 重构 | ✅ |
| 7/10–7/13 | Global time split pilot + V2 | ✅ |
| 7/13–7/16 | V3 简化模型 + per-source 公平对比 | ✅ |
| 7/16 | 嵌入塌缩诊断完成 | ✅ |
| 7/17 | V4 规划启动；规范确立；distribution shift / CKA / leave-one-view / rank-transition | ✅ |
| 7/17–7/20 | 共享小网格 + Fix（TF log 修复、MV 正确 Phase-A）| ✅ 7/20 05:27 |
| **7/20–7/21** | **3-seed stop gate（`run_5090_stopgate_3seed.sh`）** | ✅ 04:49 完成；MV 未过 gate |
| 7/21–7/22 | Early60 第二窗口 3-seed + Cross Top-K | ✅ TF≥MV≥LLM |
| **7/23** | **0723 止损；wrap-up 串行队列启动（5090 Toys + log10 ID/UniSRec）** | ✅ |
| **7/26** | Toys GTS 定稿；Beauty LOO 启动 | ✅ |
| **7/27–7/28** | Beauty/Toys Clean LOO 12/12；机制分析（CKA/LOVO/shift） | ✅ |
| **7/28–7/29** | UniSRec GTS（Beauty✅；Toys 恢复中）+ warm-only 重评估排队 | 🟡 |
| 7/30 | 白天：论文数据分析（查收 UniSRec + warm-only + paper_tables） | ⏳ |
| 7/30–8/05 | 论文骨架撰写 | ⏳ |
| TBD | WSDM abstract deadline | ⏳ |

---

## 七、变更日志

| 日期 | 变更 |
|------|------|
| 2026-07-17 | 文档创建；V1→V3 历史整理完成；V4 规划确立 |
| 2026-07-20 | Grid Fix 完成；写入 Beauty 网格最优表；stop-gate 配置按 valid 锁定（LLM/TF/MV） |
| 2026-07-20 12:50 | 增加按 test 选参诊断：MV 唯一翻转；平均 gap 不大；3-seed 建议附带 MV do=0.5 |
| 2026-07-20 15:10 | 同步三端；启动 Stop-Gate 3-seed（LLM/TF/MV/MV*×seeds 42/2024/2026） |
| 2026-07-21 11:25 | 3-seed 完成；真实 test：TF 0.0165±0.0008 ≥ LLM 0.0161±0.0005 ≈ MV 0.0160±0.0008；MV stop-gate 未通过 |
| 2026-07-22 | Early60 完成：TF 0.0053±0.0002 ≥ MV 0.0052 ≥ LLM 0.0049；两窗口 stop-gate 正式触发 |
| 2026-07-23 | 0723 止损叙事锁定；启动 `run_5090_wsdm_wrapup_serial.sh` + `run_log10_wsdm_wrapup_serial.sh`；warm-audit 完成 |
| 2026-07-26 | Toys GTS 3-seed 定稿（TF>LLM≈MV，跨域验证）；Beauty Clean LOO 启动 |
| 2026-07-28 | Beauty/Toys Clean LOO 均 12/12 定稿；机制分析完成；UniSRec GTS 启动 |
