# WSDM 2027 投稿收敛追踪

> 创建: 2026-07-17  
> 最后更新: 2026-07-20  
> 目标会议: WSDM 2027（Abstract deadline TBD，full paper TBD）  
> 本文档持续追踪至截稿，与每日进展文档(`experiment_status_*.md`)互补

---

## 一、论文定位

### 当前主线叙事（0717 老师确认）

> 严格 global temporal evaluation 下，静态文本仍显著改善 ID-only；TF-IDF 与 single-view LLM 已解释大部分收益；prompt-based multi-view 的边际价值受视角冗余、时间漂移和 item frequency 限制；Cross 对 score distribution 的作用依赖数据与窗口，并非稳定增益。

**核心贡献方向**：Leakage-aware controlled study + 机制分析

### Stop Gate（0717 老师设定）

修正 train-only TF-IDF/SVD/whitening 后，Beauty 3 seeds + ≥2 temporal windows：
- 若 MV 在大部分窗口和主指标上不能超过 single-view 一个 std → MV 降为分析对象，不做主方法
- 若 Cross 不能稳定改善 NDCG/MRR → Cross 从默认模型删除，仅作负结果+机制分析

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
| 1 | LLM no-Cross 3-seed (42, 2024, 2026) Beauty，lr=1e-4/do=0.1 | 主方法方差 | 5090 | 🟡 7/20 15:10 启动 |
| 2 | TF 3-seed Beauty，lr=5e-4/do=0.3 | baseline 方差 | 5090 | 🟡 同上队列 |
| 3 | MV 3-seed Beauty，lr=5e-4/do=0.3 + 正确 PA（主） | MV stop-gate | 5090 | 🟡 同上队列 |
| 3b | MV 3-seed Beauty，lr=5e-4/do=0.5 + 正确 PA（附带） | 选参稳定性 | 5090 | 🟡 同上队列 |
| 4 | 2 temporal windows (早期 valid cutoff) Beauty | 时间稳定性 | 5090 | ⏳ |
| 5 | Toys per-source 公平对比 | 跨域验证 | 5090 | ⏳ |

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
| **7/20–7/21** | **3-seed stop gate（`run_5090_stopgate_3seed.sh`）** | 🟡 运行中 |
| 7/23–7/26 | 机制分析补全 / 论文图表 | ⏳ |
| 7/26–7/30 | Toys/Grocery 域验证 | ⏳ |
| 7/30+ | 论文写作 | ⏳ |
| TBD | WSDM abstract deadline | ⏳ |

---

## 七、变更日志

| 日期 | 变更 |
|------|------|
| 2026-07-17 | 文档创建；V1→V3 历史整理完成；V4 规划确立 |
| 2026-07-20 | Grid Fix 完成；写入 Beauty 网格最优表；stop-gate 配置按 valid 锁定（LLM/TF/MV） |
| 2026-07-20 12:50 | 增加按 test 选参诊断：MV 唯一翻转；平均 gap 不大；3-seed 建议附带 MV do=0.5 |
| 2026-07-20 15:10 | 同步三端；启动 Stop-Gate 3-seed（LLM/TF/MV/MV*×seeds 42/2024/2026） |
