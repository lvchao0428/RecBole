# 实验进展 — 2026-07-13

> 最后更新: 2026-07-14 15:50
> 指导依据: 老师 0712 建议 (WSDM leakage-aware controlled study)
> **7/14 15:50**: boost+no-Cross 已完成 3/4（LLM MRR=**0.0171** 目前最高）；MV 跑 Phase-B ep~37。详见 `experiment_priority_20260714.md`

---

## 〇、队列状态 (7/14 15:50)

| 任务 | 机器 | 状态 | 预计完成 |
|------|:----:|:----:|:--------:|
| boost+Cross 4-config | 5090 | ✅ | 7/14 01:07 |
| **boost+no-Cross** ID / TF / LLM | **5090** | ✅ | 12:29 / 13:21 / 14:15 |
| **boost+no-Cross** MV | **5090** | 🔄 Phase-B ep~37/50 | **~16:20** |
| **UniSRec** Beauty Base | **log10** | 🔄 ep~13/50（~14min/ep） | Base ~00:00 |
| UniSRec Align+MV | log10 | ⏳ 接 Base 后 | 过夜 |

### boost+no-Cross 中间结果（test）

| 配置 | MRR@10 | NDCG@10 | R@10 | R_new |
|------|:------:|:------:|:----:|:-----:|
| ID-only | 0.0112 | 0.0151 | 0.0277 | 0.0137 |
| TF no-Cross + boost | 0.0160 | 0.0202 | 0.0340 | 0.0178 |
| **LLM no-Cross + boost** | **0.0171** | **0.0213** | **0.0350** | **0.0270** |
| MV no-Cross + boost | 🔄 | — | — | — |

**阶段性结论**：boost 在 no-Cross 下 **未恢复** MV>LLM>TF；当前 **LLM > TF > ID**，且 LLM=0.0171 高于 no-boost no-Cross 的 0.0168 与 boost+Cross 的 0.0164。

### 四象限 MRR@10（MV 格待补）

| | no-Cross | +Cross |
|--|:--------:|:------:|
| no-boost | LLM 0.0168 / TF 0.0157 / MV 0.0141 | 全 0.0157 |
| cb3+infer | LLM **0.0171** / TF 0.0160 / MV 🔄 | LLM 0.0164 / TF 0.0160 / MV 0.0152 |

---

## 一、配置核查：with-Cross 三模型 MRR 均为 0.0157

### 疑问：是否单一变量？文本特征是否生效？

**结论：配置正确、文本特征已生效；MRR 相同是严格协议下的真实收敛，不是 bug。**

| 检查项 | TF+Cross | LLM+Cross | MV+Cross |
|--------|:--------:|:---------:|:--------:|
| `use_llm` | False | **True** | False (MV 用 4-view) |
| `use_cross` | True | True | True |
| `disable_text_feature` | False | False | False |
| `item_text_emb_path` | base.ts.npy | base + qwen.ts.npy | base + 4views_ts/ |
| `text_mode` | base | **both** | multiview (via split_dir) |
| Phase-A trainable params | ~0.85M | ~1.64M | ~1.78M |
| test HR@10 | 0.0326 | 0.0320 | 0.0333 |
| test NDCG@10 | 0.0197 | 0.0195 | 0.0198 |
| test MRR@10 | 0.0157 | 0.0157 | 0.0157 |

- 三模型 **HR/NDCG 不同** → 模型确实不同、文本路径不同
- MRR@10 在 4 位小数相同 → 严格 TS+min5+no-boost+Cross 下 ranking 质量收敛到同一水平
- 去掉 Cross 后 LLM 升到 0.0168 → 进一步证明特征差异在 no-Cross 时可分辨

**单一变量对照**（seed=2025, min5, no-boost, V3 特征, 两阶段 20+50）：

| 变量 | TF | LLM | MV |
|------|:--:|:---:|:--:|
| 文本特征 | TF-IDF only | TF-IDF + Qwen | TF-IDF + 4-view Qwen |
| Cross | on | on | on |
| cold_boost / infer_boost | 0 | 0 | 0 |

---

## 二、Toys/Grocery 配置核查（ID-only 画框部分）

### 疑问：ID-only HR 高于 text 模型，是否配置错误？

**结论：评测配置一致（同 min5、同 TS split、同 seed），但训练协议不对称；且应同时看 MRR。**

#### Toys seed=2025 实测 (test)

| 配置 | 训练协议 | HR@10 | MRR@10 |
|------|----------|:-----:|:------:|
| ID-only | only_phase_a, 50ep, 无文本 | **0.0215** | 0.0074 |
| TF-IDF | two-phase 20+50, base.ts.npy | 0.0193 | **0.0107** |
| LLM | two-phase 20+50 | 0.0187 | 0.0112 |
| MV | two-phase 20+50 | 0.0190 | 0.0105 |

- `min_train_interactions=5`：valid 25594 users, test 4873 users — **ID 与 text 一致** ✅
- text 模型 `disable_text_feature=False`，`item_text_emb.base.ts.npy` 路径正确 ✅
- **HR**：ID > text（ID 覆盖率更高）
- **MRR**：text > ID（排序质量更好）

→ 这不是配置错误，而是 **HR–MRR trade-off**：text 提升排序但降低 hit 覆盖率，与 Cross 机制叙事一致。

#### 已知不对称（后续可改进）

| 项目 | ID-only | text 模型 |
|------|---------|-----------|
| 训练方式 | only_phase_a 50ep 全网训练 | Phase-A 冻 backbone + Phase-B 联合 |
| 模型类 | SASRecAlign (无文本头) | SASRecAlignV3 / MV-V3 |

Beauty 上用同一不对称协议仍得到 text >> ID，故 Toys 上 text 未超 ID 的 HR 更可能是 **domain 特性**（Toys 标题信息弱），而非实现 bug。

#### Grocery seed=2025 (test)

| 配置 | HR@10 | MRR@10 |
|------|:-----:|:------:|
| ID-only | 0.0125 | 0.0040 |
| TF-IDF | **0.0129** | **0.0065** |
| LLM | 0.0128 | **0.0068** |
| MV | 0.0121 | 0.0058 |

Grocery 上 text 在 MRR 上均优于 ID，HR 差距极小。

---

## 三、今日完成事项

### 1. 5090 主队列 (`run_5090_queue_20260712.sh`) — 14:36 全部完成 ✅

| 阶段 | 内容 | 状态 |
|------|------|:----:|
| Phase 2 | Beauty LLM/MV 3-seed (seed=2024,42) | ✅ |
| Phase 3 | MV no-Cross ablation (旧版, threshold=0.01) | ✅ |
| Phase 4a | Toys 4-config seed=2025 (ID/TF/LLM/MV) | ✅ |
| Phase 4b | Grocery 4-config seed=2025 (ID/TF/LLM/MV) | ✅ |

### 2. log10 ID-only 全量 — 05:16 全部完成 ✅

Beauty/Toys/Grocery × 3 seeds，共 9 个实验。rsync→5090 因 IP 错误失败（已修复为 `192.168.0.122`），已从 5090 侧补同步。

### 3. Advisor P(2) Cross ablation (`run_advisor_p2_ablation.sh`) — 18:19 完成 ✅

| 格子 | 实验 | 耗时 | test MRR@10 | test HR@10 |
|------|------|:----:|:-----------:|:----------:|
| A | ID-only | (已有) | 0.0112 | 0.0277 |
| B | TF-IDF + Cross | (已有) | 0.0157 | 0.0326 |
| C | TF-IDF no-Cross | 52 min | 0.0157 | 0.0332 |
| D | TF-IDF+LLM + Cross | (已有) | 0.0157 | 0.0320 |
| E | TF-IDF+LLM no-Cross | 54 min | **0.0168** | **0.0349** |
| F | MV + Cross | (已有) | 0.0157 | 0.0333 |
| G-fix | MV no-Cross (fixed) | 117 min | 0.0141 | 0.0363 |

**Cross 效应汇总** (on − off):

| 模型族 | Δ MRR@10 | Δ HR@10 | Δ R_new@10 |
|--------|:--------:|:-------:|:----------:|
| TF-IDF | 0.0000 | -0.0006 | -0.0036 |
| TF-IDF+LLM | -0.0011 | -0.0029 | +0.0040 |
| MV-Align | +0.0016 | -0.0030 | +0.0026 |

### 4. 三端代码同步 — 21:30 完成 ✅

- 本机 → 5090 → log10，含 `two_phase_train.py`、`collect_p2_ablation.py`、`sync_code_three_machines.sh`
- 修复 `run_log10_id_only_all_min5.sh` rsync IP: `192.168.0.106` → `192.168.0.122`
- 新增 `run_beauty_full_boost_nocross_v2.sh`（cb3+infer+no-cross，待跑）

---

## 二、no-Cross 层级关系分析

### 问题：去掉 Cross 后是否恢复 TF < LLM < MV 层级？

**结论：部分恢复，但未达到经典 MV > LLM > TF 层级。**

#### no-Cross + no-boost（test set, seed=2025, min5）

| 排名 | 配置 | MRR@10 | HR@10 | vs ID-only |
|:----:|------|:------:|:-----:|:----------:|
| 1 | **LLM no-Cross** | **0.0168** | 0.0349 | +50% |
| 2 | TF-IDF no-Cross | 0.0157 | 0.0332 | +40% |
| 3 | MV no-Cross | 0.0141 | 0.0363 | +26% |
| — | ID-only | 0.0112 | 0.0277 | baseline |

- text > ID-only：✅ 明确（MRR +26%~50%）
- TF < LLM：✅ LLM no-Cross 略优于 TF（+0.0011 MRR）
- MV > LLM：❌ **MV no-Cross 反而最差**（MRR 0.0141 < 0.0168）
- 经典 MV > LLM > TF：❌ 未恢复

#### with-Cross + no-boost（对比）

| 配置 | MRR@10 | HR@10 |
|------|:------:|:-----:|
| TF-IDF + Cross | 0.0157 | 0.0326 |
| LLM + Cross | 0.0157 | 0.0320 |
| MV + Cross | 0.0157 | 0.0333 |

Cross 开启时三者 MRR 完全打平（0.0157），Cross 关闭后 LLM 脱颖而出。

#### 机制解读（与老师方向对齐）

1. **Cross 不是稳定净收益组件**：LLM 去掉 Cross 后 MRR 反而升 0.0011；MV 去掉 Cross 后 MRR 降 0.0016 但 HR 升 0.003
2. **TF-IDF 是强 baseline**：no-Cross 下 TF 与 LLM 差距仅 0.0011
3. **MV 在严格协议下未展现优势**：无论 Cross 开/关，MV 的 MRR 均未超过 LLM
4. **WSDM 叙事**：复杂模型边际收益收缩 = finding，不必强行恢复层级

---

## 三、当前运行状态 (22:05)

### 5090

| 任务 | 状态 | 预计完成 |
|------|:----:|:--------:|
| boost+Cross LLM | 🔄 epoch 49/50 | **~22:15** |
| boost+Cross MV | ⏳ | ~01:30 |
| boost+no-Cross (night queue) | ⏳ 等待中 | ~07:00 |
| UniSRec seed=2025 | ⏳ | ~09:00 |

### log10：空闲

---

## 四、cb3 + infer_boost + no-Cross — 🔄 3/4 完成（7/14 15:50）

脚本：`run_beauty_full_boost_nocross_v2.sh`（7/14 12:05 补启）

| 步骤 | 配置 | 耗时 | MRR@10 | 状态 |
|------|------|:----:|:------:|:----:|
| 1/4 | ID-only | 24 min | 0.0112 | ✅ |
| 2/4 | TF no-Cross + cb3 + infer | 52 min | 0.0160 | ✅ |
| 3/4 | LLM no-Cross + cb3 + infer | 54 min | **0.0171** | ✅ |
| 4/4 | MV no-Cross + cb3 + infer | — | — | 🔄 Phase-B ep~37 → ~16:20 |

---

## 五、Beauty 完整结果汇总 (test set)

| 模型 | HR@10 | MRR@10 | 来源 |
|------|:-----:|:------:|------|
| ID-only | 0.0318 | 0.0114 | log10 3-seed |
| TF-IDF | 0.0323 | 0.0156 | 5090 3-seed |
| TF-IDF+LLM | 0.0319 | 0.0159 | 5090 3-seed |
| MV-Align | 0.0327 | 0.0154 | 5090 3-seed |

### Toys / Grocery seed=2025 no-boost

| 数据集 | ID | TF | LLM | MV |
|--------|:--:|:--:|:---:|:--:|
| Toys HR@10 | 0.0215 | 0.0193 | 0.0187 | 0.0190 |
| Grocery HR@10 | 0.0125 | 0.0129 | 0.0128 | 0.0121 |

Toys/Grocery 上 text 模型未稳定超过 ID-only。

---

## 六、Beauty 完整结果汇总 (test set)

| 时间 | 任务 | 机器 | 优先级 |
|------|------|------|:------:|
| **12:05** | 补启 `run_beauty_full_boost_nocross_v2.sh`（已验证运行） | 5090 | P0 |
| **~17:30** | 收集 boost+no-Cross 结果，对比层级是否恢复 | — | P0 |
| 晚间 | 撰写四象限表：{boost/no-boost}×{Cross/no-Cross} | 本机 | P1 |
| ~17:30→19:30 | UniSRec portability (Beauty seed=2025) 自动接棒 | 5090 | P1 |
| 明日 | Toys/Grocery 3-seed text 扩展 | 5090 | P2 |
| 明日+ | 机制分析脚本：score entropy, rank-transition | 本机 | P3 |

---

## 七、今日 (7/14) 时间线与可得出结论

| 时刻 | 实验完成 | 可得出结论 |
|------|----------|------------|
| ✅ 01:07 | boost+Cross 四配置 | boost+Cross：LLM 0.0164 > TF 0.0160 > MV 0.0152 |
| ✅ 14:15 | boost+no-Cross ID/TF/LLM | LLM **0.0171** > TF 0.0160；层级仍非 MV 主导 |
| **~16:20** | boost+no-Cross MV | **四象限齐 → boost/Cross 是否进主方法** |
| 过夜 | UniSRec Base→Align→MV (log10) | portability（慢于预期） |
| 晚间 | 四象限定稿 | 主方法建议 |

### 今日结论（15:50 阶段性）

1. no-boost + no-Cross：LLM 最优 0.0168 ✅
2. boost + Cross：LLM > TF > MV ✅（未恢复经典层级）
3. boost + no-Cross：LLM **0.0171** > TF 0.0160（MV 待出）← **暂不支持恢复层级**
4. boost/Cross 倾向：均非稳定恢复 MV>LLM>TF 的手段；TF-IDF 仍是强 baseline

### 明日待排

| 优先级 | 任务 | 估时 |
|:------:|------|:----:|
| P2 | Toys/Grocery text 3-seed | ~17h |
| P3 | 机制分析 + 公平协议重跑 | TBD |
| P4 | Book-Crossing | TBD |

---

## 八、剩余实验计划（按优先级）

| 优先级 | 任务 | 前置条件 | 估时 | 状态 |
|:------:|------|---------|:----:|:----:|
| P0 | boost+no-Cross 4-config | — | 5.5h | 🔄 5090 运行中 |
| P1 | 四象限对比表：{no-boost, boost} × {Cross, no-Cross} | P0 完成 | — | ⏳ |
| P1 | UniSRec portability (Beauty) | — | ~2–3h | 🔄 **log10 运行中**（结果 sync→5090） |
| P2 | Toys/Grocery 3-seed text 扩展 | P1 无阻塞发现 | ~24h | 未排（轻量可走 log10） |
| P3 | 机制分析：rank-transition, score entropy, head/tail | checkpoint ready | 本机 | 未排 |
| P4 | WSDM 叙事重构 + 正式表格 | 全部 P0-P2 | — | 未排 |
| P5 | 简化模型：从 config 删除 cold_boost/infer_boost 默认值 | 论文定稿前 | — | 未排 |

---

## 九、老师指导要点

1. TF-IDF 是 strong baseline；复杂模型边际收益收缩 = WSDM finding
2. SE 已删；cold_boost/infer_boost 不进入主方法（仅作 fallback 验证）
3. Cross 作为机制旋钮：当前数据表明 **LLM 去掉 Cross 更优**，MV Cross 仅 MRR +0.0016 但 HR -0.003
4. item frequency 分桶改名 low/mid/head
5. 评测：global TS split + min_train_interactions=5
6. UniSRec portability + 第三数据集必须在新 protocol 下重跑

---

## 十、结果收集路径

```
5090: /home/charlie/project/RecBole/logs/
  ts_beauty_{tfidf,llm,mv}_v2_seed{2024,2025,42}.log     ← Beauty no-boost 3-seed
  ts_beauty_{tfidf,llm,mv}_nocross_v2*.log              ← Cross ablation
  ts_beauty_fb_{id_only,tfidf,llm,mv}_seed2025.log     ← boost+Cross (running)
  ts_beauty_fb_nc_*.log                                  ← boost+no-Cross (pending)
  ts_{toys,grocery}_*_v2_seed2025.log                   ← Toys/Grocery
  log10/ts_*_id_only_min5_seed*.log                     ← ID-only 3-seed

收集命令:
  bash scripts/sync_5090_ts_results_to_local.sh
  python3 scripts/collect_p2_ablation.py
```
