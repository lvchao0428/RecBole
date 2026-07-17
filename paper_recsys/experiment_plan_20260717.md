# 实验计划 — 2026-07-17

> 基于: `0717zhidao.txt` 老师最新建议  
> 状态: **待审核**（审核通过后再执行）  
> 关联: `WSDM_convergence_tracker.md`, `WSDM_experiment_spec.md`

---

## 一、0717 老师核心要求总结

1. **接受当前格局**：TF ≈ LLM > MV，不追层级
2. **诊断 valid→test 44% 下降**：检查 temporal distribution shift
3. **共享小网格调参**：lr/dropout/wd 各 2-3 值，同预算，仅用 valid 选参
4. **Rolling validation window**：加早期 valid 窗口检验排序稳定性
5. **Item-frequency 分桶**：保留，改名 low/mid/head，加 item age
6. **MV 不扩维**：先诊断 view redundancy（pairwise cosine/CKA/leave-one-view-out）
7. **Stop gate**：3-seed + 2-window，MV 不超 single-view 一个 std → 降级为分析对象
8. **WSDM 叙事**：leakage-aware controlled study + 机制分析

---

## 二、执行优先级

### P0 — 数据分析（不消耗 GPU，立即可做）

| # | 任务 | 工具/脚本 | 机器 | 预计耗时 | 产出 |
|---|------|----------|:----:|:--------:|------|
| A1 | **Valid/Test distribution shift 分析** | `scripts/analyze_distribution_shift.py` | 5090 / 本地 | 5min | json + 分析结论 |
| A2 | **View redundancy 深度分析** (pairwise cosine + CKA + effective rank) | `scripts/analyze_view_redundancy.py` | 5090 / 本地 | 10min | json + 冗余判定 |
| A3 | **SVD 谱图可视化**（论文 Figure） | `scripts/diagnose_collapse.py` + matplotlib | 本地 | 30min | PDF figure |

**A1 目的**：回答老师问题——valid→test 44% 下降是 temporal distribution shift 还是过拟合。检查 item popularity、catalog overlap、target item age、用户历史长度差异。

**A2 目的**：正式确认 "prompt-based multi-view information is largely redundant" 结论，为论文提供 CKA 数值证据。

**A3 目的**：生成论文用 SVD 谱图 Figure（singular value spectrum: LLM vs MV）。

---

### P1 — 机制分析（需 GPU 推理，不需重新训练）

| # | 任务 | 工具/脚本 | 机器 | 预计耗时 | 产出 |
|---|------|----------|:----:|:--------:|------|
| B1 | **Rank-transition matrix** (LLM no-Cross vs +Cross) | `scripts/analyze_rank_transition.py` | 5090 | 1-2h | md report |
| B2 | **Score entropy + Top-K margin** | 同上 (内置) | 5090 | 同上 | 同上 |
| B3 | **Leave-one-view-out** (MV checkpoint) | `scripts/analyze_leave_one_view_out.py` | 5090 | 1h | md report |

**B1-B2 目的**：生成 Cross 机制分析的直接证据——target rank 迁移、score 集中度变化。对应老师："Cross 最值得做的是 per-user target rank transition"。

**B3 目的**：确认各 view 的边际贡献，如果 4 个 view drop 后 MRR 变化相似 → 冗余证据。

---

### P2 — 共享小网格（需 GPU 训练）

| # | 任务 | 工具/脚本 | 机器 | 预计耗时 | 产出 |
|---|------|----------|:----:|:--------:|------|
| C1 | **TF-IDF 小网格** (lr×3, dropout×3, wd×3 = 27 runs) | `scripts/run_shared_grid.py` | 5090 | ~6-8h | best valid config |
| C2 | **LLM 小网格** (同上 27 runs) | 同上 | 5090 | ~6-8h | best valid config |
| C3 | **MV 小网格** (同上 27 runs) | 同上 | 5090 | ~8-10h | best valid config |

**注意**：27 组全开太慢，建议先做关键 9 组（lr×3, dropout×3, wd 固定为 0）= 9 runs × 3 models = 27 runs 总共 ~9h。

**备选方案**：如果预算有限，先只做 lr×3（每模型 3 runs），再根据最优 lr 展开 dropout。

---

### P3 — Stop Gate 实验（依赖 P2 网格结果）

| # | 任务 | 机器 | 预计耗时 | 产出 |
|---|------|:----:|:--------:|------|
| D1 | **LLM no-Cross 3-seed** (seed=42/2024/2025, 用 P2 最优 config) | 5090 | 6h | 均值±std |
| D2 | **MV no-Cross 3-seed** (同上) | 5090 | 8h | 均值±std |
| D3 | **Rolling early-valid window** (Beauty, 移动 cutoff) | 5090 | 4h | 排序稳定性 |
| D4 | **Toys per-source 公平对比** (4 模型 × no-Cross) | 5090 | 8h | 跨域验证 |

---

### P4 — 论文写作准备（P1-P3 完成后）

| # | 任务 | 说明 |
|---|------|------|
| E1 | 整理 Figure: SVD 谱图 + Rank-transition matrix | 两张核心 figure |
| E2 | 整理 Table: 主表 (3-seed 均值±std) + 分桶表 | 用 P3 数据 |
| E3 | 更新 WSDM_convergence_tracker.md | 记录 V4 实验结果 |

---

## 三、今日执行建议（时间线）

| 时间 | 行动 | 说明 |
|------|------|------|
| **上午** | 执行 A1 + A2 + A3 | 不耗 GPU，纯数据分析 |
| **中午** | 启动 B1 + B2（5090 GPU） | rank-transition 推理 |
| **下午** | B3 leave-one-view-out + 准备网格脚本 | |
| **晚上** | 启动 C1 小网格（先 9 组 TF-IDF） | 过夜跑 |

---

## 四、不执行项（对齐 0717 建议）

| 不做的事 | 原因 |
|----------|------|
| MV 调参追层级 | 老师明确"不要再加 trick 追回层级" |
| 4×256 MV 扩维 | 老师明确反对，view 冗余不是容量问题 |
| 加回 cold boost / SE | 已简化为 V4 |
| 根据 test 选配置 | 老师明确禁止 |
| 重跑全部 40 组 | 只做 stop gate 必须的最小实验集 |

---

## 五、新增基础设施清单（已完成编写）

| 脚本 | 功能 | 对应需求 |
|------|------|---------|
| `scripts/analyze_distribution_shift.py` | valid/test 分布漂移诊断 | 0717 #2 |
| `scripts/analyze_rank_transition.py` | Cross rank-transition + score entropy | 0717 机制分析 |
| `scripts/analyze_leave_one_view_out.py` | Leave-one-view-out 消融 | 0717 #6 |
| `scripts/analyze_view_redundancy.py` | Pairwise cosine + CKA + effective rank | 0717 #6 |
| `scripts/run_shared_grid.py` | 共享小网格 runner | 0717 #3 |
| `scripts/diagnose_collapse.py` | 嵌入塌缩诊断（已有） | SVD 谱图 |

---

## 六、审核确认项

请确认以下决策后再开始执行：

- [ ] P0 数据分析 (A1-A3) 是否立即执行？
- [ ] P1 机制分析 (B1-B3) 是否今天启动？
- [ ] P2 小网格大小：27 组全开 vs 先 9 组（lr×dropout）？
- [ ] P3 stop gate 是否等 P2 最优 config 出来后再跑？还是直接用当前 config 做 3-seed？
- [ ] Toys 域验证放 P3 还是提前？

---

> 审核通过后执行 → 结果更新到 `WSDM_convergence_tracker.md` + 每日进展文档
