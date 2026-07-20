# 实验计划 — 2026-07-19

> 创建时间: 2026-07-18 23:30  
> 上一日进展: [`experiment_status_20260718.md`](experiment_status_20260718.md)  
> 实验规范: [`WSDM_experiment_spec.md`](WSDM_experiment_spec.md)

---

## 〇、实验背景（新对话必读）

### 研究目标

WSDM 2027 投稿：**Leakage-aware controlled study — 探究不同文本特征（TF-IDF / LLM / Multi-View）对序列推荐的真实增益**。

核心观点：
1. 文本特征对推荐有效（+40%~50% vs ID-only）
2. LLM embedding ≈ TF-IDF（在 leakage-free 设置下差距极小）
3. Multi-View prompt 产生语义重述而非互补信息（CKA>0.83, leave-one-view-out Δ=0）

### 模型架构

- **SASRecAlignV3**: 基于 SASRec 的文本增强模型，支持 `text_mode=base`(TF-IDF) / `both`(TF-IDF+LLM) / `llm`
- **SASRecAlignMultiViewV3**: V3 的子类，额外支持 4-view prompt (description/function/audience/style)
- **Per-source align**: 每个文本来源（TF-IDF/LLM/各 view）独立投影 + 独立 InfoNCE 对齐
- **Two-phase training**: Phase-A 冻结 backbone 训练文本头 → Phase-B 全参数微调

### 评测协议

- **Global time split** 80/10/10，min_train_interactions=5
- **Full ranking** over all warm candidates
- **主指标**: MRR@10（valid 选参），报告 MRR/NDCG/HR@10 + 分桶 (new/few/frequent)
- **公平性**: 所有模型同网格同预算调参

### 当前阶段

**共享小网格 Grid Search + Fix 阶段** — 在 Beauty 数据集上用 3×3 网格 (lr × dropout) 为三个模型找最优超参，之后进入 3-seed 验证。

---

## 一、当前 5090 后台任务

| 任务 | 脚本 | PID | 启动时间 | 预计完成 |
|------|------|:---:|:--------:|:--------:|
| Fix TF-IDF 补跑 9 组 | `run_5090_grid_fix.sh` | 2943801 | 7/18 23:26 | **~7/19 07:50** |
| Fix MV 正确参数 9 组 | 同上（串行执行） | — | ~7/19 07:50 | **~7/20 01:50** |

**检查命令**:
```bash
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader && echo "=== Process ===" && ps aux | grep "[p]ython.*two_phase_train" | head -3 && echo "=== Queue log ===" && tail -10 logs/grid_fix_nohup.log && echo "=== Fix CSV ===" && cat logs/grid_fix_results_beauty_seed2025.csv'
```

---

## 二、今日待办（优先级排序）

### P0 — 查收 Fix 结果

1. **检查 TF-IDF 补跑结果**（预计 ~07:50 前完成）
   - 读取 `logs/grid_fix_results_beauty_seed2025.csv`
   - 读取各 `logs/grid_beauty_tfidf_lr*_seed2025.log` 提取 test 指标
   - 与 LLM grid 结果对比

2. **检查 MV 正确参数 grid 结果**（预计 ~01:50 (7/20) 完成）
   - 如果已完成部分组，先记录中间结果
   - 对比 MV 旧参数结果，确认 Phase-A 修正是否有效

### P1 — 启动 3-seed 验证（Fix 全部完成后）

根据 Grid 最优配置启动 3-seed 实验：

```bash
# 需要为以下 3 个模型各跑 3 seeds (42/2024/2026)
# 1. TF-IDF: 使用 fix grid 最优配置
# 2. LLM: lr=1e-4/do=0.1（grid 最优）
# 3. MV: 使用 fix grid 最优配置
```

### P2 — Rolling early-valid window（3-seed 后）

- 验证 valid 选参的排序稳定性
- 至少 2 个不同 valid window

### P3 — Toys 域验证（最后）

- 将 Beauty 最优配置迁移到 Toys 域
- 验证跨域泛化性

---

## 三、已完成实验汇总

### 3.1 Grid Search — LLM (9/9 完成, 完整结果) ✅

| lr | dropout | valid MRR | test MRR@10 | NDCG@10 | HR@10 |
|:--:|:-------:|:---------:|:-----------:|:-------:|:-----:|
| **1e-4** | **0.1** | **0.0279** | **0.0162** | **0.0199** | 0.0319 |
| **5e-4** | **0.3** | **0.0279** | 0.0155 | 0.0193 | **0.0321** |
| 1e-4 | 0.3 | 0.0271 | 0.0148 | 0.0184 | 0.0303 |
| 5e-4 | 0.5 | 0.0260 | 0.0137 | 0.0170 | 0.0281 |
| 1e-3 | 0.3 | 0.0260 | 0.0145 | 0.0185 | 0.0316 |
| 5e-4 | 0.1 | 0.0258 | 0.0152 | 0.0186 | 0.0298 |
| 1e-3 | 0.1 | 0.0257 | 0.0137 | 0.0177 | 0.0310 |
| 1e-3 | 0.5 | 0.0239 | 0.0112 | 0.0151 | 0.0281 |
| 1e-4 | 0.5 | 0.0238 | 0.0125 | 0.0154 | 0.0249 |

### 3.2 Grid Search — TF-IDF (9/9 valid only, test 被覆盖) ⚠️

| lr | dropout | valid MRR |
|:--:|:-------:|:---------:|
| **5e-4** | **0.3** | **0.0274** |
| 1e-4 | 0.1 | 0.0269 |
| 1e-4 | 0.3 | 0.0268 |
| 1e-3 | 0.3 | 0.0264 |
| 5e-4 | 0.5 | 0.0258 |
| 5e-4 | 0.1 | 0.0255 |
| 1e-3 | 0.5 | 0.0255 |
| 1e-3 | 0.1 | 0.0248 |
| 1e-4 | 0.5 | 0.0225 |

> test 数据在 fix 补跑中，预计 7/19 07:50 完成

### 3.3 Grid Search — MV 旧参数 (5/9, 仅供参考) 🟡

| lr | dropout | valid MRR | test MRR |
|:--:|:-------:|:---------:|:--------:|
| **5e-4** | **0.3** | **0.0273** | **0.0165** |
| 1e-4 | 0.1 | 0.0269 | 0.0156 |
| 1e-4 | 0.3 | 0.0267 | 0.0146 |
| 5e-4 | 0.1 | 0.0261 | 0.0146 |
| 1e-4 | 0.5 | 0.0237 | 0.0129 |

> 使用默认 Phase-A (8ep, 无 lr groups)。正确参数版本在 fix 脚本中重跑。

### 3.4 7/17 关键分析结论

| 分析 | 核心结论 |
|------|----------|
| Distribution Shift | test unseen item +6.4%, 所有模型 valid→test ~44% 下降 |
| View Redundancy (CKA) | 所有 view 对 CKA>0.83, 语义重述 |
| Effective Rank | 4-view concat eff rank 0.535, 维度塌缩 |
| Leave-One-View-Out | Drop 任何单 view Δ=0, 完全冗余 |
| Rank-Transition (Cross) | Cross 净负向, score 均匀化但不提升 rank |

---

## 四、已知问题与修复状态

| ID | 问题 | 状态 | 修复 |
|:--:|------|:----:|------|
| 1 | MV valid→test 泛化差 | ✅ 已解释 | Temporal shift, 非模型问题 |
| 2 | Per-source align 梯度分散 | ⚠️ 待观测 | 需 per-source grad norm 可视化 |
| 3 | TF-IDF log 被 LLM 覆盖 | 🔧 fix 中 | fix 脚本用 `tfidf_` 前缀补跑 |
| 4 | MV Phase-A 极低 (valid 0.006) | 🔧 fix 中 | fix 脚本传正确 Phase-A 参数 |

---

## 五、累计关键结论

| # | 结论 | 证据 | 状态 |
|:-:|------|------|:----:|
| 1 | 文本特征整体有效 (+40%~50% vs ID-only) | V3 per-source align | ✅ 稳固 |
| 2 | TF-IDF ≈ LLM >> MV (grid valid) | Grid search | ✅ 稳固 |
| 3 | View CKA > 0.83 → 语义重述 | CKA + cosine | ✅ 确认 |
| 4 | 4-view eff rank 0.535 → 维度塌缩 | SVD + effective rank | ✅ 确认 |
| 5 | Leave-one-view Δ=0 → 完全冗余 | Leave-one-view-out | ✅ 确认 |
| 6 | Cross 净负向 | Rank-transition | ✅ 确认 |
| 7 | Valid→test ~44% 下降 = temporal shift | Distribution shift | ✅ 确认 |
| 8 | LLM 最优: lr=1e-4/do=0.1 或 lr=5e-4/do=0.3 | Grid search | 🟡 待 3-seed |
| 9 | MV 旧参数最优: lr=5e-4/do=0.3 (valid=0.0273) | Grid (旧PA) | 🟡 待正确参数 |
| 10 | Grid 三模型 valid MRR 相近 (0.0269~0.0279) | Grid search | 🟡 待 test 公平对比 |

---

## 六、实验规范（快速参考）

> 完整规范见 [`WSDM_experiment_spec.md`](WSDM_experiment_spec.md)

### 关键规则

1. **后台执行**: 所有训练通过 `nohup bash <script>.sh > logs/<name>_nohup.log 2>&1 &` 执行，对话框不持续跟进
2. **结果查收**: 新对话开始时先读取 nohup log / queue log / CSV 获取已完成结果
3. **选参规则**: 仅根据 valid set 选配置，禁止用 test 反复选参
4. **公平对比**: TF/LLM/MV 同网格同预算，per-source align 统一
5. **Preflight checklist**: 每次启动实验前过一遍 §七 的 21 条检查项

### Preflight 快速检查

```bash
ssh charlie@www.ultrapp.online 'echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader && echo "=== Processes ===" && ps aux | grep -E "two_phase|python.*scripts" | grep -v grep && echo "=== PYTHONPATH ===" && echo $PYTHONPATH && echo "=== Logs dir ===" && ls -d /home/charlie/project/RecBole/logs/ 2>/dev/null && echo "OK" || echo "MISSING: mkdir -p logs"'
```

---

## 七、依赖文件列表

新对话建立上下文时需读取的文件（按优先级排序）：

| 优先级 | 文件 | 用途 |
|:------:|------|------|
| **P0** | `paper_recsys/experiment_plan_20260719.md` | **本文件**，当日实验计划和上下文 |
| **P0** | `paper_recsys/experiment_status_20260718.md` | 上一日完整进展（grid 结果、分析、时间线） |
| **P1** | `paper_recsys/WSDM_experiment_spec.md` | 实验规范全文（评测协议、模型配置、preflight） |
| **P1** | `paper_recsys/WSDM_convergence_tracker.md` | 整体收敛追踪（版本迭代、stop-gate） |
| **P2** | `paper_recsys/experiment_status_20260717.md` | 7/17 分析结论详情 (distribution shift, CKA, rank-transition) |
| **P2** | `paper_recsys/0712zhidao.txt` | 老师指导：leakage-aware controlled study 方向 |
| **P2** | `paper_recsys/0717zhidao.txt` | 老师指导：实验方法论（小网格、3-seed、rolling valid） |
| **P3** | `run_5090_grid_fix.sh` | 当前后台运行的 fix 脚本 |
| **P3** | `run_5090_shared_grid.sh` | 原始 grid 脚本（已有 TF 和 LLM 结果） |
| **P3** | `scripts/two_phase_train.py` | 两阶段训练核心脚本 |

### 关键模型代码

| 文件 | 说明 |
|------|------|
| `recbole/model/sequential_recommender/sasrecalignv3.py` | TF-IDF / LLM 模型 |
| `recbole/model/sequential_recommender/sasrecalignmultiviewv3.py` | Multi-View 模型 |

### 关键配置文件

| 文件 | 说明 |
|------|------|
| `sasrec_align_base_stratified_v3_ts.yaml` | TF-IDF 配置 |
| `sasrec_align_v3_stratified_ts.yaml` | LLM 配置 |
| `sasrec_align_multi_view_v3_stratified_ts.yaml` | MV 配置 |

---

## 八、预期 7/19 结束时的产出

- [ ] TF-IDF fix grid 9 组完整结果（valid + test MRR/NDCG/HR + 分桶）
- [ ] MV 正确参数 grid 部分/全部结果
- [ ] 三模型 Grid 最优配置确定
- [ ] 更新 `experiment_status_20260719.md`
- [ ] 如 fix 全部完成，启动 3-seed 验证后台运行
