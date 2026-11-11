# 实验计划 — 2026-07-27

> 创建: 2026-07-26 16:05  
> 上一日进展: [`experiment_status_20260726.md`](experiment_status_20260726.md)  
> 老师指导: [`0723zhidao.txt`](0723zhidao.txt)  
> 规范: [`WSDM_experiment_spec.md`](WSDM_experiment_spec.md)

---

## 〇、实验背景（新对话必读）

### 叙事（已锁定）

**When More Text Is Not Better: A Leakage-Aware Study of Text Representations for Sequential Recommendation**

三核心问题：
1. LOO vs GTS 是否改变 text-enhanced SR 结论 → **已有初步数据支持：LOO 模型排序翻转**
2. TF / LLM / MV 谁在 temporal shift 下更鲁棒
3. MV 无增益 = view redundancy（非容量）

**停止**：扩 MV、调 boost、加 Cross、追求层级。

### 已完成实验总览

| 实验 | 数据集 | 状态 | 核心结论 |
|------|--------|:----:|----------|
| GTS Grid + 3-seed stop-gate | Beauty | ✅ | TF 0.0165 ≥ LLM 0.0161 ≈ MV 0.0160 |
| GTS Early60 3-seed | Beauty | ✅ | TF ≥ MV ≥ LLM；两窗口 stop-gate 触发 |
| GTS 3-seed | Toys | ✅ | TF 0.0113 > LLM 0.0107 ≈ MV 0.0109；跨域验证 |
| Beauty ID-only 3-seed | Beauty | ✅ | GTS 0.0106；LOO 0.0175 |
| Clean LOO 3-seed | Beauty | 🟡 7/12 完成 | LOO MRR ~1.5-1.7× GTS；MV 排序翻转 |
| Warm-target audit | Beauty/Toys | ✅ | Beauty unseen 27.8%；Toys 34.5% |
| 嵌入塌缩诊断 | Beauty | ✅ | MV eff-rank 136/256；view 高冗余 |
| Rank-transition / CKA / LOVO | Beauty | ✅ | Cross 无稳定趋势；MV view 冗余 |

---

## 一、后台任务（勿在对话框跟进）

| 机器 | 脚本 | 内容 | 查收命令 | 预计完成 |
|------|------|------|----------|----------|
| **5090** | `run_5090_beauty_loo_resume.sh` | Beauty LOO: MV s2024 + seed2026 = 5 runs | `tail -10 logs/beauty_loo_queue.log` | **~22:10 (7/26)** |
| **5090** | `run_5090_toys_loo_serial.sh` | Toys LOO: ID/TF/LLM/MV × 3 seeds = 12 runs | `tail -10 logs/toys_loo_queue.log` | **~12:10 (7/27)** |

> Toys LOO 通过 watcher 自动接续 Beauty LOO，无需手动启动。

```bash
# 5090 快照
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader && echo "=== beauty ===" && tail -5 logs/beauty_loo_queue.log && echo "=== toys ===" && tail -5 logs/toys_loo_queue.log 2>/dev/null && echo "=== beauty csv ===" && cat logs/beauty_loo_results.csv && echo "=== toys csv ===" && cat logs/toys_loo_results.csv 2>/dev/null'
```

---

## 二、今日待办（优先级）

### P0 — 查收 Beauty Clean LOO 结果
1. 确认 12/12 runs 全部完成（`logs/beauty_loo.DONE` 存在）
2. 汇总 LOO 3-seed mean±std
3. 产出论文核心 Table：**LOO vs GTS protocol 对照**
4. 分析模型排序是否翻转（LOO 下 MV 排名 vs GTS 下 MV 排名）

### P0 — 查收 Toys Clean LOO 结果
1. 确认 12/12 runs 全部完成（`logs/toys_loo.DONE` 存在）
2. 汇总 Toys LOO 3-seed mean±std
3. 产出 Toys LOO vs GTS 对照 Table
4. **已部署**：`run_5090_toys_loo_serial.sh` 通过 watcher 自动接续 Beauty LOO，预计 7/27 ~12:10 完成

### P1 — Warm-target Only 重评估
1. 用已有 GTS ckpt 在 warm-target subset 上重新 eval（不需重训）
2. 需改造 eval 脚本或添加 target filter
3. 对比 warm-only vs all-target 的指标差异

### P2 — 机制定稿
1. 整理 Figure 数据：SVD 谱图 + effective rank + CKA heatmap
2. Temporal shift 可视化：popularity / unseen share / catalog turnover
3. Cross → Appendix 材料整理

---

## 三、依赖文件

| 优先级 | 路径 |
|:------:|------|
| P0 | `paper_recsys/0723zhidao.txt` — 老师指导（核心方向）|
| P0 | `paper_recsys/experiment_status_20260726.md` — 最新进展 |
| P0 | `paper_recsys/WSDM_convergence_tracker.md` — 收敛追踪 |
| P0 | `paper_recsys/WSDM_experiment_spec.md` — 实验规范 |
| P0 | `logs/beauty_loo_results.csv` — LOO 结果 |
| P0 | `logs/wsdm_wrapup_toys_gts.csv` — Toys GTS 结果 |
| P1 | `logs/stopgate_3seed_beauty_metrics.csv` — Beauty GTS stop-gate |
| P1 | `logs/early60_stopgate_beauty.csv` — Beauty Early60 |
| P2 | `scripts/audit_warm_targets.py` — warm-target 审计 |

---

## 四、预期产出（7/27 结束时）

- [ ] Beauty LOO 3-seed 完整结果表（~22:10 7/26 完成）
- [ ] Toys LOO 3-seed 完整结果表（~12:10 7/27 完成）
- [ ] LOO vs GTS protocol 对照 Table × 2 数据集（论文核心）
- [ ] 更新 `experiment_status_20260727.md`
- [ ] 更新 `WSDM_convergence_tracker.md`（LOO 结果归档）

---

## 五、论文进度自检

| 组件 | 需要的数据 | 状态 |
|------|-----------|:----:|
| Table 1: GTS Main (Beauty) | ID/TF/LLM/MV 3-seed | ✅ |
| Table 2: GTS Main (Toys) | ID/TF/LLM/MV 3-seed | ✅ |
| Table 3: LOO vs GTS (Beauty) | Clean LOO 3-seed | 🟡 差 5 runs → ~22:10 (7/26) |
| Table 4: LOO vs GTS (Toys) | Clean LOO 3-seed | 🟡 已部署 → ~12:10 (7/27) |
| Figure: SVD spectrum + eff-rank | 已有 json | ✅ 数据就绪 |
| Figure: Temporal shift | distribution_shift json | ✅ 数据就绪 |
| Figure: CKA heatmap | 已有数据 | ✅ 数据就绪 |
| Appendix: Cross analysis | rank-transition data | ✅ 数据就绪 |
| Warm-target reeval | 需 eval 脚本改造 | ❌ 未开始 |
