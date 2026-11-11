# 实验计划 — 2026-07-24

> 创建: 2026-07-23 23:40  
> 上一日进展: [`experiment_status_20260723.md`](experiment_status_20260723.md)  
> 老师指导: [`0723zhidao.txt`](0723zhidao.txt)  
> 规范: [`WSDM_experiment_spec.md`](WSDM_experiment_spec.md)

---

## 〇、实验背景（新对话必读）

### 叙事（已锁定，勿再改方向）

**When More Text Is Not Better: A Leakage-Aware Study of Text Representations for Sequential Recommendation**

三问题：
1. LOO vs GTS 是否改变 text-enhanced SR 结论
2. TF / LLM / MV 谁在 temporal shift 下更鲁棒
3. MV 无增益 = view redundancy（非容量）

**停止**：扩 MV、调 boost、加 Cross、追求层级。

### 已完成证据

| 项 | 结论 | 文件 |
|----|------|------|
| Main 3-seed | TF≥LLM≈MV | `logs/stopgate_3seed_beauty_metrics.csv` |
| Early60 3-seed | TF≥MV≥LLM | `logs/early60_stopgate_beauty.csv` |
| Warm audit | Beauty test unseen 27.8%；Toys 34.5% | `paper_recsys/warm_target_audit_*.md` |
| Mechanism | CKA/LOVO/rank-transition | `paper_recsys/rank_transition_beauty_*.md` |

---

## 一、后台任务（勿在对话框跟进程）

| 机器 | 脚本 | 内容 | 查收命令 |
|------|------|------|----------|
| **5090** | `run_5090_wsdm_wrapup_serial.sh` | Toys GTS ID/TF/LLM/MV × 3 seeds | `tail -20 logs/wsdm_wrapup_5090_queue.log` |
| **log10** | `run_log10_wsdm_wrapup_serial.sh` | Beauty ID × 3 seeds → UniSRec | `tail -20 logs/wsdm_wrapup_log10_queue.log` |

```bash
# 5090 快照
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader && tail -15 logs/wsdm_wrapup_5090_queue.log && cat logs/wsdm_wrapup_toys_gts.csv 2>/dev/null | tail -20'

# log10 快照
ssh charlie@192.168.0.107 'cd /home/charlie/project/RecBole && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader && tail -15 logs/wsdm_wrapup_log10_queue.log && cat logs/wsdm_wrapup_log10.csv'
```

---

## 二、今日待办（优先级）

### P0 — 查收后台结果
1. 汇总 Toys GTS 已完成组的 valid/test MRR（与 Beauty Main 排序对照）
2. 汇总 Beauty ID-only 3-seed mean±std
3. 确认 UniSRec 是否完成 / 失败原因

### P0 — 主表口径落地
1. 在规范中写死：**主表 = warm-target only**
2. 规划 warm-only 重评估（离线 mask candidates 或 eval 脚本改造）——优先复用已有 ckpt，避免全量重训

### P1 — Clean LOO vs GTS（0723 第二组）
1. 新建 clean LOO yaml（统一代码，no-boost/no-Cross）
2. Beauty LOO × ID/TF/LLM/MV × 3 seeds 串行脚本（接在 Toys 队列后，或 Toys 完成后启动）
3. 论文 Table：LOO 排序 vs GTS 排序是否反转

### P2 — 机制定稿
1. 主文：CKA / eff-rank / LOVO + temporal shift（含 catalog turnover）
2. Cross → Appendix（已有 rank_transition）

---

## 三、依赖文件

| 优先级 | 路径 |
|:------:|------|
| P0 | `paper_recsys/0723zhidao.txt` |
| P0 | `paper_recsys/experiment_status_20260723.md` |
| P0 | `paper_recsys/WSDM_convergence_tracker.md` |
| P0 | `run_5090_wsdm_wrapup_serial.sh` / `run_log10_wsdm_wrapup_serial.sh` |
| P1 | `logs/stopgate_3seed_beauty_metrics.csv` / `logs/early60_stopgate_beauty.csv` |
| P1 | `scripts/audit_warm_targets.py` |
| P2 | `scripts/two_phase_train.py` + Toys/Beauty YAML |

---

## 四、预期产出（7/24 结束时）

- [ ] Toys GTS 部分/全部结果表
- [ ] Beauty ID-only 3-seed
- [ ] 更新 `experiment_status_20260724.md`
- [ ] 若 Toys 完成：启动 Beauty clean LOO 队列（或写好脚本待命）
- [ ] warm-only 重评估方案落地（脚本或配置）
