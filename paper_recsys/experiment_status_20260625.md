# 5090 实验状态梳理（2026-06-25）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-25 20:30（5090 实查 · Phase 3 已排队）  
> **GPU 状态**: RTX 5090 **训练中**（Grocery MV seed=42）  
> **当前主线**: Grocery MV seed=42 收尾 → **Phase 3 机制/分析自动拉起**（`run_5090_wait_and_post_pipeline_20260625.sh`）

**相关文档**:
- 结果汇总: [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)
- 分析与规划: [`experiment_plan_20260625.md`](experiment_plan_20260625.md)
- 导师意见: [`0625zhidao.txt`](0625zhidao.txt) · [`0617zhidao.txt`](0617zhidao.txt) · [`0618zhidao.txt`](0618zhidao.txt)

---

## 1. 进度总览

```
[✅] Beauty/Toys 主表 (SASRec, 4 seeds)
[✅] BC phaseb50 seed=2024（appendix 数据）
[✅] Grocery seed=2024 四配置
[✅] UniSRec Beauty 三配置 seed=2024（Base 作外部 baseline）
[🔄] Grocery multiseed 2025/2026/42（11/12 完成，seed42 MV 运行中）
[🔄] Phase 3–4 机制+GRU4Rec（主 pipeline 结束后自动执行）
[📋] GRU4Rec 第二 backbone（空档再补，未入 Phase 3）
```

| 实验 | 状态 | 备注 |
|------|------|------|
| UniSRec Beauty Base/AlignV3/AlignMV | ✅ 6/25 06:36 | Base → 主文外部 baseline |
| Grocery multiseed 2025 ×4 | ✅ 6/25 11:42 | MV MRR 3.03% |
| Grocery multiseed 2026 ×4 | ✅ 6/25 16:48 | MV MRR 3.02% |
| Grocery multiseed 42 | 🔄 3/4 | ID/TF-IDF/LLM ✅ · MV 19:10 起 |
| 主流水线 | 🔄 | `run_5090_main_pipeline_20260625_resume.sh` |

---

## 2. 核心结果快照（test @10 · seed=2024 为主）

### Beauty: SASRec MV vs UniSRec（0625 主文口径）

| 模型 | 角色 | MRR@10 | HR@10 | NDCG@10 |
|------|------|--------|-------|---------|
| SASRec ID-only | 内部 ablation | 2.16% | 5.31% | 2.91% |
| **SASRec MV-Align** | **主方法** | **3.18%** | **5.79%** | **3.79%** |
| **UniSRec Base** | **外部 baseline** | **2.47%** | **6.60%** | **3.44%** |

→ MV-Align MRR **+29% vs UniSRec Base**；UniSRec HR 更高 → MRR–HR trade-off 叙事。

### Grocery: 第三域 multiseed（MRR@10）

| Config | 2024 | 2025 | 2026 | 42 | mean±std† |
|--------|------|------|------|-----|-----------|
| ID-only | 2.08 | 2.08 | 2.08 | 2.08 | 2.08±0.00 |
| TF-IDF | 2.93 | 2.92 | 2.92 | 2.91 | 2.92±0.01 |
| TF-IDF+LLM | 2.95 | 2.94 | 2.96 | 2.97 | 2.96±0.01 |
| MV-Align | 3.01 | 3.03 | 3.02 | 🔄 | 3.02±0.01 |

† MV 42 完成后更新为 4-seed。

### Book-Crossing: appendix boundary case（seed=2024）

| Config | MRR@10 | HR@10 |  vs ID MRR |
|--------|--------|-------|------------|
| ID-only | 1.95% | 3.87% | — |
| MV-Align | **2.27%** | 3.87% | +16% |

→ 非 Amazon 外部域；MRR 有涨、HR 持平，适合 appendix stress test。

---

## 3. 5090 监控

```bash
ssh charlie@www.ultrapp.online
cd /home/charlie/project/RecBole
tail -f logs/main_pipeline_20260625.log
tail -f logs/grocery_MV_seed42.log
pgrep -af two_phase_train; nvidia-smi
```

---

## 4. Phase 3 自动队列（主 pipeline 结束后）

| 阶段 | 脚本 | GPU | 内容 |
|------|------|-----|------|
| 3A | `paper_recsys/run_post_analysis.sh` | 0 | main_table · Coverage · per-user t-test · gate 分桶 |
| 3B | `run_beauty_mechanism_save_scores.sh` | ✅ | Beauty ID/TF-IDF/MV eval-only → scores |
| 3C | `visualize_ablation.py` | 0 | concentration 图 + case study notes |
| **4** | `run_gru4rec_v3_multiseed_beauty_toys.sh` | ✅ | **5090+log10 并行**：log10=ID+TF-IDF(16) · 5090=LLM+MV(16) |

**log10 分流**（1080Ti 11GB · `log10` → 192.168.0.137）:
- 预检: `bash scripts/verify_log10_datasets.sh` ✅ 2026-06-25 已 sync Toys base.npy
- 5090 启动: `bash scripts/remote_start_log10_gru4rec_baselines.sh`
- 完成后回拉: `bash scripts/wait_and_pull_log10_gru4rec.sh` → `saved/gru4rec_*` + `run_metrics/`

**未入队**: UniSRec Base Toys（1 seed，可选）

---

## 5. 下一步（0625 优先级）

1. **自动**: Phase 3–4 跑完 → 检查机制图 + GRU4Rec 32 组 run_metrics
2. **写作**: Grocery 4-seed mean±std 进主表 · GRU4Rec 作 appendix 第二 backbone 小表
3. **可选**: UniSRec Base Toys（1 seed）

详见 [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md) § 后续优先级。

**文档同步**:
```bash
rsync -avz paper_recsys/*.md charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/
```
