# 5090 实验状态梳理（2026-06-25）

> **⚠️ 已 supersede**: 最新进展见 [`experiment_status_20260626.md`](experiment_status_20260626.md)（2026-06-26 08:20）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-25 22:25（5090 + log10 实查）  
> **GPU 状态**: RTX 5090 **训练中**（Phase 4 · Beauty GRU4Rec TF-IDF+LLM seed=42）  
> **当前主线**: Phase 4 GRU4Rec 并行（5090 text 16 + log10 baseline 16）

**相关文档**:
- 结果汇总: [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)
- log10 状态: [`experiment_status_log10_20260625.md`](experiment_status_log10_20260625.md)
- 分析与规划: [`experiment_plan_20260625.md`](experiment_plan_20260625.md)
- 导师意见: [`0625zhidao.txt`](0625zhidao.txt) · [`0617zhidao.txt`](0617zhidao.txt) · [`0618zhidao.txt`](0618zhidao.txt)

---

## 1. 进度总览

```
[✅] Beauty/Toys 主表 (SASRec, 4 seeds)
[✅] BC phaseb50 seed=2024（appendix 数据）
[✅] Grocery 四配置 × 4 seeds（12/12 完成 6/25 21:54）
[✅] UniSRec Beauty 三配置 seed=2024（Base 作外部 baseline）
[✅] Phase 3A 后处理（表格 · Coverage · per-user t-test · gate 分桶）
[✅] Phase 3B 机制 scores（Beauty ID/TF-IDF/MV top-100）
[✅] Phase 3C 分数分布图（id_vs_tfidf · tfidf_vs_mv）
[🔄] Phase 4 GRU4Rec（5090 LLM+MV 1/16 · log10 ID+TF-IDF 1/16）
[📋] UniSRec Base Toys（1 seed，可选，未入队）
```

| 实验 | 状态 | 备注 |
|------|------|------|
| 主流水线 Phase 1–2 | ✅ 6/25 21:54 | UniSRec Beauty×3 + Grocery multiseed 全部完成 |
| Grocery MV seed=42 | ✅ 6/25 21:53 | test MRR@10 **3.00%** |
| Phase 3A–3C | ✅ 6/25 22:03 | 机制分析完成；3B 曾 OOM，已 fix top-k-only |
| Phase 4 5090 text | 🔄 1/16 | `run_5090_phase4_text_only.sh` · Beauty LLM seed=42 Phase-A ep~7/20 |
| Phase 4 log10 baseline | 🔄 1/16 | 见 [`experiment_status_log10_20260625.md`](experiment_status_log10_20260625.md) |

---

## 2. 核心结果快照（test @10 · seed=2024 为主）

### Beauty: SASRec MV vs UniSRec（0625 主文口径）

| 模型 | 角色 | MRR@10 | HR@10 | NDCG@10 |
|------|------|--------|-------|---------|
| SASRec ID-only | 内部 ablation | 2.16% | 5.31% | 2.91% |
| **SASRec MV-Align** | **主方法** | **3.18%** | **5.79%** | **3.79%** |
| **UniSRec Base** | **外部 baseline** | **2.47%** | **6.60%** | **3.44%** |

→ MV-Align MRR **+29% vs UniSRec Base**；UniSRec HR 更高 → MRR–HR trade-off 叙事。

### Grocery: 第三域 multiseed（MRR@10）— 4 seed 齐全

| Config | 2024 | 2025 | 2026 | 42 | mean±std |
|--------|------|------|------|-----|-----------|
| ID-only | 2.08 | 2.08 | 2.08 | 2.08 | 2.08±0.00 |
| TF-IDF | 2.93 | 2.92 | 2.92 | 2.91 | 2.92±0.01 |
| TF-IDF+LLM | 2.95 | 2.94 | 2.96 | 2.97 | 2.96±0.01 |
| MV-Align | 3.01 | 3.03 | 3.02 | **3.00** | **3.02±0.01** |

### Book-Crossing: appendix boundary case（seed=2024）

| Config | MRR@10 | HR@10 |  vs ID MRR |
|--------|--------|-------|------------|
| ID-only | 1.95% | 3.87% | — |
| MV-Align | **2.27%** | 3.87% | +16% |

---

## 3. Phase 3 机制分析（✅ 6/25 22:03）

| 阶段 | 产出 | 路径 |
|------|------|------|
| 3A | main_table · Coverage · per-user t-test · gate 分桶 | `paper_recsys/run_post_analysis.sh` |
| 3B | top-100 scores ×3（ID / TF-IDF / MV） | `ablation_study_doc/scores/beauty_*_mechanism_phase_b_topk_scores.npy` |
| 3C | 分数分布对比图 | `ablation_study_doc/figures/id_vs_tfidf/` · `tfidf_vs_mv/` |

**3B 备注**: 全量 score 矩阵 OOM（~46GB RAM）→ `trainer.py` 改为 batch 内 top-100 增量保存；`save_peruser_topk` 已关闭（per-user 数据 Phase 3A 已有）。

---

## 4. Phase 4 GRU4Rec（🔄 进行中）

**分工**:
| 机器 | 任务 | 脚本 | 进度 |
|------|------|------|------|
| **log10** | ID + TF-IDF · Beauty/Toys × 4 seeds | `run_log10_gru4rec_baselines.sh` | **1/16** |
| **5090** | TF-IDF+LLM + MV · Beauty/Toys × 4 seeds | `run_5090_phase4_text_only.sh` | **1/16** |

**5090 当前**: Beauty GRU4Rec TF-IDF+LLM seed=42 · Phase-A epoch ~7/20 · GPU ~95%

**5090 完成后**: 自动 `scripts/wait_and_pull_log10_gru4rec.sh` 回拉 log10 的 `saved/gru4rec_*` + `run_metrics/`

**已修复（6/25 22:20）**:
- `gru4rec_align_toys_qwen_stratified_v3.yaml` LLM 路径 → `qwen2.5_7b.base.npy`
- `two_phase_run_gru4rec*.sh` 续行空行 bug → 5090 + log10 已同步
- `remote_start_log10` ssh 阻塞 → `ssh -n` + `</dev/null>`

**预估完成**: 并行 wall-clock ~**6/28–6/29**（各 ~2–3 天串行 16 runs）

---

## 5. 5090 监控

```bash
ssh charlie@www.ultrapp.online
cd /home/charlie/project/RecBole
tail -f logs/gru4rec_beauty_text_seed42.log      # Phase 4 text
tail -f logs/post_main_pipeline_20260625.log   # 总进度
pgrep -af two_phase_train; nvidia-smi
```

---

## 6. 下一步

1. **自动**: Phase 4 32 组跑完 → 汇总 GRU4Rec run_metrics → appendix 第二 backbone 小表
2. **写作**: Grocery 4-seed 进主表 · 机制图（concentration + gate）进 WSDM 机制节
3. **可选**: UniSRec Base Toys（1 seed）· case study 人工挑选

**文档同步**:
```bash
rsync -avz paper_recsys/*.md charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/
rsync -avz paper_recsys/*.md charlie@192.168.0.137:/home/charlie/project/RecBole/paper_recsys/
```
