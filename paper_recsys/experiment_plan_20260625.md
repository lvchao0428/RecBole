# WSDM 实验分析与规划（2026-06-25）

> **来源**: `0617zhidao.txt` · `0618zhidao.txt` · `0625zhidao.txt` · `0201/0215/0311zhidao`  
> **更新**: 2026-06-25 17:30 CST（5090 实查 + 本地整理）  
> **关联**: [`experiment_status_20260625.md`](experiment_status_20260625.md) · [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)

---

## 0. 文章定位（0617 / 0618）

WSDM 版定位：**full-ranking 下文本特征、cross、alignment 如何影响序列推荐行为的机制研究**，而非 pursuit SOTA 或堆 ablation。

- 核心 observation：cross 让 **HR（覆盖/召回）** 与 **MRR/NDCG（头部排序）** 出现分叉
- 表述：用 **full-ranking scoring**，不与 semantic ID + beam search 正面对打
- SE Block：降级为 implementation detail，不占主叙事
- 超参：三数据集可 dataset-specific，但必须 **同一 search space + 验证协议 + 预算** 透明写出；Grocery/BC 复刻 Beauty balanced default 是加分项

---

## 1. 需观测指标清单

### 1.1 主表 / 分层指标（0625 + 早期指导）

| 指标组 | 具体指标 | 观测目的 | 写作口径 |
|--------|----------|----------|----------|
| Overall 排序 | MRR@10, NDCG@10 | re-ranking 质量 | 主表必报 |
| Overall 覆盖 | HR@10, Recall@10 | candidate-generation / 覆盖 | cross 与 MRR 分叉的核心 |
| 冷启动分层 | MRR_new, NDCG_new | 冷启动排序质量 | Beauty/Toys 两边都应最好 |
| 冷启动分层 | HR_new | 冷启动命中 | Beauty：**narrows gap**（1.76 vs 1.88）；Toys：**exceeds**（2.04 vs 1.92） |
| 中/高频 | MRR_few/freq, HR_few/freq | trade-off 是否牺牲 frequent | appendix 或主表 few 列 |
| 稳定性 | 4-seed mean±std | 可复现性 | Grocery multiseed 跑完后补 |
| 显著性 | paired t-test（per-user） | vs strongest baseline | HR/NDCG **\*\***，MRR **\*** |

**0625 新强调**：Grocery / BC 也要报 **MRR/NDCG 稳、HR narrow gap**，与 Beauty/Toys 同一分层口径。

### 1.2 机制分析指标（0617 · 后处理为主，不必大网格训练）

| 类别 | 指标/分析 | 做法 | 现状 |
|------|-----------|------|------|
| Cross top-concentration | Top-20/100 分数 Gini、entropy、head item 占比 | 对比 ID / TF-IDF / MV 的 score distribution | 有 `save_test_scores` + `ablation_study_doc/visualize_ablation.py`，Beauty 未系统跑 |
| View/Gate 分桶 | 4 view 在 new/few/freq 下平均 gate 权重 | 加载 MV checkpoint，按 item 频次桶统计 | 模型有 gate，缺分桶汇总脚本 |
| Case study | 同一 user 下 TF-IDF vs LLM vs MV 的 Top-K 变化 | 挑 new/tail item，展示低频 item 被 MV 往前拉 | 未做 |

### 1.3 长尾 / 覆盖补充（0201 / 0215）

| 指标 | 用途 | 现状 |
|------|------|------|
| Coverage_new@10 | 证明长尾/new 未被牺牲 | run summary JSON 已有 |
| Recall@100（B 通道） | 更宽候选覆盖 | 需确认 stratified eval 统一导出 |
| HR@10 vs MRR@10 Pareto | 4 旋钮敏感性，Pareto 前沿 | `plot_sensitivity.py` 有基础 |

---

## 2. 已完成 vs 缺失

| 维度 | 已完成 | 还缺 |
|------|--------|------|
| Beauty/Toys 主表 4-seed | ✅ | paired t-test 汇总、Coverage 进表 |
| Grocery 第三域 | 2024/2025/2026 ✅，42 🔄 | 4-seed mean±std |
| BC appendix | seed2024 phaseb50 ✅ | 小表 + boundary 叙事 |
| UniSRec 外部 baseline | Beauty Base ✅ | Toys Base（可选） |
| 机制图（0617 三类） | 工具在 repo | Beauty 上系统跑一遍 |
| GRU4Rec 第二 backbone | 📋 | 空档再补 |

---

## 3. 当前 5090 进度（2026-06-25 17:30）

```
[✅] UniSRec Beauty 三配置 seed=2024        6/25 06:36
[✅] Grocery multiseed 2025 ×4               6/25 11:42
[✅] Grocery multiseed 2026 ×4               6/25 16:48
[🔄] Grocery multiseed 42                    ID ✅ 17:08 · TF-IDF 运行中
```

**流水线**: `run_5090_main_pipeline_20260625_resume.sh`  
**预计**: seed=42 剩余 ~5h → 今晚 ~22:00 前后 pipeline 完成

---

## 4. 后续规划（按优先级）

### P0 — 等 pipeline（今晚，无新设计）

- Grocery seed=42 剩余 3 配置 → **4-seed mean±std** → 更新主表第三域
- 产出：Grocery overall + new/few/freq 与 Beauty/Toys 同格式

### P1 — 分析为主（不占用 5090 训练）

| # | 任务 | 对应指导 | 预估 |
|---|------|----------|------|
| 1 | 从 logs/run_metrics 拉 **Grocery 4-seed 分层表** + mean±std | 0625 | 1–2h |
| 2 | Beauty **MV vs UniSRec Base** 外部 baseline 表 | 0625 | ✅ 见 results 文档 |
| 3 | **Cross concentration 图**（ID/TF-IDF/MV） | 0617 | 半天 |
| 4 | **Gate 分桶分析**（new/few/freq × 4 view） | 0617 | 半天 |
| 5 | **2–3 case study**（new item Top-K） | 0617 | 1天 |
| 6 | **Coverage_new@10** 进 appendix | 0201 | 2h |
| 7 | **paired t-test** vs strongest baseline | 0311 | 2h |

### P2 — 轻量训练（5090 空档，0625 顺序）

| 优先级 | 实验 | 目的 | 建议 |
|--------|------|------|------|
| 1 | ~~UniSRec Beauty Base~~ | 外部 text baseline | ✅ |
| 2 | **UniSRec Base on Toys** | 回应 competitiveness | ~1.6h/seed |
| 3 | **GRU4Rec + MV** Beauty/Toys | 第二 backbone sanity | UniSRec 后、空档再补 |
| 4 | UniSRec AlignV3/MV | portability | Beauty 已有，appendix 即可 |
| 5 | all-MiniLM 第二 encoder | encoder ablation | 0625 优先级最低 |

### P3 — 论文结构（0618，非实验）

- SE 从主方法图弱化 → ablation 一句
- 召回 → **full-ranking scoring**
- 超参 → shared-default sanity + validation-selected 放 appendix/repo
- BC → appendix 小表 + boundary case

---

## 5. 主文指标最小集合（WSDM）

**主表（每数据集）**
- Overall: HR@10, NDCG@10, MRR@10（mean±std）
- New stratum: HR_new, NDCG_new, MRR_new
- 外部 baseline 行: **UniSRec Base**

**机制节（0617）**
- Cross: Top-K score Gini/entropy + head item share
- MV: gate 权重 × strata
- 1–2 case study 图

**Appendix**
- few/freq 分层、Coverage_new@10、BC boundary 小表、UniSRec portability、超参 search protocol

---

## 6. 数据集角色（0625）

| 数据集 | 角色 | 状态 |
|--------|------|------|
| Beauty + Toys | 主表（Amazon 两域，4-seed） | ✅ |
| **Grocery** | **第三域主表** | multiseed 🔄 |
| Book-Crossing | external stress test / appendix | seed2024 ✅，不必 multiseed |
| Food | 非 Amazon 对照 | ✅ |
| UniSRec Base | 外部 text-enhanced baseline | Beauty ✅ |

---

## 7. 执行顺序（收敛版）

```text
本周
  [今晚] Grocery seed=42 跑完 → 4-seed 主表
  [明后天] 机制分析三连（concentration + gate + case study）on Beauty
  [并行] Coverage_new、paired t-test、Grocery 分层表

有空档
  UniSRec Base Toys（1 seed 先行）

可选
  GRU4Rec sanity（Beauty 1 seed 四配置）
  BC 小表进 appendix（已有 seed2024）
```

---

## 8. 机制分析工具索引

| 分析 | 路径 |
|------|------|
| Score distribution / concentration | `ablation_study_doc/visualize_ablation.py` |
| 保存 test scores | `--save_test_scores` on `run_recbole.py` / `two_phase_train.py` |
| Gate 权重 | checkpoint `text_view_gate_params` · `markdown/GATE_COMPARISON_SUMMARY.md` |
| 敏感性 Pareto | `paper_recsys/scripts/plot_sensitivity.py` |
| 分层 metrics 拉取 | `tools/pull_bc_metrics.py` · run_metrics JSON |
