# WSDM 实验分析与规划（2026-06-25）

> **来源**: `0617zhidao.txt` · `0618zhidao.txt` · `0625zhidao.txt` · `0201/0215/0311zhidao`  
> **更新**: 2026-06-25 19:30 CST（5090 实查 + 本地整理）  
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
| Beauty/Toys 主表 4-seed | ✅ seed 文件 + main_table | Coverage 进表（可提取）；机制图 |
| Grocery 第三域 | 2024–2026 ✅，42 MV 🔄 | 4-seed mean±std 定稿 |
| BC appendix | seed2024 phaseb50 ✅ | 小表 + boundary 叙事 |
| UniSRec 外部 baseline | Beauty Base ✅ | Toys Base（可选） |
| per-user t-test | ✅ npy 已有，脚本可跑 | 写入论文脚注 |
| 机制图（0617 三类） | 工具在 repo | **scores 未保存** → 见 plan §9.5 |
| GRU4Rec 第二 backbone | 📋 | 空档再补 |

> 完整数据盘点见 [`experiment_plan_20260625.md`](experiment_plan_20260625.md) **§9**。

---

## 3. 当前 5090 进度（2026-06-25 19:30）

```
[✅] UniSRec Beauty 三配置 seed=2024        6/25 06:36
[✅] Grocery multiseed 2025 ×4               6/25 11:42
[✅] Grocery multiseed 2026 ×4               6/25 16:48
[🔄] Grocery multiseed 42                    ID/TF-IDF/LLM ✅ · MV 19:10 起
```

**流水线**: `run_5090_main_pipeline_20260625_resume.sh`  
**预计**: seed=42 MV ~2.7h → pipeline **今晚 ~22:00** 完成

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
| **Grocery** | **第三域主表** | multiseed 11/12 🔄 |
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

---

## 9. 数据资产盘点（0617/0618 指标 · 5090 实查 2026-06-25）

> 结论先说：**Beauty/Toys 主表四 seed 完整可恢复**；Grocery/BC/UniSRec 在 `run_metrics/` JSON 中可拉；**0617 三类机制图大多需后处理或轻量复跑**，不是主表缺失。

### 9.1 指标需求 ↔ 数据来源对照

| 指标/分析 | 0617/0618 要求 | 数据在哪 | 能否直接拿到 | 若缺失怎么办 |
|-----------|----------------|----------|--------------|--------------|
| **主表 overall** | MRR/NDCG/HR@10 | `seed{42,2024,2025,2026}.txt` | ✅ 本地+5090 均有 | `compute_table_stats.py` 重算 |
| **主表分层** | MRR/NDCG/HR × new/few/freq | 同上 seed 文件 | ✅ | 同上 |
| **4-seed mean±std** | 0625/0311 | `main_table.txt` | ✅ 5090 可重算 | — |
| **seed 级 t-test** | 0311 sanity | `compute_table_stats.py` 内置 | ✅ 已出 MV vs LLM/TF-IDF | 注意 n=4 低 power |
| **per-user t-test** | 0311/0402 主文星号 | `saved/peruser/*_topk.npy` | ✅ **已有** Beauty/Toys | 见 §9.4 |
| **Coverage_new@10** | 0201/0215 长尾 | seed 文件列 + `run_metrics` JSON | ✅ 有原始值 | 需写提取脚本进 appendix |
| **Grocery 第三域** | 0625 | `run_metrics/20260625-*` + 2024 旧文件 | ✅ 2024–2026；42 MV 🔄 | pipeline 完成后汇总 |
| **BC appendix** | 0618 boundary | `run_metrics/*book*` Phase-B | ✅ seed2024 四配置 | 已在 results 文档 |
| **UniSRec 外部 baseline** | 0625 | `run_metrics/20260625-*UniSRec*` | ✅ Beauty seed2024 三配置 | Toys 可选补跑 |
| **Cross concentration** | 0617 机制图① | `ablation_study_doc/scores/*.npy` | ❌ **从未保存** | **轻量复跑** + `--save_test_scores` |
| **Gate 分桶** | 0617 机制图② | MV checkpoint `.pth` | ⚠️ 有 ckpt、**无分桶脚本** | **写 inference 脚本**，不必重训 |
| **Case study Top-K** | 0617 机制图③ | `saved/peruser/*_topk.npy` | ⚠️ 仅 LLM vs MV | 补 ID/TF-IDF topk 或 scores |
| **超参透明** | 0618 | yaml + phase 日志 | ✅ Grocery/BC 复刻 balanced | 写 appendix search protocol |

### 9.2 主表 Beauty/Toys — 完整可用

**权威来源**：`paper_recsys/seed42.txt` · `seed2024.txt` · `seed2025.txt` · `seed2026.txt`

- 本地与 5090 **均存在**（各 13 行：Beauty 4 配置 + Toys 4 配置 + 表头）
- 每行含 **80+ 列**：overall + 分层 + **Coverage_new/few/freq**（header 里有，但 `compute_table_stats.py` 目前只导出 12 列主表指标）
- **不在** `run_metrics/*.txt` JSON 里（Beauty/Toys SASRec 历史跑在 seed 文件时代完成）

**验证命令**（5090）：
```bash
cd paper_recsys && source ../scripts/recbole_env.sh && python compute_table_stats.py
# → 输出 main_table.txt，Beauty/Toys 4-seed mean±std 与 seed 级 t-test
```

**Beauty MV-Align 4-seed mean（5090 重算）**：MRR@10 **3.19±0.02%**，HR@10 **5.79±0.04%**，MRR_new **1.12±0.03%**

### 9.3 扩展数据集 — run_metrics JSON

| 数据集 | run_metrics 条目 | 四配置齐全？ | 备注 |
|--------|------------------|-------------|------|
| Grocery | 28+ 条（0624–0625） | ✅ 2024–2026；42 缺 MV | MV seed42 今晚完成 |
| Book-Crossing | 20+ 条 | ✅ seed2024 phaseb50 | ID/TF-IDF/LLM/MV Phase-B 均有 |
| UniSRec Beauty | 3 条 Phase-A | Base/AlignV3/AlignMV | 无 SASRec 对照 JSON |
| Food | 5 条 | 部分 | appendix 对照，非主表 |

**Grocery Coverage_new@10 示例（MV test）**：2024 **12.26%** · 2025 **13.32%** · 2026 **12.75%**

### 9.4 per-user 显著性 — 已有，可直接写进论文

5090 上 `saved/peruser/` 已有 4 个文件（seed=2025 重跑时生成）：

| 文件 | 用途 |
|------|------|
| `beauty_tfidf_llm_topk.npy` | Beauty baseline top-k |
| `beauty_mv_7b_topk.npy` | Beauty MV top-k |
| `toys_tfidf_llm_topk.npy` | Toys baseline |
| `toys_mv_7b_topk.npy` | Toys MV |

**已验证可跑**（5090）：
```bash
python paper_recsys/compute_peruser_significance.py \
  --model_a saved/peruser/beauty_tfidf_llm_topk.npy \
  --model_b saved/peruser/beauty_mv_7b_topk.npy \
  --label_a "TF-IDF+LLM" --label_b "MV-Align(7B)"
# Beauty @10: HR p=1.56e-05***, NDCG p=7.3e-04***, MRR p=0.032*
```

Toys @10 同样显著（MRR p≈0.05*，HR/NDCG p<0.001**）。

### 9.5 0617 机制分析 — 缺口与最小复跑方案

#### ① Cross top-concentration（Gini / entropy / head share）

| 状态 | ❌ 无数据 |
|------|----------|
| 原因 | 历史训练**未加** `--save_test_scores`；5090 `find` 无 `*test_score*` / `scores/` |
| **最小复跑** | Beauty · **seed=2024** · 3 配置：**ID / TF-IDF / MV** · eval only 或短 phase-B + `--save_test_scores` |
| 工具 | `ablation_study_doc/visualize_ablation.py --plot-score-dist` |
| 预估 | ~3 × 1.5h ≈ **4–5h GPU**（可串行） |

#### ② View/Gate 分桶（new/few/freq × 4 view）

| 状态 | ⚠️ checkpoint 有，分析脚本无 |
|------|---------------------------|
| 已有 | `saved/peruser_runs/beauty_mv_7b/*.pth`（seed2025）；模型内 `text_view_gate_params` |
| **不需要完整重训** | 写脚本：load ckpt → 按 item 频次桶聚合 4 view gate |
| 预估 | **0 GPU**（纯后处理，半天开发） |

#### ③ Case study（TF-IDF / LLM / MV Top-K 变化）

| 状态 | ⚠️ 部分 |
|------|--------|
| 已有 | LLM vs MV 的 per-user topk（Beauty/Toys） |
| 缺口 | ID-only、TF-IDF 单列的 Top-K 对比；或需 full score 挑 tail item |
| **方案 A** | 从已有 topk 做 2 模型 case study（弱化版） |
| **方案 B** | 同 ① 的 `save_test_scores` 跑 TF-IDF + MV + ID，离线挑 case |
| 预估 | A=0h；B 并入 ① 复跑 |

### 9.6 Coverage_new 进表 — 不需复跑

- **Beauty/Toys**：从 `seed*.txt` 解析 `Coverage_new@10`（列在 header 中）
- **Grocery/BC**：从 `run_metrics` JSON 的 `Coverage_new@10` 字段直接读
- 动作：写一个 `extract_coverage.py` 汇总进 appendix 表（**0 GPU**）

### 9.7 复跑优先级（仅机制/补充项）

| 优先级 | 任务 | GPU | 说明 |
|--------|------|-----|------|
| **0** | 从 seed/run_metrics **提取 Coverage** | 0 | 今晚可做 |
| **0** | **per-user t-test** 结果写入主表脚注 | 0 | 数据已有 |
| **1** | Beauty **save_test_scores** ×3（ID/TF-IDF/MV） | ~5h | 0617 机制图①③ |
| **2** | **Gate 分桶脚本** + 跑 beauty_mv ckpt | 0 | 0617 机制图② |
| **3** | Case study 成图 | 0 | 依赖 1 或现有 topk |
| **4** | UniSRec Base Toys（1 seed） | ~1.6h | 0625 可选 |
| **5** | GRU4Rec Beauty+Toys **4 seed × 4 config** | ~60–80h | 第二 backbone；**已入 Phase 4 队列** |

**5090 自动排队**（主 pipeline 结束后）:
```bash
# 已在跑的主 pipeline 结束后自动拉起（无需手动）:
nohup bash run_5090_wait_and_post_pipeline_20260625.sh \
  >> logs/wait_post_pipeline_20260625_nohup.log 2>&1 &

# 或新跑完整 pipeline 时末尾已 chain:
bash run_5090_main_pipeline_20260625.sh   # → Phase 3 自动执行
```

Phase 3 脚本: `run_5090_post_main_pipeline_20260625.sh`
- 3A: `paper_recsys/run_post_analysis.sh`（Coverage / t-test / gate）
- 3B: `run_beauty_mechanism_save_scores.sh`（eval-only scores）
- 3C: `visualize_ablation.py` concentration 图
- **4**: `run_gru4rec_v3_multiseed_beauty_toys.sh`（Beauty+Toys × seed{42,2024,2025,2026} × ID/TF-IDF/LLM/MV）

**明确不需要为主表复跑**：Beauty/Toys 四 seed 四配置（seed 文件完整）；Grocery/BC 已有 run_metrics。

