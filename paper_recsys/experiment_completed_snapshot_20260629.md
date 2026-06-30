# 已完成实验总览（2026-06-29）

> 更新时间: 2026-06-29 21:00 CST  
> 口径: 只汇总**已完成且可引用**的实验数据；正在运行或尚未整理成正式结果的任务单独放在文末。  
> 主要来源: `experiment_results_all_20260625.md` · `gru4rec_phase4_matrix_20260629.md` · `peruser_significance_20260625.txt` · `coverage_table_20260625.txt`

---

## 1. 当前已完成内容

| 模块 | 状态 | 备注 |
|------|------|------|
| SASRec 主表（Beauty/Toys 4 seeds） | ✅ | 主表完整 |
| Grocery 4-seed 主表 | ✅ | 第三域主表完整 |
| Book-Crossing（seed=2024） | ✅ | appendix / boundary case |
| UniSRec Beauty（Base / Align / MV） | ✅ | external text baseline + portability |
| UniSRec Beauty balanced（Align / MV） | ✅ | 6/29 补齐 |
| GRU4Rec Phase 4 text（LLM+MV） | ✅ | 16/16 |
| GRU4Rec Phase 4 baseline（ID+TF-IDF） | ✅ | 16/16 |
| Phase 3 机制分析：concentration / gate / t-test / coverage | ✅ | case study 仍待整理 |

---

## 2. SASRec 主结果结论

### 2.1 跨数据集核心现象：ID-only vs MV-Align

| 数据集 | ID-only MRR@10 | MV-Align MRR@10 | MRR 提升 | ID-only HR@10 | MV-Align HR@10 | HR 变化 |
|--------|----------------|-----------------|----------|---------------|----------------|---------|
| Beauty | 0.0244 | 0.0283 | +16.0% | 0.0540 | 0.0537 | -0.6% |
| Toys | 0.0188 | 0.0220 | +17.0% | 0.0410 | 0.0399 | -2.7% |
| Book-Crossing | 0.0195 | 0.0227 | +16.4% | 0.0387 | 0.0387 | ±0.0% |
| Grocery | 0.0208 | 0.0301 | +44.7% | 0.0636 | 0.0602 | -5.3% |

### 2.2 当前最稳定的主文叙事

- `cross + align` 在四个数据集上都稳定提升 `MRR/NDCG`
- `HR-MRR trade-off` 普遍存在，但强弱因数据集而异
- Grocery 是最强的第三域主表证据；Book-Crossing 更适合作为 appendix / stress test

---

## 3. Grocery 4-seed 主表

> Grocery 现在已经可以作为主文第三数据集使用。

| Config | MRR@10 mean±std | HR@10 mean±std | 结论 |
|--------|------------------|----------------|------|
| ID-only | 2.08±0.00 | 6.36±0.00 | baseline |
| TF-IDF | 2.92±0.01 | 5.77±0.03 | 排序明显提升，HR 下降 |
| TF-IDF+LLM | 2.96±0.01 | 5.81±0.02 | 比 TF-IDF 再好一点 |
| MV-Align | 3.02±0.01 | 6.05±0.02 | MRR 最优，HR 相比 ID-only 仅 narrow gap |

结论可直接写成：

- `MRR/NDCG` 很稳，MV-Align 相对 ID-only 约 `+45%`
- `HR@10` 没有完全超过 ID-only，但显著缩小了文本列与 ID-only 的差距

---

## 4. UniSRec 已完成结果

### 4.1 Beauty：external text baseline + portability

| 模型族 | Config | MRR@10 | HR@10 | NDCG@10 | 角色 |
|--------|--------|--------|-------|---------|------|
| SASRec | ID-only | 2.16% | 5.31% | 2.91% | 内部 ablation |
| SASRec | MV-Align (7B) | 3.18% | 5.79% | 3.79% | 主方法 |
| UniSRec | Base | 2.47% | 6.60% | 3.44% | 外部 text baseline |
| UniSRec | AlignV3 | 3.25% | 6.52% | 4.02% | portability |
| UniSRec | AlignMultiViewV3 | 3.33% | 6.67% | 4.11% | portability |
| UniSRec | AlignV3 balanced | 3.27% | 6.57% | 4.04% | balanced sanity |
| UniSRec | AlignMultiViewV3 balanced | 3.25% | 6.53% | 4.02% | balanced sanity |

当前可用结论：

- UniSRec Base 作为外部文本基线是成立的
- Align / MultiView 插件在 UniSRec 上同样提升 `MRR/NDCG`
- 这组结果更适合写成 `external baseline + portability check`，不宜扩成第二主线

### 4.2 Toys：当前状态

- `UniSRec Base on Toys (seed=2024)` 已在 5090 启动并进入训练
- 因此 **Toys-UniSRec 结果尚未纳入本汇总**

---

## 5. GRU4Rec Phase 4 完成情况

### 5.1 完成矩阵

| 模块 | 完成度 |
|------|--------|
| 5090 text（LLM+MV） | 16/16 |
| baseline（ID+TF-IDF） | 16/16 |
| 合计 checkpoint 块 | 32/32 |

### 5.2 Seed 完整性

| 配置族 | 42 | 2024 | 2025 | 2026 |
|--------|----|------|------|------|
| Beauty LLM | ✅ | ✅ | ✅ | ✅ |
| Beauty MV | ✅ | ✅ | ✅ | ✅ |
| Toys LLM | ✅ | ✅ | ✅ | ✅ |
| Toys MV | ✅ | ✅ | ✅ | ✅ |
| Beauty ID | ✅ | ✅ | ✅ | ✅ |
| Beauty TF-IDF | ✅ | ✅ | ✅ | ✅ |
| Toys ID | ✅ | ✅ | ✅ | ✅ |
| Toys TF-IDF | ✅ | ✅ | ✅ | ✅ |

说明：

- 这部分当前已经是**训练完成**状态
- 已整理出的快照见：`gru4rec_results_snapshot_20260629.md`
- 但 GRU4Rec 的完整四配置正式论文表格还没有像 SASRec / Grocery 那样全部整理完
- 现阶段最可信的完成依据是 `gru4rec_phase4_matrix_20260629.md`

---

## 6. 补充指标与机制分析

### 6.1 paired t-test（Beauty / Toys）

对比口径：`MV-Align(7B)` vs `TF-IDF+LLM`

| 数据集 | Hit@10 | NDCG@10 | MRR@10 |
|--------|--------|----------|--------|
| Beauty | `p=1.56e-05` (***) | `p=7.30e-04` (***) | `p=3.22e-02` (*) |
| Toys | `p=2.23e-07` (***) | `p=3.20e-04` (***) | `p=4.95e-02` (*) |

这和当前主文脚注口径一致：`HR/NDCG` 更强，`MRR` 保守标 `*`。

### 6.2 Coverage_new@10（Beauty / Toys，4-seed mean）

| 数据集 | ID-only | TF-IDF | TF-IDF+LLM | MV-Align |
|--------|---------|--------|------------|----------|
| Beauty | 15.85 | 15.11 | 14.99 | 14.35 |
| Toys | 19.51 | 22.00 | 22.49 | 22.30 |

当前可读结论：

- Beauty 上 `Coverage_new@10` 不是 MV 的优势项
- Toys 上文本列整体优于 ID-only，MV 与 TF-IDF+LLM 接近

### 6.3 Gate bucket（Beauty）

当前输出：

- global gate: `view_0=0.4054`, `view_1=0.3976`, `view_2=0.3859`, `view_3=0.4122`
- bucket 文件已生成：`mechanism_gate_buckets_beauty.txt`

备注：

- 这项已经有结果，但还更像“机制补充材料”
- 如果要进主文，建议后续把解释和可视化再打磨一下

### 6.4 Cross concentration

已完成并有产物：

- `ablation_study_doc/scores/beauty_id_only_seed2024_mechanism_phase_b_topk_scores.npy`
- `ablation_study_doc/scores/beauty_tfidf_v3_seed2024_mechanism_phase_b_topk_scores.npy`
- `ablation_study_doc/scores/beauty_mv_v3_seed2024_mechanism_phase_b_topk_scores.npy`
- 快照分析：`mechanism_concentration_beauty_20260629.md`
- 图：`ablation_study_doc/figures/beauty_mechanism_concentration_rank_curve_20260629.png`

可直接支持 `0617zhidao` 提到的 score concentration / popularity allocation 叙事。

### 6.5 Case study 候选（Beauty）

- 候选清单：`mechanism_case_study_candidates_beauty_20260629.md`
- 当前已经能筛出：
  - `MV rank<=10` 且 `ID/TF-IDF` 都未进前 10 的用户
  - `MV` 相对 `TF-IDF` 提前至少 20 个 rank 且进前 20 的用户
- 但要做成最终论文案例图，还需要把 item index 反查成 title / metadata

---

## 7. 当前仍未收口的项

| 项目 | 状态 |
|------|------|
| case study（new / tail item Top-K） | ⚠️ 还没有整理成正式材料 |
| GRU4Rec 正式结果表 | ✅ 已补齐四配置结果，见 `gru4rec_results_snapshot_20260629.md` |
| UniSRec Toys Base（seed=2024） | ✅ 已完成，checkpoint 与日志已同步到本地 |

---

## 8. 建议把这份总览怎么用

- 主文主线看 `第 2–4 节`
- 补充指标 / appendix 看 `第 6 节`
- 训练完成度与 checkpoint 完整性看 `第 5 节`
- 后续待补任务直接看 `第 7 节`
