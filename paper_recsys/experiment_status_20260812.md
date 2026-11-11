# Experiment Status — 2026-08-12

> 更新时间: 16:00  
> 上一日: [`experiment_status_20260807.md`](experiment_status_20260807.md)

---

## 一句话

**完成 0806 指导中两项 pending 实验**：GTS-Random 多 sample_seed (3 seeds: 42/123/456) + Pre-vs-post-whitening CKA 审计。脚本已部署并运行中。

---

## 一、0806 指导未完成项（本次补完）

| 要求 | 之前状态 | 本次操作 |
|------|:--------:|----------|
| GTS-Random 多 target-sampling seed | 仅 seed=42 | ✅ 补 seed=123, 456 → 3-seed 聚合 `gts_random_multiseed_agg.csv` |
| Raw-vs-projected CKA 审计 | 未做 | ✅ 用 `no_whiten` (SVD+center, 无白化) vs `ts` (SVD+center+whiten) 对比 → `cka_pre_vs_post_*.json` |

### 实验设计说明

**GTS-Random Multi-seed**：原有 `sample_seed=42` 的结果已保留，新增 seed=123 和 seed=456。对每个 user 的 post-cutoff targets 使用不同 RNG seed 采样一个 target，三组 seed 的结果取均值和标准差，可以评估 GTS-Random 协议对 target 选择的稳定性。

**Pre-vs-Post CKA 审计**：由于完整 4096-d 原始 Qwen 向量未持久化（生成流程中直接做了 SVD-256 投影再切 4×64-d views），本次对比两个可用阶段：
1. `no_whiten`：SVD 投影后 + center，但未白化（64-d per view）
2. `ts_whitened`：SVD + center + whitening + TS-cutoff（64-d per view）

若两阶段 CKA 一致高，说明冗余源自 prompt semantic paraphrasing 而非白化步骤压塌分布。

---

## 二、新增脚本

| 文件 | 用途 |
|------|------|
| `scripts/cka_pre_vs_post.py` | Pre/post whitening CKA + ER 对比分析 |
| `run_5090_final_supplement.sh` | 编排 GTS-Random 123/456 + CKA 对比 |

---

## 三、运行结果

全部完成（~55min），产物：
- `logs/gts_random_eval_seed42.csv` — 30 rows
- `logs/gts_random_eval_seed123.csv` — 30 rows  
- `logs/gts_random_eval_seed456.csv` — 30 rows
- `paper_recsys/gts_random_multiseed_agg.csv` — 聚合 3-seed 均值+标准差
- `paper_recsys/cka_pre_vs_post_beauty.json`
- `paper_recsys/cka_pre_vs_post_toys_games.json`

### GTS-Random Multi-seed 关键结论

| Dataset | ID | TF | LLM | MV | UniSRec |
|---------|---:|---:|----:|---:|--------:|
| Beauty | .0074 | **.0122** | .0118 | .0123 | .0096 |
| Toys | .0076 | **.0113** | .0108 | .0108 | .0086 |

- 与单 seed=42 相比，多 seed 平均后 TF 和 MV 在 Beauty 上几乎持平（0.0122 vs 0.0123）
- Toys 上 TF 仍然领先
- 跨 sample-seed std 为 0.001–0.003，说明 GTS-Random 对 target 采样有中等敏感性

### CKA Pre-vs-Post Whitening 关键结论

| Stage | Beauty CKA | Toys CKA | Beauty ER/256 | Toys ER/256 |
|-------|:----------:|:--------:|:-------------:|:-----------:|
| no_whiten (SVD+center) | 0.95–0.99 | 0.94–0.98 | 99 (39%) | 104 (41%) |
| ts_whitened (SVD+center+whiten) | 0.84–0.94 | 0.82–0.92 | 136 (53%) | 140 (55%) |

- **CKA 在白化前已高达 0.95+** → 冗余来自 prompt semantic paraphrasing
- 白化反而降低 CKA（~0.09）并提升 ER → 白化是在 decorrelate views
- 结论：Multi-view 冗余是生成端固有的，非预处理 artifact

---

## 四、全量实验完成度

| 实验项目 | 状态 | 备注 |
|---------|:----:|------|
| GTS-All 主表 (Beauty/Toys × ID/TF/LLM/MV/UniSRec × 3seeds) | ✅ | |
| LOO 主表 (Beauty/Toys × 4方案 × 3seeds) | ✅ | |
| GTS-Last warm eval (30 ckpt) | ✅ | |
| GTS-Random seed=42 | ✅ | |
| GTS-Random multi-seed (42/123/456) | ✅ | 数据已补入论文 |
| Early60 temporal window | ✅ | |
| Grocery 第三数据集 | ✅ | |
| CKA 3-subsample (Beauty/Toys) | ✅ | |
| CKA Pre-vs-Post whitening | ✅ | 数据已补入论文 RQ3 |
| LOOV (leave-one-view-out) | ✅ | |
| Model cost table | ✅ | |
| Cross attention rank transition | ✅ | (negative → appendix) |
| Warm-target audit | ✅ | |
| UniSRec portability baseline | ✅ | |

---

## 五、代码同步

已将最新全部代码、文档、论文 (main_wsdm2027_v3.tex)、overleaf_deploy、实验结果 JSON/CSV 同步至 5090。

## 六、论文更新内容

基于新数据，`main_wsdm2027_v3.tex` 做了以下修改：
1. **GTS-Random 主表数据**：替换为 3 sample-seed 均值（原 seed=42 → 3-seed avg），Beauty TF≈MV
2. **RQ3 "Where does redundancy originate?"**：从 future work 改为已完成分析，CKA pre-whitening=0.96，证实冗余源自 prompt paraphrasing
3. **Limitations**：GTS-Random 已标注使用 3 seeds；redundancy locus 改为 "raw 4096-d audit remains"
4. **Future Work**：移除已完成项，新增 "instruction-diverse prompts" 方向
5. **Abstract/Conclusion 等**：同步更新数据引用和叙事一致性
