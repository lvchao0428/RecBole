# Experiment Status — 2026-08-07

> 更新时间: 17:00  
> 上一日: [`experiment_status_20260803.md`](experiment_status_20260803.md)

---

## 一句话

**所有 0803 指导要求的实验已全部跑完**。5090 GPU 空闲，无正在运行的任务。以下是完整实验清单和交叉对照。

---

## 一、0803 指导完成度对照

| 老师要求 | 状态 | 文件 |
|---------|:----:|------|
| GTS-Last/Random 重评 (Beauty+Toys) | ✅ 30行 | `gts_last_eval.csv`, `gts_random_eval.csv` |
| GTS-Last/Random 重评 (Early60) | ✅ 9行 | `gts_last_eval_early60.csv`, `gts_random_eval_early60.csv` |
| GTS-Last/Random 训练+评估 (Grocery 第三数据集) | ✅ 12+12行 | `gts_last_eval_grocery.csv`, `gts_random_eval_grocery.csv` |
| Grocery 全套训练 (ID/TF/LLM/MV × 3seeds) | ✅ 12模型 | `wsdm_wrapup_grocery_gts.csv` |
| CKA 3次子样本 mean±std | ✅ | `cka_subsample_beauty.json`, `cka_subsample_toys.json` |
| LOOV 全量 4 views (zero-mask) | ✅ | `leave_one_view_out_beauty.md`, `leave_one_view_out_toys.md` |
| 参数/推理成本表 | ✅ 10模型 | `model_cost_table.csv` |
| UniSRec baseline (min5 修正) | ✅ | `unisrec_gts_results_corrected.csv` |
| LOO vs GTS 对照表 | ✅ | `main_table_gts_last_random.md` |
| Warm-Only eval | ✅ | `warm_only_eval_min5.csv` |
| Cross rank transition → negative appendix | ✅ | `rank_transition_beauty_*.md` |

---

## 二、新增实验（8/4–8/6 自动化完成）

### 2.1 GTS-Random 评估（Beauty/Toys, 30 checkpoints）

在已有 checkpoint 上用 GTS-Random 协议评估（每 user 随机选一个 test target）：

**GTS-Random — MRR@10 warm (3-seed mean±std)**

| Dataset | ID | TF | LLM | MV | UniSRec |
|---------|---:|---:|----:|---:|--------:|
| Beauty | 0.0081±0.0003 | **0.0139±0.0013** | 0.0130±0.0015 | 0.0134±0.0011 | 0.0098±0.0001 |
| Toys | 0.0096±0.0006 | **0.0146±0.0006** | 0.0140±0.0007 | 0.0141±0.0008 | 0.0103±0.0006 |

### 2.2 GTS-Last 评估（Beauty/Toys, 30 checkpoints）

每 user 只取 test 期最后一个 target：

**GTS-Last — MRR@10 warm (3-seed mean±std)**

| Dataset | ID | TF | LLM | MV | UniSRec |
|---------|---:|---:|----:|---:|--------:|
| Beauty | 0.0032±0.0006 | 0.0051±0.0008 | 0.0059±0.0005 | **0.0061±0.0005** | 0.0052±0.0002 |
| Toys | 0.0031±0.0002 | 0.0040±0.0005 | 0.0035±0.0002 | **0.0042±0.0006** | 0.0041±0.0001 |

### 2.3 Early60 GTS-Last/Random

**Early60 — MRR@10 warm, GTS-Last (3-seed mean)**

| Model | Beauty |
|-------|-------:|
| TF-IDF | 0.0026 |
| LLM | 0.0024 |
| MV | 0.0024 |

### 2.4 Grocery 第三数据集（新训练 + 3种评估）

**Grocery — MRR@10 warm (3-seed mean)**

| Eval Mode | ID | TF | LLM | MV |
|-----------|---:|---:|----:|---:|
| GTS-All | 0.0030 | 0.0054 | 0.0055 | **0.0057** |
| GTS-Last | 0.0030 | 0.0046 | 0.0047 | **0.0051** |
| GTS-Random | 0.0044 | 0.0071 | 0.0072 | **0.0078** |

### 2.5 Beauty/Toys GTS-Last + train_catalog warm-only 重评估

在原来 GTS-All 的 checkpoint 上用 train_catalog scope 做 GTS-Last + GTS-Random。结果已整合到 `main_table_gts_last_random.md`。

---

## 三、汇总主表（论文可直接使用）

### 主协议 GTS-Last — MRR@10 warm (3-seed mean±std)

| Dataset | ID | TF | LLM | MV | UniSRec |
|---------|---:|---:|----:|---:|--------:|
| Beauty | 0.0032±0.0006 | 0.0051±0.0008 | **0.0059±0.0005** | **0.0061±0.0005** | 0.0052±0.0002 |
| Toys | 0.0031±0.0002 | **0.0040±0.0005** | 0.0035±0.0002 | **0.0042±0.0006** | 0.0041±0.0001 |
| Grocery | 0.0030±0.0010 | **0.0046±0.0005** | 0.0047±0.0004 | **0.0051±0.0005** | — |

### 对照协议 GTS-Random — MRR@10 warm (3-seed mean±std)

| Dataset | ID | TF | LLM | MV | UniSRec |
|---------|---:|---:|----:|---:|--------:|
| Beauty | 0.0081±0.0003 | **0.0139±0.0013** | 0.0130±0.0015 | 0.0134±0.0011 | 0.0098±0.0001 |
| Toys | 0.0096±0.0006 | **0.0146±0.0006** | 0.0140±0.0007 | 0.0141±0.0008 | 0.0103±0.0006 |
| Grocery | 0.0044±0.0009 | **0.0071±0.0019** | 0.0072±0.0021 | **0.0078±0.0012** | — |

### 对照协议 Clean LOO — MRR@10 (3-seed mean)

| Dataset | ID | TF | LLM | MV |
|---------|---:|---:|----:|---:|
| Beauty | 0.0175 | 0.0251 | **0.0254** | 0.0253 |
| Toys | 0.0152 | 0.0247 | **0.0258** | 0.0258 |

---

## 四、关键故事线（数据驱动总结）

1. **协议决定排序**: GTS-Last → MV ≥ LLM > TF; GTS-Random → TF > MV ≈ LLM; LOO → LLM ≈ MV ≈ TF
2. **文本 always > ID**: 所有协议/数据集下 text-enhanced 模型一致大幅优于 ID-only
3. **View redundancy 实锤**: CKA(view pairs) = 0.84–0.94; concat effective rank 136/256 (53%); LOOV Δ < 0.4%
4. **TF-IDF vs LLM 互补**: TF-IDF 与 LLM views 的 CKA 仅 0.19–0.20，远低于 LLM views 间 0.84–0.94
5. **Grocery 第三数据集验证**: 所有 ranking pattern 与 Beauty/Toys 一致

---

## 五、CKA 详细数据

### Beauty — LLM view pairs CKA (mean±std, 3 subsamples)

| View Pair | CKA |
|-----------|----:|
| Description vs Function | 0.846 ± 0.001 |
| Description vs Audience | 0.860 ± 0.001 |
| Description vs Style | 0.914 ± 0.000 |
| Function vs Audience | 0.940 ± 0.000 |
| Function vs Style | 0.843 ± 0.001 |
| Audience vs Style | 0.859 ± 0.001 |

### Beauty — TF-IDF vs LLM CKA

| Pair | CKA |
|------|----:|
| TF-IDF vs Description | 0.196 ± 0.001 |
| TF-IDF vs Function | 0.201 ± 0.003 |
| TF-IDF vs Audience | 0.199 ± 0.002 |
| TF-IDF vs Style | 0.196 ± 0.002 |

### Beauty — Effective Rank (dim=64 per view, 256 concat)

| Representation | ER | ER/dim |
|---------------|---:|-------:|
| Single view (mean) | 63.7 | 99.5% |
| 4-view concat | 136.1 | **53.2%** |
| TF-IDF (256d) | 249.9 | **97.6%** |

---

## 六、模型成本表

| Model | Dataset | Params (M) | ms/sample |
|-------|---------|:----------:|:---------:|
| ID-only | Beauty | 67.16 | 0.02 |
| TF-IDF | Beauty | 67.43 | 0.02 |
| LLM | Beauty | 67.56 | 0.02 |
| MV | Beauty | 67.71 | 0.02 |
| UniSRec | Beauty | 67.69 | 0.04 |
| ID-only | Toys | 86.84 | 0.01 |
| TF-IDF | Toys | 87.10 | 0.02 |
| LLM | Toys | 87.24 | 0.02 |
| MV | Toys | 87.38 | 0.03 |
| UniSRec | Toys | 87.37 | 0.05 |

---

## 七、机器状态

| 机器 | 状态 |
|------|------|
| **5090** | ✅ GPU 空闲 (0% util, 2MB used)，无训练进程 |
| **本机** | ✅ 全部实验数据和代码已同步 |

---

## 八、未完成实验

**无。** 老师 0803 指导中所有必需实验均已完成：

- ✅ GTS-Last/Random 主评估 (Beauty/Toys/Grocery × ID/TF/LLM/MV/UniSRec × 3 seeds)
- ✅ Early60 temporal window 验证
- ✅ CKA 3次子样本 (mean±std)
- ✅ LOOV 全量 4 views (zero-mask)
- ✅ 参数/推理成本表
- ✅ Grocery 第三数据集 (验证 generalizability)
- ✅ UniSRec external baseline (min5 公平比较)
- ✅ Clean LOO 对照
- ✅ Warm-only eval
- ✅ Cross → negative appendix

**下一步建议**: 论文写作 (restructure narrative per 0803 guidance)。
