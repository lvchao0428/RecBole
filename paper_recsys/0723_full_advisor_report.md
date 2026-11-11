# 0723 指导：完整数据汇报与论文就绪度审计

> 指导原文：`paper_recsys/0723zhidao.txt`  
> 数据冻结时间：2026-08-02 16:05  
> 主训练口径：`min_train_interactions=5`、no-boost、no-Cross、per-source align、3 seeds  
> 主指标：MRR@10；文中 `±` 统一采用 **sample standard deviation (`ddof=1`)**

---

## 一、先给结论

### 1.1 老师提出的三个论文问题，目前能回答到什么程度

| 核心问题 | 当前答案 | 证据强度 | 论文就绪度 |
|---|---|:---:|:---:|
| Q1. Clean LOO 与 leakage-free GTS 是否改变结论？ | **会。** GTS 下 TF 最优或并列最优；LOO 抬高指标并把排序改成 LLM/MV≈TF 或略高 | 强 | **可写**（all-target/min5/MRR） |
| Q2. TF、single-view LLM、prompt MV 谁在 temporal shift 下更鲁棒？ | **TF 最稳健；LLM/MV 没有稳定超过 TF。** Beauty 两窗口同向，Toys Main 复现 | 中强 | **基本可写**；Toys Early 为可选加强 |
| Q3. MV 没增益是容量不足还是 view redundancy？ | **更符合 view redundancy。** CKA 高、concat effective rank 仅 53–55%、LOVO 影响小 | 强（表征层）/中强（行为层） | **基本可写**；LOVO 应补 min5 同口径复评 |

### 1.2 关键论文论点是否数据完备

| 论文论点 | 判定 | 说明 |
|---|:---:|---|
| 文本模型明显高于 ID-only | **已支持** | Beauty/Toys、GTS/LOO 均一致；尚未声称统计显著性 |
| GTS 会改变绝对指标与模型排序 | **已支持** | 两数据集 Clean LOO vs GTS 3-seed |
| 复杂 LLM/MV 没有稳定超过 TF-IDF | **已支持** | Beauty Main/Early + Toys Main；差距小且无稳定层级 |
| MV 受 view redundancy 限制 | **已支持** | 两数据集 CKA、effective rank；LOVO 方向一致 |
| Temporal shift 是三路共同掉点主因 | **已支持（相关性证据）** | 三路 valid→test 同步下降；popularity/unseen/catalog 同时漂移 |
| Warm-target-only 最终主表 | **尚未 paper-ready** | 当前离线重评没有复现 min5，且没有把候选集限制为 train warm catalog |
| UniSRec 与主方法公平对比 | **尚未 paper-ready** | 6 个 test 已算出，但离线加载未复现 min5，不能直接和主训练日志表比较 |
| Cross 是稳定增益机制 | **不成立** | improved/degraded 近各半；只适合 Appendix negative analysis |

**总判断：核心故事已经成立，但最终论文主表还不能冻结。**  
不需要重新训练；需要用一个统一 evaluator 对现有 checkpoint 做一次同口径复评。

---

## 二、最重要的数据口径审计

### 2.1 当前存在四种统计总体

| 数据产物 | 用户/样本口径 | Beauty test n | Toys test n | 可用于什么 |
|---|---|---:|---:|---|
| 训练日志主结果（`two_phase_train.py --min_train_interactions 5`） | 所有 train 历史≥5 的 test 交互 | **12,001** | **13,186** | 当前 Clean LOO/GTS 主结论 |
| `warm_only_eval.csv`、当前 LOVO、当前 UniSRec test | checkpoint 重载后的未过滤 test 交互 | **79,724** | **90,069** | 诊断性分析；不可直接和 min5 主表混排 |
| `warm_target_audit_*.json` | train 历史≥5 用户在窗口中的最后一个 target | 6,332 | 6,665 | target-level unseen audit |
| `distribution_shift_*_formal.json` | 全局时间切分中的全部 test 交互 | 202,307 | 225,278 | 数据分布漂移描述 |

### 2.2 同一个 checkpoint 在两种人群上的差异

| Checkpoint | 训练日志：min5 test MRR | 离线重载：未过滤 test MRR | 训练 n | 离线 n |
|---|---:|---:|---:|---:|
| Beauty TF seed42 | 0.0160 | 0.020239 | 12,001 | 79,724 |
| Beauty MV seed42 | 0.0159 | 0.020657 | 12,001 | 79,724 |
| Toys TF seed42 | 0.0116 | 0.025321 | 13,186 | 90,069 |
| Toys MV seed42 | 0.0102 | 0.024600 | 13,186 | 90,069 |

差异不是 checkpoint 损坏，也不是模型发生变化，而是 `load_data_and_model()` 没有复现
`two_phase_train.py` 中自定义的 `min_train_interactions=5` 过滤。

### 2.3 当前 warm-only 结果还差哪一步

`scripts/eval_warm_targets_only.py` 当前做了：

1. 保留 warm target 行；
2. 同时报 all-target 与 warm-target 指标。

但它尚未做：

1. 按训练主协议过滤 `train history ≥ 5`；
2. 把排序候选集限制为 `train warm catalog`；
3. 纳入 Beauty ID-only 三个 checkpoint。

因此，现有 warm-only 表能说明“过滤 target 后排序大致不变”，但不能作为最终论文主表。

**2026-08-02 16:15 修复状态**：`eval_warm_targets_only.py`、
`eval_ckpt_test.py` 与 `analyze_leave_one_view_out.py` 已默认重建
`min_train_interactions=5` dataloader；warm-only 默认使用 train-catalog
候选集。旧 CSV/MD 仍是修复前结果，必须重跑后才能替换本报告中的诊断表。
本地语法、lint 与 candidate-mask 单元 smoke test 已通过；5090 当前有约 15 GB
GPU 任务占用，因此未抢占 GPU 启动远端 checkpoint sanity run。

---

# 第一组实验：Audit unseen target 与最终评测口径

## 3.1 GTS cutoff 与 last-target audit

| Dataset | Train cutoff | Valid cutoff | Train catalog | Eligible users | Valid unseen | Test unseen | Catalog turnover |
|---|---|---|---:|---:|---:|---:|---:|
| Beauty | 2014-03-09 | 2014-05-17 | 217,470 | 39,530 | 1,193/7,845 = 15.2% | **1,762/6,332 = 27.8%** | 23.3% |
| Toys | 2014-01-17 | 2014-04-08 | 283,233 | 44,024 | 1,543/7,307 = 21.1% | **2,300/6,665 = 34.5%** | 31.1% |

## 3.2 老师提到的 17.8% 来自哪个口径

| Dataset | Split | Head | Mid | Low | Unseen |
|---|---|---:|---:|---:|---:|
| Beauty | Valid 全交互 | 63.1% | 15.9% | 9.7% | 11.3% |
| Beauty | Test 全交互 | 56.6% | 14.9% | 10.7% | **17.8%** |
| Toys | Valid 全交互 | 60.1% | 16.3% | 10.6% | 13.0% |
| Toys | Test 全交互 | 52.3% | 15.5% | 11.0% | **21.3%** |

`17.8%` 和 `27.8%` 不冲突：

- 17.8%：Beauty 全 test 交互 bucket；
- 27.8%：Beauty eligible 用户的 last target；
- 27.25%：当前未过滤离线 evaluator 的全部 test 行。

论文必须固定一种主口径，不能把三者混写成同一统计量。

## 3.3 Popularity 与 catalog 漂移

| Dataset | 指标 | Valid | Test | 变化 |
|---|---|---:|---:|---:|
| Beauty | Target popularity mean | 112.84 | 101.79 | −9.8% |
| Beauty | Target popularity median | 21 | 15 | −28.6% |
| Beauty | New-item ratio | 23.5% | 27.3% | +3.8pp |
| Beauty | Zero-train-count targets | 22,952 | 35,965 | +56.7% |
| Toys | Target popularity mean | 144.66 | 119.12 | −17.7% |
| Toys | Target popularity median | 18 | 11 | −38.9% |
| Toys | New-item ratio | 27.1% | 32.8% | +5.7pp |
| Toys | Zero-train-count targets | 29,332 | 47,945 | +63.5% |

## 3.4 当前 warm-only 离线重评：全量 per-seed 数据

> **诊断表，非最终论文表。** 该表是未复现 min5 的 79,724/90,069 test 样本，
> 候选集仍是全 item catalog。

### Beauty（n_all=79,724，n_warm=58,001，unseen=27.25%）

| Model | Seed | MRR all | MRR warm | NDCG all | NDCG warm | HR all | HR warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF | 42 | 0.020239 | 0.027818 | 0.024534 | 0.033723 | 0.038608 | 0.053068 |
| TF | 2024 | 0.020262 | 0.027851 | 0.024667 | 0.033905 | 0.039022 | 0.053637 |
| TF | 2026 | 0.021018 | 0.028890 | 0.025484 | 0.035029 | 0.040051 | 0.055051 |
| LLM | 42 | 0.019719 | 0.027104 | 0.023955 | 0.032927 | 0.037831 | 0.051999 |
| LLM | 2024 | 0.020431 | 0.028084 | 0.024781 | 0.034062 | 0.038959 | 0.053551 |
| LLM | 2026 | 0.020284 | 0.027880 | 0.024432 | 0.033582 | 0.037968 | 0.052189 |
| MV | 42 | 0.020657 | 0.028393 | 0.025156 | 0.034578 | 0.039863 | 0.054792 |
| MV | 2024 | 0.020087 | 0.027610 | 0.024404 | 0.033544 | 0.038483 | 0.052896 |
| MV | 2026 | 0.020396 | 0.028035 | 0.024862 | 0.034174 | 0.039436 | 0.054206 |

### Toys（n_all=90,069，n_warm=61,056，unseen=32.21%）

| Model | Seed | MRR all | MRR warm | NDCG all | NDCG warm | HR all | HR warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| ID | 42 | 0.017277 | 0.025471 | 0.023076 | 0.034013 | 0.041524 | 0.061190 |
| ID | 2024 | 0.017318 | 0.025532 | 0.023296 | 0.034343 | 0.042279 | 0.062320 |
| ID | 2026 | 0.017961 | 0.026491 | 0.022958 | 0.033860 | 0.038937 | 0.057423 |
| TF | 42 | 0.025321 | 0.037354 | 0.029688 | 0.043795 | 0.043855 | 0.064695 |
| TF | 2024 | 0.025007 | 0.036891 | 0.029411 | 0.043386 | 0.043600 | 0.064318 |
| TF | 2026 | 0.025002 | 0.036882 | 0.029264 | 0.043170 | 0.043067 | 0.063532 |
| LLM | 42 | 0.024243 | 0.035764 | 0.028355 | 0.041829 | 0.041668 | 0.061468 |
| LLM | 2024 | 0.024528 | 0.036183 | 0.028587 | 0.042172 | 0.041668 | 0.061468 |
| LLM | 2026 | 0.024585 | 0.036268 | 0.028464 | 0.041990 | 0.040957 | 0.060420 |
| MV | 42 | 0.024600 | 0.036289 | 0.028841 | 0.042546 | 0.042523 | 0.062729 |
| MV | 2024 | 0.025314 | 0.037343 | 0.029737 | 0.043868 | 0.044055 | 0.064990 |
| MV | 2026 | 0.025143 | 0.037091 | 0.029737 | 0.043867 | 0.044588 | 0.065776 |

### 3.5 当前诊断表的 3-seed 聚合

| Dataset | Model | MRR all | MRR warm | NDCG all | NDCG warm | HR all | HR warm |
|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | TF | 0.020506±0.000443 | **0.028186±0.000610** | 0.024895±0.000514 | **0.034219±0.000707** | 0.039227±0.000743 | **0.053919±0.001021** |
| Beauty | LLM | 0.020145±0.000376 | 0.027689±0.000517 | 0.024389±0.000415 | 0.033524±0.000570 | 0.038253±0.000616 | 0.052580±0.000847 |
| Beauty | MV | 0.020380±0.000285 | 0.028013±0.000392 | 0.024807±0.000379 | 0.034099±0.000521 | **0.039261±0.000707** | **0.053965±0.000971** |
| Toys | ID | 0.017519±0.000384 | 0.025831±0.000572 | 0.023110±0.000172 | 0.034072±0.000247 | 0.040913±0.001753 | 0.060311±0.002564 |
| Toys | TF | **0.025110±0.000183** | **0.037042±0.000270** | **0.029454±0.000215** | **0.043450±0.000317** | 0.043507±0.000402 | 0.064182±0.000593 |
| Toys | LLM | 0.024452±0.000183 | 0.036072±0.000270 | 0.028469±0.000116 | 0.041997±0.000172 | 0.041431±0.000410 | 0.061119±0.000605 |
| Toys | MV | 0.025019±0.000373 | 0.036908±0.000550 | 0.029438±0.000517 | 0.043427±0.000763 | **0.043722±0.001072** | **0.064498±0.001582** |

### 第一组结论

1. Unseen target 很多，老师要求先固定 protocol 是正确的；
2. 当前诊断结果显示 warm-only 抬高绝对值，但 TF/LLM/MV 排序基本不变；
3. **最终 warm-only 主表仍缺统一 min5 + warm candidate mask + Beauty ID。**

---

# 第二组实验：Clean LOO vs GTS、双窗口与 UniSRec

## 4.1 统一实验设置

| 项目 | 统一设置 |
|---|---|
| 模型 | ID-only、TF-IDF、single-view LLM、prompt MV |
| Text align | per-source align |
| Boost / Cross | cold boost=0、infer boost=0、no-Cross |
| Filter | min_train_interactions=5 |
| Seeds | 42、2024、2026（Early60 使用 2025、42、2024） |
| 选参 | Shared valid grid；不为 MV 单独扩网格 |
| 固定超参 | ID/LLM: lr=1e-4, dropout=0.1；TF/MV: lr=5e-4, dropout=0.3 |

## 4.2 Beauty GTS Main：全量 per-seed MRR@10

| Model | Seed42 | Seed2024 | Seed2026 | Mean±sample std |
|---|---:|---:|---:|---:|
| ID-only | 0.0112 | 0.0104 | 0.0103 | 0.0106±0.0005 |
| TF | 0.0160 | 0.0161 | 0.0174 | **0.0165±0.0008** |
| LLM | 0.0156 | 0.0161 | 0.0165 | 0.0161±0.0005 |
| MV | 0.0159 | 0.0153 | 0.0169 | 0.0160±0.0008 |

Text 相对 ID-only 的 mean 提升：

- TF：+55.2%
- LLM：+51.1%
- MV：+50.8%

## 4.3 Beauty Clean LOO：全量 per-seed MRR@10

| Model | Seed42 | Seed2024 | Seed2026 | Mean±sample std |
|---|---:|---:|---:|---:|
| ID-only | 0.0179 | 0.0171 | 0.0174 | 0.0175±0.0004 |
| TF | 0.0250 | 0.0253 | 0.0249 | 0.0251±0.0002 |
| LLM | 0.0251 | 0.0252 | 0.0258 | **0.0254±0.0004** |
| MV | 0.0258 | 0.0251 | 0.0250 | 0.0253±0.0004 |

## 4.4 Toys GTS Main：全量 per-seed MRR@10

| Model | Seed42 | Seed2024 | Seed2026 | Mean±sample std |
|---|---:|---:|---:|---:|
| ID-only | 0.0070 | 0.0070 | 0.0069 | 0.0070±0.0001 |
| TF | 0.0116 | 0.0112 | 0.0110 | **0.0113±0.0003** |
| LLM | 0.0111 | 0.0105 | 0.0104 | 0.0107±0.0004 |
| MV | 0.0102 | 0.0112 | 0.0113 | 0.0109±0.0006 |

Text 相对 ID-only 的 mean 提升：

- TF：+61.7%
- LLM：+53.1%
- MV：+56.5%

## 4.5 Toys Clean LOO：全量 per-seed MRR@10

| Model | Seed42 | Seed2024 | Seed2026 | Mean±sample std |
|---|---:|---:|---:|---:|
| ID-only | 0.0150 | 0.0158 | 0.0148 | 0.0152±0.0005 |
| TF | 0.0248 | 0.0252 | 0.0240 | 0.0247±0.0006 |
| LLM | 0.0251 | 0.0250 | 0.0272 | **0.0258±0.0012** |
| MV | 0.0243 | 0.0240 | 0.0290 | **0.0258±0.0028** |

注意 Toys LOO 的 LLM/MV 方差明显增大，尤其 MV seed2026=0.0290；
所以不能把 0.0258 vs TF 0.0247 解读成稳定优势。

## 4.6 Clean LOO vs GTS 聚合

| Dataset | Protocol | ID | TF | LLM | MV | 结论 |
|---|---|---:|---:|---:|---:|---|
| Beauty | GTS | 0.0106 | **0.0165** | 0.0161 | 0.0160 | TF ≥ LLM≈MV |
| Beauty | LOO | 0.0175 | 0.0251 | **0.0254** | 0.0253 | LLM≈MV≈TF |
| Toys | GTS | 0.0070 | **0.0113** | 0.0107 | 0.0109 | TF > MV≈LLM |
| Toys | LOO | 0.0152 | 0.0247 | **0.0258** | **0.0258** | LLM≈MV≥TF，但方差大 |

LOO/GTS 的 mean 倍数：

| Dataset | ID | TF | LLM | MV |
|---|---:|---:|---:|---:|
| Beauty | 1.64× | 1.52× | 1.58× | 1.58× |
| Toys | 2.18× | 2.19× | 2.42× | 2.36× |

**核心结论：LOO 不只是把数值抬高，还改变“谁最好”的点估计叙述。**
这里的“排序改变”指 3-seed mean 的次序，不主张已证明统计显著的 rank reversal；
更稳妥的论文结论是：LOO 下看似存在的复杂文本优势，在 GTS 下消失。

## 4.7 Beauty Main / Early60：全量 per-seed 数据

| Model | Seed | Valid MRR | Test MRR | Valid NDCG | Test NDCG |
|---|---:|---:|---:|---:|---:|
| TF | 2025 | 0.0332 | 0.0052 | 0.0401 | 0.0063 |
| TF | 42 | 0.0341 | 0.0056 | 0.0412 | 0.0068 |
| TF | 2024 | 0.0335 | 0.0052 | 0.0403 | 0.0064 |
| LLM | 2025 | 0.0324 | 0.0051 | 0.0389 | 0.0061 |
| LLM | 42 | 0.0327 | 0.0050 | 0.0393 | 0.0061 |
| LLM | 2024 | 0.0332 | 0.0047 | 0.0396 | 0.0057 |
| MV | 2025 | 0.0333 | 0.0051 | 0.0402 | 0.0061 |
| MV | 42 | 0.0326 | 0.0051 | 0.0392 | 0.0063 |
| MV | 2024 | 0.0333 | 0.0053 | 0.0399 | 0.0064 |

| Window | TF test MRR | LLM test MRR | MV test MRR | 排序 |
|---|---:|---:|---:|---|
| Main | **0.0165±0.0008** | 0.0161±0.0005 | 0.0160±0.0008 | TF ≥ LLM≈MV |
| Early60 | **0.0053±0.0002** | 0.0049±0.0002 | 0.0052±0.0001 | TF ≥ MV ≥ LLM |

## 4.8 当前 UniSRec test：全量数据

> **已成功算出，但仍是未复现 min5 的离线口径，不能直接放入主对比表。**

| Dataset | Seed | Valid MRR | Test MRR | Test NDCG | Test HR |
|---|---:|---:|---:|---:|---:|
| Beauty | 42 | 0.0230 | 0.0161 | 0.0229 | 0.0449 |
| Beauty | 2024 | 0.0232 | 0.0167 | 0.0234 | 0.0453 |
| Beauty | 2026 | 0.0235 | 0.0167 | 0.0233 | 0.0448 |
| Toys | 42 | 0.0258 | 0.0194 | 0.0264 | 0.0487 |
| Toys | 2024 | 0.0255 | 0.0194 | 0.0267 | 0.0502 |
| Toys | 2026 | 0.0261 | 0.0196 | 0.0266 | 0.0490 |

| Dataset | Valid MRR mean±std | Test MRR mean±std |
|---|---:|---:|
| Beauty | 0.0232±0.0003 | 0.0165±0.0003 |
| Toys | 0.0258±0.0003 | 0.0195±0.0001 |

### 第二组结论

1. Q1 已由两数据集 3-seed 支持：LOO 会改变数值和排序；
2. Q2 已由 Beauty 两窗口 + Toys Main 支持：复杂文本无稳定优势；
3. UniSRec checkpoint 与 test 评测代码已可用，但必须补同一 min5/warm-candidate evaluator 后才能宣称公平 portability。

---

# 第三组实验：Mechanism evidence

## 5.1 Pairwise CKA：全量数据

CKA 越接近 1，表示两组 embedding 在样本间的几何关系越相似。

| Pair | Beauty CKA | Toys CKA |
|---|---:|---:|
| view0–view1 | 0.8432 | 0.8296 |
| view0–view2 | 0.8581 | 0.8522 |
| view0–view3 | 0.9123 | 0.9027 |
| view1–view2 | **0.9382** | **0.9195** |
| view1–view3 | 0.8393 | 0.8138 |
| view2–view3 | 0.8563 | 0.8360 |
| TF–view0 | 0.2495 | 0.3119 |
| TF–view1 | 0.2576 | 0.3342 |
| TF–view2 | 0.2537 | 0.3342 |
| TF–view3 | 0.2457 | 0.3062 |

结论：

- inter-view CKA：Beauty 0.839–0.938，Toys 0.814–0.919；
- TF–view CKA 仅 0.246–0.334；
- 四个 prompt view 互相像，TF-IDF 反而提供更不同的信息。

## 5.2 Effective rank：全量数据

| Representation | Beauty rank/dim | Beauty ratio | Toys rank/dim | Toys ratio |
|---|---:|---:|---:|---:|
| view0 | 63.75/64 | 99.61% | 63.77/64 | 99.64% |
| view1 | 63.73/64 | 99.58% | 63.77/64 | 99.65% |
| view2 | 63.75/64 | 99.61% | 63.77/64 | 99.64% |
| view3 | 63.79/64 | 99.67% | 63.79/64 | 99.66% |
| **4-view concat** | **136.94/256** | **53.49%** | **141.43/256** | **55.25%** |
| Drop view0 | 116.55/192 | 60.71% | 119.87/192 | 62.43% |
| Drop view1 | 116.21/192 | 60.53% | 118.57/192 | 61.76% |
| Drop view2 | 117.69/192 | 61.29% | 120.84/192 | 62.94% |
| Drop view3 | 115.30/192 | 60.05% | 118.72/192 | 61.84% |
| TF-IDF | 253.10/256 | 98.87% | 253.10/256 | 98.87% |
| TF + 4 views | 374.29/512 | 73.10% | 381.97/512 | 74.60% |

这排除了“每个 view 自身塌缩/容量不够”的简单解释：
每个 view 单独近满秩，但四个 view 拼接后只提供约 53–55% 的独立维度。

## 5.3 Leave-one-view-out：全量数据

> Zero-mask 推理，保持投影层维度不变。当前 evaluator 未复现 min5；
> 因而可作为机制证据，但最终 main-paper 数字建议同口径重跑。

| Dataset | Config | MRR@10 | Δ vs Full | Relative drop |
|---|---|---:|---:|---:|
| Beauty | Full | 0.020657 | — | — |
| Beauty | Drop description (v0) | 0.020344 | −0.000312 | 1.51% |
| Beauty | Drop function (v1) | 0.020577 | −0.000080 | 0.39% |
| Beauty | Drop audience (v2) | 0.020332 | −0.000325 | 1.57% |
| Beauty | Drop style (v3) | 0.020281 | −0.000376 | 1.82% |
| Toys | Full | 0.024600 | — | — |
| Toys | Drop description (v0) | 0.024193 | −0.000407 | 1.65% |
| Toys | Drop function (v1) | 0.024407 | −0.000192 | 0.78% |
| Toys | Drop audience (v2) | 0.024308 | −0.000292 | 1.19% |
| Toys | Drop style (v3) | 0.023847 | −0.000752 | 3.06% |

去掉单个 view 只下降 0.4–3.1%，支持“高冗余、低边际贡献”。
更准确的措辞应是 **LOVO effect is small**，而不是“严格等于 0”。

## 5.4 valid→test drop：主训练 min5 口径

| Dataset | Protocol | TF | LLM | MV |
|---|---|---:|---:|---:|
| Beauty | GTS | 43.1% | 44.3% | 43.9% |
| Beauty | LOO | 18.5% | 18.1% | 17.7% |
| Toys | GTS | 54.9% | 55.3% | 54.3% |
| Toys | LOO | 20.7% | 18.2% | 15.4% |

GTS 内三路 drop 非常接近，说明主要是共同的未来分布变难，
而不是 MV 或 LLM 单独过拟合。LOO 的 drop 不到 GTS 的一半，
进一步说明 LOO 掩盖 temporal shift。

## 5.5 Cross negative analysis：全量摘要

| Model | Improved | Degraded | Unchanged | Entropy Δ | Top1–Top10 margin Δ |
|---|---:|---:|---:|---:|---:|
| TF | 47.6% | 50.9% | 1.4% | +0.0082 | −0.0873 |
| LLM | 49.6% | 49.2% | 1.3% | +0.0014 | −0.0403 |
| MV | 51.5% | 47.3% | 1.2% | −0.0098 | +0.0653 |

不同模型的 entropy/margin 方向甚至不一致，不能构造稳定机制叙事。
Cross 应放 Appendix，作为 dataset/window-dependent 的负结果。当前 rank-transition
使用的是 `ps_beauty_*_seed2025` pilot checkpoint，而非 stop-gate 主表 checkpoint，
因此不应把其绝对指标与主表合并。

### 第三组结论

1. View redundancy 证据链完整：CKA 高 → concat rank 低 → LOVO 边际影响小；
2. Temporal shift 证据链完整：unseen/new-item 增加、popularity 降低、三路同步 drop；
3. Cross 没有一致机制，只保留负结果；
4. 行为层 LOVO 最终数字仍应补 min5 同口径复评。

---

## 六、论文关键论点与建议措辞

| 论点 | 可用措辞 | 不应过度声称 |
|---|---|---|
| Protocol effect | “Clean LOO and GTS yield materially different absolute scores and model rankings.” | 不说 GTS 在统计意义上必然优于所有 LOO |
| Text benefit | “All text-enhanced variants consistently outperform ID-only.” | 不说 LLM/MV 稳定优于 TF |
| Robustness | “TF-IDF is the most consistently competitive representation under temporal evaluation.” | 不说 TF 在所有 seed/dataset 都显著第一 |
| MV mechanism | “High inter-view CKA, low concatenated effective rank, and small LOVO effects indicate redundancy.” | 不把相关性写成唯一因果机制 |
| Temporal shift | “Shared valid-to-test degradation aligns with increased unseen items and reduced popularity.” | 不写成已证明的因果关系 |
| Cross | “Cross interactions show no stable direction across representations.” | 不硬解释 entropy/margin |

推荐标题：

> **When More Text Is Not Better: A Leakage-Aware Study of Text Representations for Sequential Recommendation**

推荐主贡献：

1. 揭示 LOO 与 leakage-free GTS 会改变 text-enhanced SR 的模型排序；
2. 在两 Amazon 域和多时间窗下，说明简单 TF-IDF 与复杂 LLM/MV 性能相当且更稳定；
3. 用 CKA、effective rank、LOVO 解释 multi-view 的边际收益受 view redundancy 限制；
4. 给出 warm-target/candidate-catalog 等 temporal evaluation 报告规范。

---

## 七、数据完备度矩阵

| 项目 | 训练完成 | 数据已提取 | 口径一致 | 可直接进论文 |
|---|:---:|:---:|:---:|:---:|
| Beauty GTS ID/TF/LLM/MV 3-seed MRR | ✅ | ✅ | ✅ min5 | ✅ |
| Toys GTS ID/TF/LLM/MV 3-seed MRR | ✅ | ✅ | ✅ min5 | ✅ |
| Beauty Clean LOO 3-seed MRR | ✅ | ✅ | ✅ min5 | ✅ |
| Toys Clean LOO 3-seed MRR | ✅ | ✅ | ✅ min5 | ✅ |
| Beauty Main/Early | ✅ | ✅ | ✅ min5 | ✅ |
| Unseen / popularity / turnover audit | — | ✅ | 多口径已标注 | ✅（需写清 denominator） |
| Warm-only 最终主表 | checkpoint ✅ | ✅ 诊断版 | ❌ 未复现 min5/candidate mask | **❌** |
| UniSRec portability | checkpoint ✅ | ✅ 诊断版 | ❌ 未复现 min5 | **❌** |
| CKA / effective rank | — | ✅ | 不依赖 eval cohort | ✅ |
| LOVO | checkpoint ✅ | ✅ | ❌ 未复现 min5 | 可作补充；主文前建议复评 |
| Cross rank transition | checkpoint ✅ | ✅ | 3,000-sample analysis | ✅ Appendix |
| 全模型 NDCG@10 / HR@10 主表 | 日志存在 | 部分 CSV | 尚未统一整理 | **❌** |
| Toys Early60 | 未跑 | — | — | 可选，不阻塞核心故事 |

### 总体判定

- **论点级证据：约 85–90% 完备。**
- **论文最终主表：约 65–70% 完备。**
- 缺口主要是评测器统一，不是重新训练。

---

## 八、最小收口任务（不需要重训）

### P0：冻结论文主表前必须完成

1. **Unified evaluator 已实现，待远端复评验证**：
   - 复用 `_data_preparation_with_filter(..., min_train_interactions=5)`；
   - target 只保留 warm；
   - candidate 只保留 train warm catalog；
   - 同时输出 MRR/NDCG/HR@10；
2. 对 Beauty/Toys 的 ID/TF/LLM/MV × 3 seeds，共 24 个 GTS checkpoint 复评；
3. 用同一 evaluator 复评 UniSRec Beauty 3 seeds（最好 Beauty+Toys 共 6 个）；
4. 用 min5 cohort 重跑 Beauty/Toys LOVO；
5. 所有表统一使用 sample std，并保存 per-seed 原值。

### P1：可选加强

1. Toys Early60；
2. paired significance / bootstrap CI；
3. 若实现真正 content-only scoring，再报告 unseen-target 模型比较；
4. 增加非 Amazon 数据集以强化外部有效性。

---

## 九、最终汇报口径

可以向老师明确汇报：

> 三个核心问题已经有一致的数据答案：  
> （1）Clean LOO 与 GTS 会改变绝对指标和模型排序；  
> （2）严格时间评测下 TF-IDF 最稳定，LLM/MV 没有稳定额外收益；  
> （3）MV 的低边际收益与 view redundancy 和 temporal shift 一致。  
> 但在冻结论文最终主表前，还需把 warm-only、UniSRec 和 LOVO 用 min5 + train-warm candidate 的统一 evaluator 复评。该工作不需要重训，只是统一评测口径。

---

## 十、主要数据源

- `paper_recsys/0723zhidao.txt`
- `logs/stopgate_3seed_beauty_metrics.csv`
- `logs/wsdm_wrapup_log10.csv`
- `logs/wsdm_wrapup_toys_gts.csv`
- `logs/beauty_loo_results.csv`
- `paper_recsys/beauty_loo_results.csv`（已同步的权威副本）
- `logs/toys_loo_results.csv`
- `paper_recsys/toys_loo_results.csv`（已补齐 12/12）
- `logs/early60_stopgate_beauty.csv`
- `logs/warm_only_eval.csv`
- `logs/unisrec_gts_results_corrected.csv`
- `paper_recsys/wsdm_wrapup_log10.csv`（Beauty ID 3-seed）
- `paper_recsys/warm_target_audit_amazon_{beauty,toys_and_games}.json`
- `paper_recsys/distribution_shift_{beauty,toys}_formal.json`
- `paper_recsys/view_redundancy_{beauty,toys}.json`
- `paper_recsys/leave_one_view_out_{beauty,toys}.md`
- `paper_recsys/rank_transition_beauty_{tf,llm,mv}.md`

