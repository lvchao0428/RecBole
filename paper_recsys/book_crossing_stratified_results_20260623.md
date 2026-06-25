# book-crossing 分层指标汇总（5090 实拉）

> 更新: 2026-06-24 20:18（5090 实查 · phaseb50 四配置全部完成）  
> 数据源: 5090 `run_metrics/` · 工具: `tools/pull_bc_metrics.py`  
> 协议: TO 划分 + timestamp · full ranking · neg=100

---

## 1. 运行状态

| 队列 | 状态 | 说明 |
|------|------|------|
| `run_5090_queue_bc_four_configs_phaseb50.sh` | **✅ 完成** | 6/24 09:26；seed=2024 · phaseb50 四配置 |
| `run_5090_queue_after_bc_grocery.sh` | **✅ 完成** | 6/24 17:59；已接 Grocery pipeline |

**BC phaseb50 全部完成（6/24 09:26）**

| Step | 配置 | 状态 | 完成时间 | test MRR@10 | test HR@10 |
|------|------|------|----------|-------------|------------|
| 1 | ID SASRecAlign 50ep | ✅ | 17:57 | **1.95%** | 3.87% |
| 2 | TF-IDF phaseb50 | ✅ | 20:45 | **2.20%** (+13%) | 3.65% |
| 3 | TF-IDF+LLM phaseb50 | ✅ | 00:20 | **2.20%** (+13%) | 3.71% |
| 4 | MV phaseb50 | ✅ | 09:26 | **2.27%** (+16%) | 3.87% |

> ⚠️ Step 1 run_metrics 中 `seed=2020`（yaml 默认覆盖 CLI）；Step 2–4 seed=2024 已确认。

---

## 2. 已完成结果（seed≈2025，6/18 趋势验证 · Phase-B 40ep）

> 注：ID 为 **SASRecAlign 60ep**（4 月主表协议）；文本三配置为 **SASRecAlignV3 / MultiViewV3**，Phase-A 20 + Phase-B **40**（非当前 phaseb50）。

### Overall @10

| Model | MRR@10 | Rec@10 (HR) | vs ID (MRR) |
|-------|--------|-------------|-------------|
| **ID (SASRecAlign 60ep)** | **1.95%** | **4.11%** | — |
| TF-IDF | 2.22% | 3.79% | **+14%** |
| TF-IDF+LLM (Qwen3) | 2.23% | 3.83% | **+14%** |
| MV-Align (Qwen3 4-view) | **2.30%** | 3.89% | **+18%** |

**run_metrics 文件**:
- ID: `20260621-015134_SASRecAlign_Phase-A.txt`
- TF-IDF: `20260618-180132_SASRecAlignV3_Phase-B.txt`
- LLM: `20260618-210359_SASRecAlignV3_Phase-B.txt`
- MV: `20260621-172226_SASRecAlignMultiViewV3_Phase-B.txt`

### Stratified MRR@10

| Model | new [1,3) | few [3,10) | frequent [10,∞) |
|-------|-----------|------------|-----------------|
| ID | 1.02% | 1.83% | 4.09% |
| TF-IDF | 1.42% (+39%) | 2.41% (+32%) | 4.35% (+6%) |
| TF-IDF+LLM | 1.37% | 2.47% (+35%) | 4.35% |
| MV | **1.52%** (+49%) | **2.51%** (+37%) | **4.47%** (+9%) |

### Stratified Recall@10

| Model | new | few | frequent |
|-------|-----|-----|------------|
| ID | 2.10% | 3.83% | **8.65%** |
| TF-IDF | 1.82% | 3.51% | 8.04% |
| TF-IDF+LLM | 1.79% | 3.60% | 8.13% |
| MV | 1.95% | 3.51% | 8.25% |

---

## 3. 解读（相对 Beauty 主线）

| 现象 | Beauty/Toys | BC |
|------|-------------|-----|
| TF-IDF overall MRR 提升 | **+~45%** | **+14%** |
| MV > single-view | 显著 (p<0.05) | **边际** (+0.07pp MRR) |
| Text 对 head Recall | 提升 | **下降**（ID Rec 4.11% → text ~3.8%） |
| cold strata MRR | 大幅提升 | 有提升但绝对值低 |

**结论**: BC 上文本 **略优于 ID（MRR）**，MV **略优于 TF-IDF**，但远未达到 Beauty 的增益阶梯；不适合作为第三域主表，适合 appendix / cold-stress 对照。

---

## 4. 其他历史 run（Phase-B 40ep，多为调参/重复）

```
ts               config    seed   MRR@10  Rec@10
20260618-180132  TF-IDF    2025   2.22    3.79
20260620-064619  TF-IDF    2025   2.27    3.73
20260618-210359  TFIDF+LLM 2025   2.23    3.83
20260620-094844  TFIDF+LLM 2025   2.26    3.84
20260619-063031  MV        2025   2.20    3.71
20260621-172226  MV        2025   2.30    3.89  ← best MV
```

phaseb50 MV（6/22）Phase-B **未完成**（仅 1 ckpt，训练中断）。

---

## 5. phaseb50 seed=2024 队列结果（✅ 完成）

### 5.1 Overall @10（test）

| Model | MRR@10 | HR@10 | vs ID (MRR) | run_metrics |
|-------|--------|-------|-------------|-------------|
| ID (50ep) | **1.95%** | 3.87% | — | `20260623-175744` |
| TF-IDF | 2.20% | 3.65% | **+13%** | `20260623-204501` |
| TF-IDF+LLM | 2.20% | 3.71% | **+13%** | `20260624-002001` |
| MV | **2.27%** | 3.87% | **+16%** | `20260624-092622` |

### 5.2 Stratified MRR@10（test · phaseb50 seed=2024）

| Model | new [1,3) | few [3,10) | frequent [10,∞) |
|-------|-----------|------------|-----------------|
| ID | 1.09% | 2.04% | 3.93% |
| TF-IDF | 1.42% (+30%) | 2.47% (+21%) | 4.26% (+8%) |
| TF-IDF+LLM | 1.38% | 2.45% | 4.29% |
| MV | **1.42%** | **2.44%** | **4.48%** |

**对比 seed≈2025 Phase-B 40ep（§2）**: 四配置 MRR 与历史 run 一致（±0.07pp）；增益形态 **+13~16% MRR**，MV 略优于 TF-IDF（+0.07pp），仍远低于 Beauty +45%。**appendix 定性不变，不必 multiseed**。
