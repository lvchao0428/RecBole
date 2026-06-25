# 5090 实验状态梳理（2026-06-25）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-25 17:10（5090 实查 · 本地+5090 已同步）  
> **GPU 状态**: RTX 5090 **训练中**（Grocery TF-IDF seed=42，util ~77%，显存 ~7GB）  
> **当前主线**: UniSRec Beauty ✅ → **Grocery multiseed** 🔄（seed42 TF-IDF 进行中）

**相关文档**:
- 结果汇总: [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)
- **分析与规划**: [`experiment_plan_20260625.md`](experiment_plan_20260625.md)
- 导师意见: [`0625zhidao.txt`](0625zhidao.txt) · [`0617zhidao.txt`](0617zhidao.txt) · [`0618zhidao.txt`](0618zhidao.txt)
- 前日状态: [`experiment_status_20260624.md`](experiment_status_20260624.md)

---

## 1. 进度总览

```
[✅] Beauty/Toys 主表 (SASRec, 4 seeds)
[✅] BC phaseb50 seed=2024
[✅] Grocery seed=2024 四配置
[✅] UniSRec Beauty 三配置 seed=2024     6/25 06:36
[🔄] Grocery multiseed (2025/2026/42)   9/12 完成，seed42 TF-IDF 运行中
[📋] GRU4Rec 第二 backbone              pipeline 结束后
```

| 实验 | 状态 | 完成/预计 |
|------|------|-----------|
| UniSRec Beauty (Base/AlignV3/AlignMV) | ✅ | 6/25 06:36 |
| Grocery multiseed seed=2025 ×4 | ✅ | 6/25 11:42 |
| Grocery multiseed seed=2026 ×4 | ✅ | 6/25 16:48 |
| Grocery multiseed seed=42 | 🔄 1/4 | ID ✅ 17:08 · TF-IDF 运行中 |
| 主流水线 | 🔄 | `run_5090_main_pipeline_20260625_resume.sh` |

**日志**: `logs/main_pipeline_20260625.log` · Grocery: `logs/grocery_*_seed*.log`

---

## 2. UniSRec Beauty vs SASRec MV（seed=2024 · 主文外部 baseline）

| 模型 | 角色 | MRR@10 | HR@10 | NDCG@10 |
|------|------|--------|-------|---------|
| SASRec ID-only | 内部 ablation | 2.16% | 5.31% | 2.91% |
| **SASRec MV-Align (7B)** | **主方法** | **3.18%** | **5.79%** | **3.79%** |
| **UniSRec Base** | **外部 baseline** | **2.47%** | **6.60%** | **3.44%** |
| UniSRec AlignV3 | portability | 3.25% | 6.52% | 4.02% |
| UniSRec AlignMVV3 | portability | 3.33% | 6.67% | 4.11% |

**写作口径**（`0625zhidao.txt`）: 主文对比 **MV-Align vs UniSRec Base**；AlignV3/MV 仅 appendix portability。

详见 [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md) § Beauty 对比表。

---

## 3. Grocery multiseed 进展（test @10）

seed=2024 见 [`experiment_summary_20260624.md`](experiment_summary_20260624.md)。

### seed=2025 / 2026（已完成 · test）

| Config | seed=2025 MRR@10 | seed=2026 MRR@10 | HR@10 (2025/2026) |
|--------|------------------|------------------|-------------------|
| TF-IDF | 2.92% | 2.92% | 5.77% / 5.73% |
| TF-IDF+LLM | 2.94% | 2.96% | 5.81% / 5.81% |
| MV | **3.03%** | **3.02%** | 6.06% / 6.06% |

与 seed=2024（MV 3.01%）一致，multiseed 形态稳定 ✅

### seed=42（进行中）

| Step | 状态 | 时间 |
|------|------|------|
| ID-only | ✅ | 6/25 17:08 |
| TF-IDF | 🔄 Phase-A/B | 17:08 起 |
| TF-IDF+LLM | ⏳ | — |
| MV | ⏳ | — |

**预计**: seed=42 剩余 ~5h → pipeline 今晚 ~22:00 前后完成

---

## 4. 5090 监控命令

```bash
ssh charlie@www.ultrapp.online
cd /home/charlie/project/RecBole
tail -f logs/main_pipeline_20260625.log
tail -f logs/grocery_TF-IDF_seed42.log
pgrep -af run_5090; nvidia-smi
```

---

## 5. 下一步（对齐导师优先级）

详见 [`experiment_plan_20260625.md`](experiment_plan_20260625.md)（0617/0618/0625 指标清单 + 实验规划）。

1. **等 Grocery multiseed 跑完** → 4-seed mean±std
2. **P1 分析**（不占 GPU）：cross concentration · gate 分桶 · case study · Coverage · paired t-test
3. **P2 可选训练**：UniSRec Base Toys → GRU4Rec sanity

**文档同步**:

```bash
rsync -avz paper_recsys/*.md charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/
```
