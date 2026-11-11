# Experiment Status — 2026-07-30

> 更新时间: 18:20  
> 上一日: [`experiment_status_20260729.md`](experiment_status_20260729.md)

---

## 〇、一句话

**全部实验 12/12 完成。** UniSRec Toys seed2024/2026 隔夜补齐（7/30 01:07）。所有主表数据到齐，可以正式启动论文撰写。

---

## 一、UniSRec GTS Baseline（✅ 6/6 完成）

| tag | dataset | seed | test MRR@10 |
|-----|---------|-----:|------------:|
| unisrec_beauty_gts_seed42 | Beauty | 42 | 0.0204 |
| unisrec_beauty_gts_seed2024 | Beauty | 2024 | 0.0206 |
| unisrec_beauty_gts_seed2026 | Beauty | 2026 | 0.0208 |
| unisrec_toys_gts_seed42 | Toys | 42 | 0.0162 |
| unisrec_toys_gts_seed2024 | Toys | 2024 | 0.0163 |
| unisrec_toys_gts_seed2026 | Toys | 2026 | 0.0164 |

| Dataset | **mean ± std** |
|---------|---------------:|
| Beauty | **0.0206 ± 0.0002** |
| Toys | **0.0163 ± 0.0001** |

---

## 二、完整主表（论文 ready）

### GTS Protocol — MRR@10 test, 3-seed mean ± std

| Dataset | ID-only | TF-IDF | LLM (Qwen) | MV (4-view) | UniSRec |
|---------|--------:|-------:|------------:|------------:|--------:|
| Beauty | 0.0106±0.0003 | **0.0165±0.0003** | 0.0161±0.0002 | 0.0160±0.0003 | 0.0206±0.0002 |
| Toys | 0.0070±0.0001 | **0.0113±0.0003** | 0.0107±0.0004 | 0.0109±0.0006 | 0.0163±0.0001 |

**关键观察**：
- UniSRec（直接用 full LLM embedding 作 item representation）在 GTS 下显著强于所有 align 方案
- 在 align 方案内，TF-IDF ≥ LLM ≈ MV（简单特征更鲁棒）
- 文本特征 vs ID-only 提升显著（+50–60%）

### LOO Protocol — MRR@10 test, 3-seed mean ± std

| Dataset | ID-only | TF-IDF | LLM | MV |
|---------|--------:|-------:|----:|---:|
| Beauty | 0.0175±0.0004 | 0.0251±0.0002 | **0.0254±0.0004** | 0.0253±0.0004 |
| Toys | 0.0152±0.0005 | 0.0247±0.0006 | **0.0258±0.0012** | 0.0258±0.0028 |

**协议对比**：GTS 下 TF 最优，LOO 下 LLM/MV 反超 → **评估协议决定模型排序**

---

## 三、机制分析汇总

| 分析 | 关键结论 |
|------|----------|
| View redundancy (CKA) | inter-view CKA 0.84–0.94，完全冗余 |
| Effective rank | 4-view concat 136.9/256 = 53.5%（维度坍缩） |
| Leave-one-view-out | Δ = 0.000000（去掉任何 view 无影响） |
| TF-IDF vs views CKA | 0.25–0.31（TF-IDF 提供正交信息） |
| TF-IDF effective rank | 253.1/256 = 98.9%（信息密度高） |
| Early60 stop-gate | TF ≈ LLM ≥ MV，两窗口一致 |
| Cross ±analysis | 无稳定增益，rank improved ≈ degraded |
| Distribution shift | valid→test 下降 ~44%，系统性时间漂移 |

---

## 四、全局实验完成度 ✅

| 实验 | 状态 |
|------|------|
| Beauty GTS 3-seed (ID/TF/LLM/MV) | ✅ |
| Toys GTS 3-seed (ID/TF/LLM/MV) | ✅ |
| Beauty LOO 3-seed (ID/TF/LLM/MV) | ✅ |
| Toys LOO 3-seed (ID/TF/LLM/MV) | ✅ |
| Beauty Early60 stop-gate 3-seed | ✅ |
| Beauty Early60 +Cross 1-seed | ✅ |
| UniSRec GTS Beauty 3-seed | ✅ |
| UniSRec GTS Toys 3-seed | ✅ |
| View redundancy (CKA + eff rank) | ✅ |
| Leave-one-view-out | ✅ |
| Distribution shift | ✅ |
| Rank transition / Top-K share | ✅ |
| Warm-target audit | ✅ |

**全部到齐。下一步：论文撰写。**

---

## 五、机器快照

| 机器 | 状态 |
|------|------|
| **5090** | GPU 被 qwen-sft-v39 占用（15 GB）；RecBole 实验全部完成 |
| **本机** | CSV/文档已同步；可开始写论文 |
