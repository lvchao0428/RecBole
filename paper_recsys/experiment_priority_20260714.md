# 今日优先级 — 2026-07-14

> 更新时间: 2026-07-14 15:50  
> 依据: `0712zhidao.txt` + `experiment_status_20260713.md`  
> 状态: **双机并行** — 5090 跑 MV(4/4)；log10 跑 UniSRec Base epoch~13/50

---

## 〇、进展快照（15:50）

| 机器 | 任务 | 状态 | ETA |
|------|------|:----:|-----|
| **5090** | boost+no-Cross | 🔄 **[4/4] MV Phase-B ep~37/50** | **~16:20** |
| **log10** | UniSRec Beauty | 🔄 Base ep~13/50（~14min/ep，慢） | Base ~00:00；全套过夜 |

### boost+no-Cross 已出结果（test MRR@10）

| 配置 | MRR@10 | NDCG@10 | R@10 | R_new@10 | 状态 |
|------|:------:|:------:|:----:|:--------:|:----:|
| ID-only | 0.0112 | 0.0151 | 0.0277 | 0.0137 | ✅ |
| TF no-Cross + cb3+infer | 0.0160 | 0.0202 | 0.0340 | 0.0178 | ✅ |
| **LLM no-Cross + cb3+infer** | **0.0171** | **0.0213** | **0.0350** | **0.0270** | ✅ |
| MV no-Cross + cb3+infer | — | — | — | — | 🔄 |

**初步结论（MV 待补）**：boost+no-Cross 下暂为 **LLM > TF > ID**；LLM 创本协议最高 MRR=0.0171。经典 MV>LLM>TF **尚未恢复**（且 MV 待出）。

---

## 一、四象限对照（Beauty SASRec TS V2, seed=2025, min5）

| | **no-Cross** | **+Cross** |
|--|:------------:|:----------:|
| **no-boost** | LLM **0.0168** > TF 0.0157 > MV 0.0141 | 三者打平 **0.0157** |
| **cb3+infer** | LLM **0.0171** > TF 0.0160 > MV 🔄 | LLM 0.0164 > TF 0.0160 > MV 0.0152 |

要点：
1. **Cross 不是稳定净收益**：no-boost 下 LLM 去 Cross 更好；boost 下亦是 no-Cross LLM 更高（0.0171 > 0.0164）
2. **boost 略抬 LLM/TF**，但未恢复 MV 主导层级
3. **TF-IDF 仍是强 baseline**（与 LLM 差距 ≤0.0011）

---

## 二、机器分工

| 机器 | GPU | 适合任务 | 当前 |
|------|-----|----------|------|
| **5090** | 32GB | MV / boost+LLM | 🔄 MV fb_nc |
| **log10** | 11GB | ID-only / UniSRec / 小 batch | 🔄 UniSRec Base |

结果回收：`bash scripts/sync_log10_results_to_5090.sh`（boost 结束后自动 wait+sync）

---

## 三、未跑完清单

### P0（今日）
| # | 实验 | 状态 |
|---|------|:----:|
| 1 | boost+no-Cross MV | 🔄 ~16:20 |
| 2 | 四象限表定稿（补 MV） | ⏳ 等 #1 |

### P1
| # | 实验 | 状态 |
|---|------|:----:|
| 3 | UniSRec Beauty ×3 | 🔄 log10（比预期慢 3–4×） |
| 4 | boost+Cross 结果入正式表 | 可写 |

### P2+
Toys/Grocery 3-seed、公平协议重跑、机制分析、Book-Crossing、外部基线

---

## 四、今日时间线（修订）

| 时刻 | 事件 |
|------|------|
| ✅ 12:05 / 12:26 | 5090 boost+no-Cross / log10 UniSRec 启动 |
| ✅ 12:29 / 13:21 / 14:15 | ID / TF / LLM 完成 |
| **~16:20** | MV 完成 → **四象限齐** |
| 晚间 | 定稿：boost/Cross 是否进主方法 |
| 过夜 | UniSRec Align + MV（log10） |

监控：
```bash
tail -f logs/beauty_full_boost_nocross_v2.log
ssh charlie@192.168.0.107 'tail -f ~/project/RecBole/logs/ts_unisrec_base_seed2025.log'
```
