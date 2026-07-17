# 今日优先级 — 2026-07-15（夜）

> 更新时间: 2026-07-15 22:55  
> 详况: `experiment_status_20260715.md`  
> **7.16 全日规划（时刻表+旧数+预期）→ [`experiment_plan_20260716.md`](experiment_plan_20260716.md)**  
> 状态: **5090 跑 MV TF∥views [1/3] Phase-B**；log10 空闲

---

## 〇、快照（22:55）

| 机器 | 任务 | 状态 | ETA |
|------|------|:----:|-----|
| **5090** | MV fair 复跑（TF∥views=512） | 🔄 **[1/3]** no-Cross no-boost Phase-B | 组1~00:30；3 组~明早 |
| **log10** | — | 空闲 | 可排轻量任务 |

---

## 一、今夜（P0）— MV 按优先级复跑

脚本：`run_5090_mv_llm_parity.sh`（已在跑，勿重复启）

| 序 | 优先级 | 实验 | 对照 | 状态 |
|:--:|:------:|------|------|:----:|
| 1 | **P0** | MV no-Cross, no-boost | LLM 0.0168 / TF 0.0157 | 🔄 |
| 2 | **P0** | MV no-Cross, **cb3+infer** | LLM **0.0171** / TF 0.0160 | ⏳ |
| 3 | **P1** | MV +Cross, no-boost | Cross 象限 | ⏳ |

验收：`raw_concat_dim=512 (base=256 + views=256), mv_include_base=True, SENet=removed`

**不做**：继续沿用旧 aligned / concatfix / 仅 views 数字进主表。

---

## 二、明日计划（07-16）

### 上午（出数后立刻）

| 优先级 | 任务 | 说明 |
|:------:|------|------|
| **P0** | 收 3 组 test MRR，填四象限 MV 格 | 重点看 #2 vs LLM 0.0171 |
| **P0** | 定稿：层级是否恢复 / 主方法是否仍 LLM no-Cross | 写进 status |
| P1 | 同步 docs → 5090/log10 | `scripts/sync_code_three_machines.sh` |

### 下午 / 空闲算力（按需）

| 优先级 | 任务 | 机器 | 估时 |
|:------:|------|:----:|:----:|
| **P1** | Toys/Grocery 公平协议（同 two-phase + TF∥views MV） | 5090 | 6–8h |
| P1 | UniSRec MV-V3 移 5090（log10 OOM） | 5090 | ~2–3h |
| P2 | Beauty LLM no-Cross 3-seed | 5090/log10 | ~6h |
| P2 | 单 view 消融 `text_view_indices=[0..3]` | 5090 | ~8h |
| P2 | 机制分析（rank-transition / entropy） | 本机 | 半天 |

### 建议决策树

```
新 MV #2 (boost+no-Cross) MRR ?
  ├─ ≥ LLM 0.0171  → 主叙事可回到 MV；补 Toys/Grocery
  ├─ ≈ TF～LLM     → 维持「LLM no-Cross 最优」；MV 作公平对照
  └─ 仍明显低于 LLM → 不追超参；转 Toys 域 + 机制分析
```

---

## 三、对照锚点（不变）

| 配置 | MRR@10 |
|------|:------:|
| ID-only | 0.0112 |
| TF no-Cross no-boost | 0.0157 |
| TF no-Cross boost | 0.0160 |
| LLM no-Cross no-boost | **0.0168** |
| **LLM no-Cross boost** | **0.0171** ← 当前最优 |

---

## 四、一句话

**今夜只盯 MV TF∥views 三组；明早填表定主方法，再开 Toys/Grocery 或 UniSRec MV。**
